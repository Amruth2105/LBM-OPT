"""
2D LBM Solver for Wind Turbine Blade Optimization
Optimizes for TANGENTIAL FORCE (torque generation), not L/D.
Uses D2Q9 lattice with proper lift/drag calculation.
"""
import numpy as np
from airfoil_gen import get_mask

def run_simulation(m, p, t, iterations=400, return_plot_data=False, 
                   angle_of_attack=8, mode='turbine'):
    """
    Runs LBM simulation for a wind turbine blade section.
    
    Parameters:
        m, p, t: NACA 4-digit parameters
        angle_of_attack: Typical turbine blade operates at 5-12 degrees
        mode: 'turbine' (maximize tangential force) or 'aircraft' (maximize L/D)
    
    Returns: 
        Score based on optimization mode
    """
    # --- Simulation Constants ---
    Nx, Ny = 400, 150       # Grid size (higher res for accuracy)
    tau = 0.56              # Relaxation time
    omega = 1 / tau
    u_inf = 0.08            # Freestream velocity
    
    # D2Q9 lattice
    cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    
    # --- Initialize Fluid ---
    F = np.ones((Ny, Nx, 9))
    # Set initial velocity to freestream
    for i in range(9):
        cu = 3 * (cxs[i] * u_inf)
        F[:, :, i] = weights[i] * (1 + cu + 0.5 * cu**2 - 1.5 * u_inf**2)
    
    # --- Create Blade ---
    obstacle = get_mask(m, p, t, Nx, Ny, angle_of_attack=angle_of_attack)
    
    # Store pre-bounce F for force calculation
    Fx_total, Fy_total = 0.0, 0.0
    force_samples = 0
    
    # --- Main Loop ---
    for it in range(iterations):
        # Store for momentum exchange
        F_pre = F.copy()
        
        # 1. Streaming
        for i, cx, cy in zip(range(9), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
        
        # 2. Bounce-back on blade
        bndryF = F[obstacle, :]
        bndryF = bndryF[:, opposite]
        F[obstacle, :] = bndryF
        
        # 3. Compute macroscopic quantities
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho
        
        # 4. Inlet BC
        ux[:, 0] = u_inf
        uy[:, 0] = 0.0
        rho[:, 0] = 1.0
        
        # 5. Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy in zip(range(9), cxs, cys):
            cu = 3 * (cx * ux + cy * uy)
            Feq[:, :, i] = rho * weights[i] * (1 + cu + 0.5 * cu**2 - 1.5 * (ux**2 + uy**2))
        F += -omega * (F - Feq)
        
        # 6. Calculate forces (momentum exchange) after warmup
        if it > iterations // 2:
            # Force = momentum change at boundary
            dF = F[obstacle, :] - F_pre[obstacle, :]
            Fx = np.sum(dF * cxs)
            Fy = np.sum(dF * cys)
            Fx_total += Fx
            Fy_total += Fy
            force_samples += 1
    
    # Average forces
    if force_samples > 0:
        Fx_avg = Fx_total / force_samples
        Fy_avg = Fy_total / force_samples
    else:
        Fx_avg, Fy_avg = 0, 0
    
    # --- Convert to Lift/Drag ---
    # In our setup: flow is in +x, so:
    # Drag = force in x direction (opposite to flow)
    # Lift = force in y direction (perpendicular to flow)
    drag = -Fx_avg  # Positive drag opposes flow
    lift = -Fy_avg  # Convention: positive lift is upward
    
    # Chord length for normalization
    chord = int(Nx * 0.2)
    dynamic_pressure = 0.5 * 1.0 * u_inf**2 * chord
    
    # Coefficients
    if dynamic_pressure > 1e-10:
        Cl = lift / dynamic_pressure
        Cd = drag / dynamic_pressure
    else:
        Cl, Cd = 0, 0.01
    
    # Ensure Cd is positive and non-zero
    Cd = max(abs(Cd), 0.001)
    
    if return_plot_data:
        velocity = np.sqrt(ux**2 + uy**2)
        return velocity, Cl, Cd
    
    # --- Scoring for Wind Turbine ---
    if mode == 'turbine':
        # For wind turbine blade, we want HIGH LIFT with acceptable drag
        # Tangential force coefficient at typical inflow angle (8-12 degrees)
        phi = np.radians(angle_of_attack + 5)  # Inflow angle
        Ct = Cl * np.sin(phi) - Cd * np.cos(phi)  # Tangential force coeff
        
        # Also penalize very high drag
        ld_ratio = Cl / Cd if Cd > 0.001 else 0
        
        # Combined score: prioritize Ct but also consider structural viability
        # Thicker blades (t > 0.15) get a small bonus for structural strength
        thickness_bonus = 0.1 if t > 0.15 else 0
        
        score = Ct + 0.01 * ld_ratio + thickness_bonus
        
        return score
    else:
        # Aircraft mode: pure L/D
        return Cl / Cd if Cd > 0.001 else 0


if __name__ == "__main__":
    print("=" * 50)
    print("2D Wind Turbine Blade LBM Solver - Test")
    print("=" * 50)
    
    # Test different profiles
    profiles = [
        (0.04, 0.40, 0.12, "NACA 4412 (common aircraft)"),
        (0.04, 0.40, 0.18, "NACA 4418 (thicker)"),
        (0.06, 0.30, 0.21, "NACA 6321 (high lift, thick)"),
        (0.02, 0.40, 0.15, "NACA 2415 (low camber)"),
        (0.00, 0.00, 0.18, "NACA 0018 (symmetric)"),
    ]
    
    print("\nTesting profiles at 8° AoA:")
    print("-" * 50)
    
    for m, p, t, name in profiles:
        score = run_simulation(m, p, t, iterations=300, angle_of_attack=8, mode='turbine')
        print(f"  {name}: Score = {score:.4f}")
    
    print("-" * 50)
    print("Higher score = better for wind turbine torque")
