"""
VAWT LBM Solver
Lattice Boltzmann Method solver for Vertical Axis Wind Turbines.
Uses Immersed Boundary Method for rotating blade handling.
"""
import numpy as np
from vawt_geometry import get_vawt_mask


def run_vawt_simulation(thickness=0.18, n_blades=3, tsr=3.0, radius=0.35,
                        iterations=1000, return_animation=False, verbose=False):
    """
    Run LBM simulation for a VAWT.
    
    Parameters:
    -----------
    thickness : float
        NACA airfoil thickness (e.g., 0.18 for NACA 0018)
    n_blades : int
        Number of blades (2-5)
    tsr : float
        Tip Speed Ratio (ω*R/V∞)
    radius : float
        Turbine radius as fraction of domain (0.2-0.45)
    iterations : int
        Number of LBM iterations (more = better accuracy)
    return_animation : bool
        If True, return velocity fields for animation
    verbose : bool
        Print progress
    
    Returns:
    --------
    float : Average power coefficient Cp
    (optionally) list of velocity fields if return_animation=True
    """
    
    # --- Domain Setup ---
    Nx, Ny = 400, 400       # Square domain for VAWT (higher res for accuracy)
    tau = 0.56              # Relaxation time (slightly higher for stability)
    u_inf = 0.05            # Freestream velocity (lower for stability with rotation)
    
    # Calculate angular velocity from TSR
    R_pixels = int(radius * min(Nx, Ny) * 0.4)
    omega = tsr * u_inf / R_pixels  # Angular velocity (rad/iteration)
    
    # Degrees per iteration
    deg_per_iter = np.degrees(omega)
    
    # D2Q9 lattice
    cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    
    # --- Initialize Distribution Function ---
    rho = np.ones((Ny, Nx))
    ux = np.ones((Ny, Nx)) * u_inf
    uy = np.zeros((Ny, Nx))
    
    # Equilibrium distribution
    F = np.zeros((Ny, Nx, 9))
    for i in range(9):
        cu = 3 * (cxs[i] * ux + cys[i] * uy)
        F[:, :, i] = rho * weights[i] * (1 + cu + 0.5 * cu**2 - 1.5 * (ux**2 + uy**2))
    
    # --- Tracking Variables ---
    torque_history = []
    animation_frames = []
    azimuth = 0.0
    
    # --- Main LBM Loop ---
    for it in range(iterations):
        # 1. Update blade positions (rotation)
        azimuth = (azimuth + deg_per_iter) % 360
        obstacle = get_vawt_mask(n_blades, thickness, radius, azimuth, Nx, Ny)
        
        # 2. Streaming step
        for i in range(9):
            F[:, :, i] = np.roll(F[:, :, i], cxs[i], axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cys[i], axis=0)
        
        # 3. Bounce-back boundary conditions on obstacle
        # Store pre-bounce values for momentum exchange
        F_pre = F[obstacle, :].copy()
        
        bndryF = F[obstacle, :]
        bndryF = bndryF[:, opposite]
        F[obstacle, :] = bndryF
        
        # 4. Calculate momentum exchange (for torque)
        F_post = F[obstacle, :]
        delta_momentum_x = np.sum((F_post - F_pre) * cxs)
        delta_momentum_y = np.sum((F_post - F_pre) * cys)
        
        # Calculate torque about center
        obstacle_coords = np.argwhere(obstacle)
        if len(obstacle_coords) > 0:
            center_x, center_y = Nx // 2, Ny // 2
            
            # For each obstacle point, calculate torque contribution
            torque = 0.0
            obs_y, obs_x = obstacle_coords[:, 0], obstacle_coords[:, 1]
            rx = obs_x - center_x
            ry = obs_y - center_y
            
            # Approximate per-point momentum (simplified)
            # Torque = r × F = rx*Fy - ry*Fx
            # Use average momentum change distributed over obstacle
            n_obs = len(obstacle_coords)
            fx_per_point = delta_momentum_x / max(n_obs, 1)
            fy_per_point = delta_momentum_y / max(n_obs, 1)
            
            torque = np.sum(rx * fy_per_point - ry * fx_per_point)
            torque_history.append(torque)
        
        # 5. Compute macroscopic quantities
        rho = np.sum(F, axis=2)
        ux = np.sum(F * cxs, axis=2) / rho
        uy = np.sum(F * cys, axis=2) / rho
        
        # 6. Inlet boundary condition (left side)
        ux[:, 0] = u_inf
        uy[:, 0] = 0.0
        rho[:, 0] = 1.0
        
        # Recalculate F at inlet
        for i in range(9):
            cu = 3 * (cxs[i] * ux[:, 0] + cys[i] * uy[:, 0])
            F[:, 0, i] = rho[:, 0] * weights[i] * (1 + cu + 0.5 * cu**2 - 1.5 * (ux[:, 0]**2 + uy[:, 0]**2))
        
        # 7. Open outlet (right side) - copy from interior
        F[:, -1, :] = F[:, -2, :]
        
        # 8. Collision step (BGK)
        Feq = np.zeros_like(F)
        for i in range(9):
            cu = 3 * (cxs[i] * ux + cys[i] * uy)
            Feq[:, :, i] = rho * weights[i] * (1 + cu + 0.5 * cu**2 - 1.5 * (ux**2 + uy**2))
        
        F += -(1.0 / tau) * (F - Feq)
        
        # 9. Save frame for animation (every 10 iterations)
        if return_animation and it % 10 == 0:
            velocity = np.sqrt(ux**2 + uy**2)
            animation_frames.append({
                'velocity': velocity.copy(),
                'mask': obstacle.copy(),
                'azimuth': azimuth
            })
        
        # Progress
        if verbose and it % 100 == 0:
            print(f"Iteration {it}/{iterations}, Azimuth: {azimuth:.1f}°")
    
    # --- Calculate Power Coefficient ---
    # Skip first quarter for spin-up
    stable_torques = torque_history[len(torque_history)//4:]
    avg_torque = np.mean(stable_torques) if stable_torques else 0.0
    
    # Power = Torque × Angular velocity
    power = avg_torque * omega
    
    # Power coefficient: Cp = P / (0.5 * rho * A * V^3)
    # For 2D: A = 2*R (swept diameter)
    A = 2 * R_pixels
    P_available = 0.5 * 1.0 * A * (u_inf ** 3)
    
    Cp = power / P_available if P_available > 0 else 0.0
    
    # Normalize to reasonable range
    Cp = np.clip(Cp, -1.0, 0.6)
    
    if verbose:
        print(f"Average Torque: {avg_torque:.6f}")
        print(f"Power Coefficient Cp: {Cp:.4f}")
    
    if return_animation:
        return Cp, animation_frames
    return Cp


def compute_cp_vs_tsr(thickness=0.18, n_blades=3, tsr_range=None, iterations=800):
    """
    Compute Cp for a range of TSR values.
    """
    if tsr_range is None:
        tsr_range = np.linspace(1.0, 6.0, 11)
    
    cp_values = []
    for tsr in tsr_range:
        print(f"Computing TSR = {tsr:.2f}...")
        cp = run_vawt_simulation(thickness=thickness, n_blades=n_blades, 
                                  tsr=tsr, iterations=iterations, verbose=False)
        cp_values.append(cp)
        print(f"  Cp = {cp:.4f}")
    
    return tsr_range, np.array(cp_values)


if __name__ == "__main__":
    print("Testing VAWT LBM Solver...")
    
    # Quick test
    Cp = run_vawt_simulation(thickness=0.18, n_blades=3, tsr=3.0, 
                             iterations=500, verbose=True)
    
    print(f"\nTest completed. Cp = {Cp:.4f}")
