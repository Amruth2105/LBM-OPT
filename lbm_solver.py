import numpy as np
from airfoil_gen import get_mask

def run_simulation(m, p, t, iterations=400, return_plot_data=False):
    """
    Runs LBM simulation.
    Returns: Efficiency Score (Lift/Drag approximation)
    """
    # --- Simulation Constants ---
    Nx, Ny = 300, 100       # Grid size (Width, Height)
    tau = 0.53              # Relaxation time (Viscosity related)
    omega = 1 / tau
    
    # Lattice D2Q9 constants
    # Directions: C, E, N, W, S, NE, NW, SW, SE
    cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    
    # Indexes for bounce-back (opposite directions)
    # 0->0, 1->3, 2->4, 3->1, 4->2, 5->7, 6->8, 7->5, 8->6
    opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    
    # --- Initialize Fluid ---
    # Density (rho) is 1 everywhere initially
    F = np.ones((Ny, Nx, 9)) + 0.01 * np.random.randn(Ny, Nx, 9)
    # Initial velocity (ux=0.1 to right, uy=0)
    F[:,:,1] += 2.0 * 0.1
    F[:,:,5] += 2.0 * 0.1
    F[:,:,8] += 2.0 * 0.1
    
    # --- Create Obstacle ---
    # Angle of Attack set to 5 degrees for lift generation
    obstacle = get_mask(m, p, t, Nx, Ny, angle_of_attack=5)

    # --- Main Loop ---
    for it in range(iterations):
        # 1. Stream (Move particles)
        for i, cx, cy in zip(range(9), cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
        
        # 2. Boundary (Bounce-back)
        bndryF = F[obstacle, :]
        bndryF = bndryF[:, opposite]
        F[obstacle, :] = bndryF
        
        # 3. Collision (Relax to equilibrium)
        rho = np.sum(F, 2)
        ux  = np.sum(F * cxs, 2) / rho
        uy  = np.sum(F * cys, 2) / rho
        
        # Force inlet (Left side) to have constant velocity
        ux[:, 0] = 0.1
        uy[:, 0] = 0.0
        
        Feq = np.zeros(F.shape)
        for i, cx, cy in zip(range(9), cxs, cys):
            cu = 3 * (cx*ux + cy*uy)
            Feq[:,:,i] = rho * weights[i] * (1 + cu + 0.5*(cu**2) - 1.5*(ux**2 + uy**2))
            
        F += -(1.0/tau) * (F - Feq)

    # --- Calculate Metrics ---
    # We measure the momentum change in the fluid
    
    # Drag: Velocity deficit in the wake (Right side)
    inlet_velocity = np.mean(np.sqrt(ux[:, 10]**2 + uy[:, 10]**2))
    outlet_velocity = np.mean(np.sqrt(ux[:, -20]**2 + uy[:, -20]**2))
    drag_score = (inlet_velocity - outlet_velocity) 
    
    # Lift: Average vertical velocity component behind the blade (downwash implies lift)
    # Ideally, lift pushes air DOWN, so we look for negative Uy in wake
    lift_score = -1 * np.mean(uy[:, -20]) 
    
    # Avoid div/0
    if drag_score < 1e-6: drag_score = 1e-6
    
    score = lift_score / drag_score
    
    if return_plot_data:
        # Return velocity magnitude for plotting
        velocity = np.sqrt(ux**2 + uy**2)
        return velocity
        
    return score
