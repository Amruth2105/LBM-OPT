"""
Wind Turbine Blade Geometry Optimizer
Bayesian Optimization for NACA 4-digit parameters.
Optimizes for maximum tangential force (torque generation).
"""
from bayes_opt import BayesianOptimization
from lbm_solver import run_simulation
import time

print("=" * 60)
print("Wind Turbine Blade Geometry Optimizer")
print("=" * 60)
print("Objective: Maximize tangential force coefficient (Ct)")
print("Method: 2D LBM + Bayesian Optimization")
print("=" * 60)

iteration_count = 0

def optimize_blade(m, p, t):
    """
    Objective function for wind turbine blade optimization.
    """
    global iteration_count
    iteration_count += 1
    
    # Physical constraints for turbine blades
    if t < 0.10:
        return -5.0  # Too thin for structural loads
    if t > 0.24:
        return -3.0  # Too thick, excessive drag
    if m > 0.08:
        return -2.0  # Excessive camber
    
    # Run at typical turbine blade operating angle (8 degrees)
    score = run_simulation(m, p, t, iterations=300, angle_of_attack=8, mode='turbine')
    
    print(f"  [{iteration_count:3d}] NACA {int(m*100):01d}{int(p*10):01d}{int(t*100):02d} → Score: {score:.4f}")
    
    return score


# Search space for wind turbine blade
# - Higher thickness (12-21%) for structural strength
# - Moderate camber (0-6%)
# - Camber position (20-50%)
pbounds = {
    'm': (0.00, 0.06),   # Camber: 0-6%
    'p': (0.20, 0.50),   # Max camber position: 20-50%
    't': (0.12, 0.21)    # Thickness: 12-21%
}

optimizer = BayesianOptimization(
    f=optimize_blade,
    pbounds=pbounds,
    random_state=42,
    verbose=0
)

print("\nSearch Space:")
print(f"  Camber (m):    {pbounds['m'][0]*100:.0f}% - {pbounds['m'][1]*100:.0f}%")
print(f"  Position (p):  {pbounds['p'][0]*100:.0f}% - {pbounds['p'][1]*100:.0f}%")
print(f"  Thickness (t): {pbounds['t'][0]*100:.0f}% - {pbounds['t'][1]*100:.0f}%")
print("-" * 60)

start_time = time.time()

print("\nPhase 1: Random Exploration (5 samples)")
optimizer.maximize(init_points=5, n_iter=0)

print("\nPhase 2: Bayesian Optimization (15 iterations)")
optimizer.maximize(init_points=0, n_iter=15)

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print(f"Optimization Complete! ({elapsed/60:.1f} minutes)")
print("=" * 60)

# Extract best result
best = optimizer.max
bp = best['params']
m_best = bp['m']
p_best = bp['p']
t_best = bp['t']

# NACA designation
naca_code = f"{int(m_best*100):01d}{int(p_best*10):01d}{int(t_best*100):02d}"

print("\nOptimal Wind Turbine Blade Profile:")
print(f"  NACA Code:  {naca_code}")
print(f"  Camber:     {m_best*100:.2f}%")
print(f"  Position:   {p_best*100:.1f}%")
print(f"  Thickness:  {t_best*100:.1f}%")
print(f"  Score:      {best['target']:.4f}")

# Save best parameters
with open("best_turbine_params.txt", "w") as f:
    f.write(f"{m_best},{p_best},{t_best}")

print("\nSaved to 'best_turbine_params.txt'")
print("Run 'python visualize.py' to see the flow field!")
