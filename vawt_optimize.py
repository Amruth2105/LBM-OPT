"""
VAWT Optimizer
Bayesian Optimization for VAWT blade parameters.
Optimizes: blade thickness, number of blades, and optionally TSR.
"""
from bayes_opt import BayesianOptimization
from vawt_solver import run_vawt_simulation
import numpy as np
import time

print("=" * 60)
print("VAWT Neural Optimization - Bayesian Search")
print("=" * 60)


# Global counter for progress
iteration_count = 0


def optimize_vawt(thickness, n_blades_float, tsr):
    """
    Objective function for Bayesian Optimization.
    Maximizes power coefficient Cp.
    """
    global iteration_count
    iteration_count += 1
    
    # Convert n_blades to integer (BayesOpt only handles floats)
    n_blades = int(round(n_blades_float))
    n_blades = max(2, min(5, n_blades))  # Clamp to 2-5 blades
    
    # Physical constraints
    if thickness < 0.10:
        return -1.0  # Too thin - structurally impossible
    if thickness > 0.25:
        return -0.5  # Too thick - high drag
    
    # Run simulation
    # Use moderate iterations for search (balance speed vs accuracy)
    Cp = run_vawt_simulation(
        thickness=thickness,
        n_blades=n_blades,
        tsr=tsr,
        iterations=600,  # Higher for accuracy as requested
        verbose=False
    )
    
    print(f"  [{iteration_count:3d}] t={thickness:.3f}, N={n_blades}, TSR={tsr:.2f} → Cp={Cp:.4f}")
    
    return Cp


# Define search space
# thickness: 10% to 22% (NACA 0010 to NACA 0022)
# n_blades: 2 to 5 (will be rounded to int)
# tsr: 2.0 to 6.0 (typical operating range)
pbounds = {
    'thickness': (0.10, 0.22),
    'n_blades_float': (2.0, 5.0),
    'tsr': (2.0, 6.0)
}

optimizer = BayesianOptimization(
    f=optimize_vawt,
    pbounds=pbounds,
    random_state=42,
    verbose=0  # We handle our own printing
)

print("\nSearch Space:")
print(f"  Thickness: {pbounds['thickness'][0]:.2f} - {pbounds['thickness'][1]:.2f}")
print(f"  N Blades:  {int(pbounds['n_blades_float'][0])} - {int(pbounds['n_blades_float'][1])}")
print(f"  TSR:       {pbounds['tsr'][0]:.1f} - {pbounds['tsr'][1]:.1f}")
print("-" * 60)

start_time = time.time()

# Run optimization
# init_points: random exploration
# n_iter: Bayesian optimization steps
print("\nPhase 1: Random Exploration")
optimizer.maximize(init_points=5, n_iter=0)

print("\nPhase 2: Bayesian Optimization")
optimizer.maximize(init_points=0, n_iter=15)

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print(f"Optimization Complete! ({elapsed:.1f} seconds)")
print("=" * 60)

# Extract best result
best = optimizer.max
best_params = best['params']
best_n_blades = int(round(best_params['n_blades_float']))

print("\nOptimal VAWT Configuration:")
print(f"  NACA Profile:  00{int(best_params['thickness']*100):02d}")
print(f"  Thickness:     {best_params['thickness']*100:.1f}%")
print(f"  Number Blades: {best_n_blades}")
print(f"  Tip Speed Ratio: {best_params['tsr']:.2f}")
print(f"  Power Coefficient Cp: {best['target']:.4f}")

# Save best parameters
with open("vawt_best_params.txt", "w") as f:
    f.write(f"{best_params['thickness']},{best_n_blades},{best_params['tsr']}")

print("\nSaved optimal parameters to 'vawt_best_params.txt'")
print("Run 'python3 vawt_visualize.py' to see the flow animation!")
