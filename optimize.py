from bayes_opt import BayesianOptimization
from lbm_solver import run_simulation
import time

print("--- Starting Neural Wind Tunnel ---")

def optimize_blade(m, p, t):
    """
    Objective Function for BayesOpt.
    """
    # 1. Physical Constraints
    # If blade is too thin, it's structurally impossible (penalty)
    if t < 0.08: return -10.0 
    
    # 2. Run Physics
    # We use fewer iterations for the search phase (speed)
    score = run_simulation(m, p, t, iterations=300)
    
    return score

# Define the search space (NACA parameters)
# m (camber): 0% to 8%
# p (position): 20% to 60%
# t (thickness): 8% to 20%
pbounds = {'m': (0.0, 0.08), 'p': (0.2, 0.6), 't': (0.08, 0.20)}

optimizer = BayesianOptimization(
    f=optimize_blade,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

start_time = time.time()
# init_points: Random exploration steps
# n_iter: Optimization steps
optimizer.maximize(init_points=3, n_iter=10)

print(f"\nOptimization finished in {time.time()-start_time:.2f} seconds.")
print("Best Blade Profile Found:")
print(optimizer.max)

# Save best parameters to file for the visualizer
best_params = optimizer.max['params']
with open("best_params.txt", "w") as f:
    f.write(f"{best_params['m']},{best_params['p']},{best_params['t']}")
