import matplotlib.pyplot as plt
import numpy as np
from lbm_solver import run_simulation
import os

# Check if we have optimization results
if not os.path.exists("best_params.txt"):
    print("Please run optimize.py first!")
    exit()

# Load best parameters
with open("best_params.txt", "r") as f:
    data = f.read().split(',')
    m, p, t = float(data[0]), float(data[1]), float(data[2])

print(f"Visualizing Best Profile: m={m:.3f}, p={p:.3f}, t={t:.3f}")
print("Running High-Res Simulation (this takes a moment)...")

# Run a longer simulation for better visuals
velocity_field = run_simulation(m, p, t, iterations=1000, return_plot_data=True)

# Plotting
plt.figure(figsize=(10, 4))
plt.imshow(velocity_field, cmap='jet', origin='lower')
plt.colorbar(label='Velocity Magnitude')
plt.title(f"Optimized Flow: Camber {m:.2f}, Pos {p:.2f}, Thick {t:.2f}")
plt.xlabel("Tunnel Length")
plt.ylabel("Tunnel Height")

# Add the blade shape outline for clarity
from airfoil_gen import get_mask
mask = get_mask(m, p, t, 300, 100, angle_of_attack=5)
# Overlay mask
plt.imshow(mask, cmap='gray', alpha=0.3, origin='lower')

plt.tight_layout()
plt.savefig("optimized_blade_flow.png", dpi=150)
print("Saved visualization to 'optimized_blade_flow.png'")
plt.show()
