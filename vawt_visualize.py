"""
VAWT Visualization
Generates flow animation and Cp vs TSR performance curves.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from vawt_solver import run_vawt_simulation, compute_cp_vs_tsr
from vawt_geometry import get_vawt_mask

# Check for optimized parameters
if os.path.exists("vawt_best_params.txt"):
    with open("vawt_best_params.txt", "r") as f:
        data = f.read().split(',')
        thickness = float(data[0])
        n_blades = int(data[1])
        tsr = float(data[2])
    print(f"Loaded optimized parameters:")
    print(f"  Thickness: {thickness:.3f} (NACA 00{int(thickness*100):02d})")
    print(f"  Blades: {n_blades}")
    print(f"  TSR: {tsr:.2f}")
else:
    # Default values
    thickness = 0.18
    n_blades = 3
    tsr = 3.5
    print("No optimization results found. Using defaults:")
    print(f"  Thickness: {thickness:.3f}")
    print(f"  Blades: {n_blades}")
    print(f"  TSR: {tsr:.2f}")

print("\n" + "=" * 60)
print("Generating VAWT Flow Animation...")
print("=" * 60)

# Run high-resolution simulation for animation
print("Running simulation (this may take a few minutes)...")
Cp, frames = run_vawt_simulation(
    thickness=thickness,
    n_blades=n_blades,
    tsr=tsr,
    iterations=1200,  # More iterations for smooth animation
    return_animation=True,
    verbose=True
)

print(f"\nFinal Cp = {Cp:.4f}")
print(f"Captured {len(frames)} animation frames")

# ============================================================
# Create Animation
# ============================================================
print("\nCreating animation...")

fig, ax = plt.subplots(figsize=(8, 8))

# Initial frame
velocity = frames[0]['velocity']
mask = frames[0]['mask']

# Velocity field
im = ax.imshow(velocity, cmap='coolwarm', origin='lower', 
               vmin=0, vmax=velocity.max() * 1.2)
plt.colorbar(im, ax=ax, label='Velocity Magnitude')

# Blade overlay
blade_overlay = ax.imshow(np.where(mask, np.nan, 0), cmap='binary', 
                          origin='lower', alpha=0.8)

# Title
title = ax.set_title(f"VAWT Flow - Azimuth: 0°, Cp={Cp:.3f}", fontsize=14)
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Draw turbine center
center_x, center_y = velocity.shape[1] // 2, velocity.shape[0] // 2
ax.plot(center_x, center_y, 'ko', markersize=8)

def update(frame_idx):
    frame = frames[frame_idx]
    velocity = frame['velocity']
    mask = frame['mask']
    azimuth = frame['azimuth']
    
    im.set_data(velocity)
    
    # Update blade overlay
    blade_img = np.zeros_like(velocity)
    blade_img[mask] = 1.0
    blade_overlay.set_data(np.where(mask, np.nan, 0))
    
    # Redraw blade contour
    ax.collections.clear()
    ax.contour(mask.astype(float), levels=[0.5], colors='black', linewidths=2)
    
    title.set_text(f"VAWT Flow - Azimuth: {azimuth:.0f}°, Cp={Cp:.3f}")
    
    return [im, blade_overlay, title]

# Create animation
anim = animation.FuncAnimation(fig, update, frames=len(frames), 
                                interval=100, blit=False)

# Save as GIF
print("Saving animation as 'vawt_animation.gif'...")
anim.save('vawt_animation.gif', writer='pillow', fps=10)
print("Animation saved!")

# Also save final frame as static image
plt.savefig('vawt_flow_final.png', dpi=150, bbox_inches='tight')
print("Static image saved as 'vawt_flow_final.png'")

# ============================================================
# Generate Cp vs TSR Curve
# ============================================================
print("\n" + "=" * 60)
print("Generating Cp vs TSR Performance Curve...")
print("=" * 60)

tsr_range, cp_values = compute_cp_vs_tsr(
    thickness=thickness,
    n_blades=n_blades,
    tsr_range=np.linspace(1.5, 6.0, 10),
    iterations=500
)

# Find optimal TSR
optimal_idx = np.argmax(cp_values)
optimal_tsr = tsr_range[optimal_idx]
optimal_cp = cp_values[optimal_idx]

print(f"\nOptimal TSR: {optimal_tsr:.2f}")
print(f"Peak Cp: {optimal_cp:.4f}")

# Plot Cp vs TSR
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(tsr_range, cp_values, 'b-o', linewidth=2, markersize=8, label='Simulation')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=optimal_tsr, color='r', linestyle='--', alpha=0.5, label=f'Optimal TSR={optimal_tsr:.2f}')
ax2.scatter([optimal_tsr], [optimal_cp], color='red', s=150, zorder=5, 
            label=f'Peak Cp={optimal_cp:.3f}')

# Betz limit reference
ax2.axhline(y=0.593, color='green', linestyle=':', alpha=0.5, label='Betz Limit (0.593)')

ax2.set_xlabel('Tip Speed Ratio (TSR)', fontsize=12)
ax2.set_ylabel('Power Coefficient (Cp)', fontsize=12)
ax2.set_title(f'VAWT Performance: NACA 00{int(thickness*100):02d}, {n_blades} Blades', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([1, 7])
ax2.set_ylim([-0.2, 0.7])

plt.tight_layout()
plt.savefig('vawt_cp_vs_tsr.png', dpi=150)
print("Saved 'vawt_cp_vs_tsr.png'")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print("\nGenerated Files:")
print("  1. vawt_animation.gif  - Rotating blade flow animation")
print("  2. vawt_flow_final.png - Final flow field snapshot")
print("  3. vawt_cp_vs_tsr.png  - Power coefficient vs TSR curve")
print(f"\nOptimal Configuration:")
print(f"  Profile: NACA 00{int(thickness*100):02d}")
print(f"  Blades:  {n_blades}")
print(f"  Best TSR: {optimal_tsr:.2f}")
print(f"  Peak Cp:  {optimal_cp:.4f}")

plt.show()
