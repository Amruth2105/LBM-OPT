# Neural-Optimized Wind Turbine Blade (LBM + BayesOpt)

This project uses **Computational Fluid Dynamics (CFD)** and **Bayesian Optimization** to design optimal wind turbine blade profiles.

- **Physics Engine:** Lattice Boltzmann Method (LBM) implemented in pure Python (NumPy)
- **Geometry:** NACA 4-Digit Series parameterization
- **Optimization:** Bayesian Optimization to maximize performance

## Installation

```bash
pip install -r requirements.txt
```

---

## Part 1: Horizontal Axis Wind Turbine (HAWT)

Static airfoil optimization for maximum Lift-to-Drag ratio.

| File | Description |
|------|-------------|
| `airfoil_gen.py` | NACA airfoil geometry generator |
| `lbm_solver.py` | LBM solver for static airfoils |
| `optimize.py` | Bayesian optimization for HAWT |
| `visualize.py` | Flow visualization |

### Usage

```bash
python3 optimize.py    # Find optimal blade shape
python3 visualize.py   # Visualize the result
```

---

## Part 2: Vertical Axis Wind Turbine (VAWT)

Rotating Darrieus-type VAWT simulation using Immersed Boundary LBM.

| File | Description |
|------|-------------|
| `vawt_geometry.py` | Rotating blade geometry (NACA 00XX) |
| `vawt_solver.py` | LBM with rotating immersed boundary |
| `vawt_optimize.py` | Optimize thickness, #blades, TSR |
| `vawt_visualize.py` | Animation + Cp vs TSR curves |

### VAWT Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `thickness` | NACA profile thickness | 10% - 22% |
| `n_blades` | Number of blades | 2 - 5 |
| `tsr` | Tip Speed Ratio (ωR/V∞) | 2.0 - 6.0 |

### Usage

```bash
python3 vawt_optimize.py   # Find optimal VAWT configuration
python3 vawt_visualize.py  # Generate animation + performance curves
```

### Outputs

- `vawt_animation.gif` - Rotating blade flow animation
- `vawt_cp_vs_tsr.png` - Power coefficient performance curve
- `vawt_flow_final.png` - Final flow field snapshot

---

## Physics Notes

### 2D Simplification

Both simulations use 2D flow approximation:
- **HAWT**: Cross-section at a fixed span location
- **VAWT**: Assumes uniform flow along blade span (neglects tip losses)

This is a common research simplification that overestimates Cp by ~10-20%.

### Key Metrics

- **HAWT**: Lift-to-Drag ratio (L/D)
- **VAWT**: Power Coefficient (Cp = P / ½ρAV³)

## References

- NACA Report 460 (1933) - Airfoil geometry equations
- Lattice Boltzmann Methods for Fluid Dynamics - S. Succi
- Performance of Darrieus VAWTs - Paraschivoiu (2002)

