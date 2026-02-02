# Neural-Optimized Wind Turbine Blade (LBM + BayesOpt)

This project uses **Computational Fluid Dynamics (CFD)** and **Bayesian Optimization** to design the most efficient wind turbine blade profile.

- **Physics Engine:** Lattice Boltzmann Method (LBM) implemented in pure Python (NumPy)
- **Geometry:** NACA 4-Digit Series parameterization
- **Optimization:** Bayesian Optimization to maximize Lift-to-Drag ratio

## Installation

```bash
pip install -r requirements.txt
```

## File Structure

| File | Description |
|------|-------------|
| `airfoil_gen.py` | Generates NACA airfoil geometry on a pixel grid |
| `lbm_solver.py` | Lattice Boltzmann fluid dynamics solver (D2Q9) |
| `optimize.py` | Bayesian Optimization controller |
| `visualize.py` | Generates velocity heatmap of optimized flow |

## Usage

### 1. Run Optimization

```bash
python optimize.py
```

This tests different NACA profiles and finds the best Lift/Drag ratio.

### 2. Visualize Results

```bash
python visualize.py
```

Generates `optimized_blade_flow.png` showing the velocity field.

## How to Interpret Results

- **Blue areas:** Low velocity (wake behind blade)
- **Red areas:** High velocity (flow acceleration over curved surface)
- **Optimal shapes:** Typically asymmetric with camber for lift generation

## NACA Parameters

| Parameter | Description | Search Range |
|-----------|-------------|--------------|
| `m` | Max camber | 0% - 8% |
| `p` | Camber position | 20% - 60% |
| `t` | Max thickness | 8% - 20% |

## Next Steps

- Add Angle of Attack to optimizer search space
- Increase grid resolution for higher fidelity
- Compare results with real NACA airfoil data
