# 3D Model 

## Accessing the 3D Model Files

To use the 3D models for the Quadrupole and Dipole components, ensure that you have the necessary files provided by the developers of Cheetah. These models are essential for visualization and simulation purposes within the Cheetah framework.

## Adding 3D Model Files

Place the `.gltf` files for the Quadrupole and Dipole models in the designated directory:

```
cheetah/_assets/3D_models/
```
Once the models are added, update the path variables in `segment_3d_plotter.py` to reflect their locations correctly.

## Directory Structure

The expected directory structure for `cheetah/_assets/` is as follows:
.
├── ACHIP_EA1_2021.1351.001
├── __init__.py
├── ares_ea_cavity.glb
├── ares_ea_quadrupole.glb
├── ares_ea_screen_station.glb
├── ares_ea_small_steerer_hcor.glb
├── ares_ea_small_steerer_vcor.glb
└── ares_ea_undulator.glb
