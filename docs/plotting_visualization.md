# Plotting & 3D Visualization

Cheetah provides several utilities for plotting beam parameters and visualizing accelerator lattices in 2D and 3D.

## 2D Plotting of Beam Parameters

Cheetah can compute and plot Twiss parameters and beam statistics (like centroids or envelope sizes) along the length of a `Segment`.

### Tracking Attributes Along a Segment

To compute beam properties at every step along the lattice, use the `get_beam_attrs_along_segment` method:

```python
# Returns positions (s) and computed attributes along the lattice
s, x_envelope, y_envelope = segment.get_beam_attrs_along_segment(
    incoming_beam,
    attrs=["sigma_x", "sigma_y"]
)
```

### Plotting Twiss Parameters

To plot the horizontal ($\beta_x$) and vertical ($\beta_y$) beta functions along the lattice:

```python
# Plot Twiss parameters directly
fig, ax = segment.plot_twiss(incoming_beam)

# Or plot Twiss parameters with the lattice layout aligned underneath
fig, axes = segment.plot_twiss_over_lattice(incoming_beam)
```

### Plotting Beam Attributes

To plot general statistics (e.g. centroids `mu_x`, `mu_y` or envelopes `sigma_x`, `sigma_y`):

```python
# Plot centroids
fig, ax = segment.plot_beam_attrs(incoming_beam, attrs=["mu_x", "mu_y"])

# Plot envelopes over the lattice layout
fig, axes = segment.plot_beam_attrs_over_lattice(incoming_beam, attrs=["sigma_x", "sigma_y"])
```

### Quick Overview Plots

Use `plot_overview` to plot a standard summary of the beam centroids, standard deviations, and reference particle traces along the lattice:

```python
segment.plot_overview(incoming=incoming_beam, resolution=0.05)
```

Use `plot_mean_and_std` for a simpler plot of the centroid position accompanied by shaded bands representing the standard deviation:

```python
segment.plot_mean_and_std(incoming_beam)
```

---

## 3D Visualization

Cheetah allows you to generate a 3D CAD mesh of your lattice. This can be viewed interactively or exported to standard formats like GLB/gltf.

### Installation Prerequisite

3D visualization requires additional dependencies. Install Cheetah with the `[3d-visualization]` extra:

```bash
pip install "cheetah-accelerator[3d-visualization]"
```

### Generating and Showing a Mesh

Use the `to_mesh` method on a `Segment` to generate a 3D mesh:

```python
import cheetah

# Generate 3D mesh
mesh, _ = segment.to_mesh(
    cuteness={
        cheetah.HorizontalCorrector: 2.0,
        cheetah.VerticalCorrector: 2.0
    }
)

# Open an interactive 3D viewer
mesh.show()
```

*Note: The `cuteness` parameter is a dictionary specifying visual scaling factors for specific element classes to make them stand out in the 3D rendering.*

### Exporting a Mesh

You can export the generated mesh to a GLB or glTF file for inclusion in web pages, presentations, or CAD software:

```python
# Export to a binary GLB file
mesh.export("my_lattice_mesh.glb", file_type="glb")
```
