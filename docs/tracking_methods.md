# Tracking Methods

Cheetah supports multiple physical tracking methods (orders of tracking) for propagating beams through accelerator elements. Depending on your simulation needs, you can choose between fast first-order tracking, detailed second-order aberration tracking, or symplectic drift-kick-drift tracking.

## Supported Tracking Methods

The primary tracking methods supported by Cheetah elements are:

1. **`"linear"` (First-Order)**
   - Uses first-order transfer matrix ($R$-matrix) multiplication to propagate the beam.
   - This is highly efficient and linear, and is the default for most elements.
   - Supported by: `Drift`, `Dipole`, `Quadrupole`, `Sextupole`, `Solenoid`, `HorizontalCorrector`, `VerticalCorrector`, `CombinedCorrector`, `Undulator`.

2. **`"second_order"` (Second-Order)**
   - Uses second-order transfer tensor ($T$-tensor) multiplication to capture chromatic and geometric aberrations.
   - Differentiable with PyTorch autograd.
   - Supported by: `Drift`, `Dipole`, `Quadrupole`, `Sextupole`.

3. **`"drift_kick_drift"`**
   - Propagates particles through symplectic drift-kick-drift steps.
   - Ideal for non-linear dynamics and is required for certain elements.
   - Supported by: `Drift`, `Dipole`, `Quadrupole`, `TransverseDeflectingCavity`.

4. **Custom / Element-Specific Methods**
   - Some elements (such as diagnostics or custom physical models) use custom tracking routines that do not fall into the three categories above.
   - For these elements, their `supported_tracking_methods` is a single-item list containing their own lowercase class name.
   - Supported by: `Aperture` (`"aperture"`), `BPM` (`"bpm"`), `Screen` (`"screen"`), `SpaceChargeKick` (`"spacechargekick"`), `Marker` (`"marker"`).

## Configuring Tracking Methods

### For a Single Element

You can set the tracking method during element creation:

```python
import cheetah

# Create a quadrupole using second-order tracking
quad = cheetah.Quadrupole(
    length=0.1, 
    k1=4.2, 
    tracking_method="second_order"
)
```

You can also change the tracking method of an existing element at runtime by modifying its `tracking_method` property:

```python
# Check supported methods
print(quad.supported_tracking_methods)  # ['linear', 'second_order', 'drift_kick_drift']

# Switch to drift-kick-drift
quad.tracking_method = "drift_kick_drift"
```

### For a Whole Segment

To switch the tracking order for an entire `Segment`, iterate over its elements and set the method on those that support it:

```python
# Configure the entire segment to track using second-order maps where possible
for element in segment.elements:
    if "second_order" in element.supported_tracking_methods:
        element.tracking_method = "second_order"
```
