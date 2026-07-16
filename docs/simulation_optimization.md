# Optimizing Simulations for Speed

Cheetah is designed for high-speed beam dynamics simulations. While it is already fast out-of-the-box, several opt-in optimizations are available to speed up simulations significantly—often by several orders of magnitude—especially when running large lattices where only a small number of parameters change.

## 1. Removing Inactive Markers and Zero-Length Elements

Lattices imported from other codes often contain diagnostic markers or zero-length elements. While useful for alignment, they introduce overhead in Python/PyTorch during tracking.

You can filter them out using:

```python
# Remove all Marker elements
clean_segment = segment.without_inactive_markers()

# Remove all zero-length elements that are currently inactive
clean_segment = segment.without_inactive_zero_length_elements()
```

## 2. Replacing Inactive Elements with Drifts

Drift spaces are the computationally cheapest elements to simulate. When active elements (like quadrupoles or correctors) are switched off or set to zero, they behave exactly like drifts. Cheetah can replace them with actual `Drift` elements to speed up the tracking:

```python
# Replace inactive elements with Drift elements, keeping steerers active
optimized_segment = segment.inactive_elements_as_drifts(
    except_for=["HCOR_1", "VCOR_1", "HCOR_2", "VCOR_2"]
)
```

## 3. Transfer Map Merging

For large lattices where only a few elements are adjusted (e.g. tuning correctors while keeping quadrupoles static), Cheetah allows you to merge the transfer maps of all unchanged, skippable elements:

```python
# Merge skippable elements, leaving only the steerers unmerged
merged_segment = segment.transfer_maps_merged(
    incoming_beam=incoming_beam,
    except_for=["HCOR_1", "VCOR_1", "HCOR_2", "VCOR_2"]
)
```

### How it works
- Cheetah identifies consecutive elements that are skippable (`element.is_skippable`) and are not listed in `except_for`.
- It tracks the beam through them once to compute the composite transfer map.
- It replaces the entire sequence with a single `CustomTransferMap` element.
- During subsequent tracking calls (e.g., in optimization loops or reinforcement learning training), only the single merged map is applied, eliminating thousands of matrix multiplications.

## Recommended Optimization Flow

To get the maximum possible speed out of a large lattice:

```python
import cheetah

# 1. Load the lattice
segment = cheetah.Segment.from_bmad("large_lattice.bmad")

# 2. Prepare the optimization segment
optimized_segment = (
    segment.without_inactive_markers()
           .without_inactive_zero_length_elements()
           .inactive_elements_as_drifts(except_for=["HCOR_1", "VCOR_1"])
           .transfer_maps_merged(incoming_beam, except_for=["HCOR_1", "VCOR_1"])
)

# 3. Fast tracking loop
for i in range(1000):
    # Modify the active steerer parameters
    optimized_segment.HCOR_1.angle = torch.tensor(1e-4)
    # Track is now extremely fast!
    outgoing = optimized_segment.track(incoming_beam)
```
