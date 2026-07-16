# Optimising Simulations for Speed

Cheetah is designed for high-speed beam dynamics simulations. While it is already fast out-of-the-box, several opt-in optimisations are available to speed up simulations significantly—often by several orders of magnitude—especially when running large lattices where only a small number of parameters change.

This page guides you through the available optimisations, provides typical benchmark performance numbers, and details how to JIT-compile Cheetah using PyTorch.

---

## 1. Removing Inactive Markers and Zero-Length Elements

Lattices imported from other codes often contain diagnostic markers or zero-length elements. While useful for alignment, they introduce overhead in Python/PyTorch during tracking.

You can filter them out using:

```python
# Remove all Marker elements
clean_segment = segment.without_inactive_markers()

# Remove all zero-length elements that are currently inactive
clean_segment = segment.without_inactive_zero_length_elements()
```

*Benchmark Impact:* In a typical 1000-element lattice, removing inactive markers can reduce tracking time from **11.0 ms** to **9.63 ms** (approx. 12% speedup).

---

## 2. Replacing Inactive Elements with Drifts

Drift spaces are the computationally cheapest elements to simulate. When active elements (like quadrupoles or correctors) are switched off or set to zero, they behave exactly like drifts. Cheetah can replace them with actual `Drift` elements to speed up the tracking:

```python
# Replace inactive elements with Drift elements, keeping steerers active
optimized_segment = segment.inactive_elements_as_drifts(
    except_for=["HCOR_1", "VCOR_1", "HCOR_2", "VCOR_2"]
)
```

---

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
- During subsequent tracking calls, only the single merged map is applied, eliminating thousands of matrix multiplications.

*Benchmark Impact:* In a 1000-element lattice, merging transfer maps can reduce tracking time from **11.0 ms** to **112 μs** (a **100x speedup**!).

---

## 4. Vectorisation and Broadcasting

Instead of evaluating multiple magnet settings in a Python loop, you can pass settings as PyTorch tensors. PyTorch performs the tracking in parallel over all configurations:

```python
# Vectorised parameter sweep (e.g., 1000 corrector settings)
fully_optimized_segment.HCOR_1.angle = torch.linspace(-1e-4, 1e-4, 1000)
outgoing = fully_optimized_segment.track(incoming_beam)
```

*Benchmark Impact:* Running 1000 configurations in a sequential Python loop takes **177 ms** (~177 μs per track). Running the same sweep using PyTorch broadcasting takes just **1.17 ms** total (~1.17 μs per track, a **150x speedup** over the loop and **10,000x speedup** compared to the unoptimised loop!).

---

## 5. Just-in-Time (JIT) Compiling Cheetah

PyTorch supports compiling code to optimise runtime performance. JIT compilation provides a **10-20% speedup** on AMD CPUs, a **0.5-2x speedup** on Intel CPUs, and up to **8x speedup** on Nvidia GPUs.

JIT compilation takes around 4-20 seconds on CPU (and more on GPU) on the first tracking call, but subsequent calls are significantly faster.

```python
import torch

# Compile the track function of your segment
compiled_track = torch.compile(merged_segment.track)

# First call performs JIT compilation (slow)
_ = compiled_track(incoming_beam)

# Subsequent calls are fully compiled and optimised
outgoing = compiled_track(incoming_beam)
```

### JIT Math Precision Settings
By default, PyTorch compiles with fast-math optimisations on some platforms. If you observe precision issues after compilation, disable unsafe math optimisations before calling `torch.compile`:

```python
# C++ backend
torch._inductor.config.cpp.enable_unsafe_math_opt_flag = False
# CUDA backend
torch._inductor.config.cuda.use_fast_math = False
# ROCm backend
torch._inductor.config.rocm.use_fast_math = False
```
