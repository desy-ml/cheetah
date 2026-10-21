# Xsuite second-order reference maps

`second_order_maps.npz` contains only the second-order body-map tensors used by
`tests/test_compare_xsuite.py`. The cases and reference energy live in
`generate_maps.py`; no element parameters or metadata are stored in the archive.
The test explicitly repeats the Cheetah parameters and reference energy for
readability; keep them consistent with the generator when changing cases.

The reference maps were generated on **2026-10-21**, using Python 3.13.5 on macOS
with the following package versions:

| Package    | Version |
| ---------- | ------- |
| xsuite     | 0.66.0  |
| xtrack     | 0.118.0 |
| xpart      | 0.23.22 |
| xobjects   | 0.7.1   |
| xdeps      | 0.10.21 |
| madng-tpsa | 0.3.2   |
| numpy      | 2.2.6   |
| scipy      | 1.16.0  |

Update this date and version table when regenerating the reference maps.

Regenerate from the repository root with:

```sh
python tests/resources/xsuite/generate_maps.py
```

Generation requires `numpy`, `scipy`, `xtrack==0.118.0`, and `madng-tpsa==0.3.2`,
plus a C compiler for Xsuite's CPU kernels. Xsuite and MAD-NG TPSA are only needed
for regeneration, not for running the unit tests.

The script tracks a second-order Taylor map through a sector bend with dipole,
quadrupole, and sextupole fields and disabled entrance/exit edges. It uses the
`bend-kick-bend` model and sixth-order Yoshida integration, checking that the map
has convergence as 256 and 512 multipole kicks produce close results.

Inputs and outputs are converted to Cheetah coordinates
`(x, px, y, py, tau, p)`, where `tau = -zeta/beta0` and
`p = (E - E0)/p0c`. The nonlinear conversion between Xsuite's momentum deviation
`delta` and `p` is included through second order. Each saved tensor has shape
`(7, 7, 7)` and Cheetah's upper-triangular convention: a mixed monomial has one
coefficient, and diagonal monomials have no extra factor of two.

Tests compare every tensor entry directly with explicit tolerances for integration
error and floating-point precision.
