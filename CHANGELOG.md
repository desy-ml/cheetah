# Changelog

## v0.7.0 [🚧 Work in Progress]

### 🚨 Breaking Changes

### 🚀 Features

- `CustomTransferMap` elements created by combining multiple other elements will now reflect that in their `name` attribute (see #100) (@jank324)

### 🐛 Bug fixes

### 🐆 Other

## [v0.6.2](https://github.com/desy-ml/cheetah/releases/tag/v0.6.2) (2024-02-13)

### 🚨 Breaking Changes

- The handling of `device` and `dtype` was overhauled. They might not behave as expected. `Element`s also no longer have a `device` attribute. (see #115) (@jank324)

### 🚀 Features

- Add charge to the `ParameterBeam` and `ParticleBeam` classes (see #86) (@cr-xu)
- Add opt-in speed optimisations (see #88) (@jank324)

### 🐛 Bug fixes

- Fix the transfer maps in `Drift` and `Dipole`; Add R56 in horizontal and vertical correctors modelling (see #90) (@cr-xu)
- Fix fringe_field_exit of `Dipole` is overwritten by `fringe_field` bug (see #99) (@cr-xu)
- Fix error caused by mismatched devices on machines with CUDA GPUs (see #97 and #115) (@jank324)
- Fix error raised when tracking a `ParameterBeam` through an active `BPM` (see #101) (@jank324)
- Fix error in ASTRA beam import where the energy was set to `float64` instead of `float32` (see #111) (@jank324)
- Fix missing passing of `total_charge` in `ParameterBeam.transformed_to` (see #112) (@jank324)
- Fix `Cavitiy.__repr__` printing `voltage` value for `phase` property (see #121) (@jank324)

### 🐆 Other

- Add pull request template (see #97) (@jank324)
- Add _Acknowledgements_ section to README (see #103) (@jank324)
- `benchmark` directory was moved to `desy-ml/cheetah-demos` repository (see #108) (@jank324)
- Update citations to new arXiv preprint (see #117) (@jank324)
- Improve the docstring with proper units for the phase space dimensions (see #122) (@cr-xu)
- Link to the new paper (on arXiv) in the documentation (see #125) (@jank324)

## [v0.6.1](https://github.com/desy-ml/cheetah/releases/tag/v0.6.1) (2023-09-17)

### 🐛 Bug fixes

- Fix issue where `converters` submodule was not installed properly (see 1562496) (@jank324)

## [v0.6.0](https://github.com/desy-ml/cheetah/releases/tag/v0.6.0) (2023-09-15)

### 🚨 Breaking Changes

- Cheetah elements are now subclasses of `torch.nn.Module`, where the parameters should always be `torch.Tensor`. This makes cheetah a _fully differentiable simulation code_. (see #11)
- The `cell` keyword argument of `cheetah.Segment` has been renamed to `elements`.
- Element and beam parameters must now always be passed to the constructor or set afterwards as a `torch.Tensor`. It is no longer possible to use `float`, `int` or `np.ndarray`. (see #11)

### 🚀 Features

- Convert from Bmad lattices files (see #65) (@jank324)
- Add proper transfer map for cavity (see #65) (@jank324, @cr-xu)
- Twiss parameter calculation and generate new beam from twiss parameter (see #62) (@jank324, @cr-xu)
- Saving and loading lattices from LatticeJSON (see #9) (@cr-xu)
- Nested `Segment`s can now be flattened (@jank324)
- Ocelot converter adds support for `Cavity`, `TDCavity`, `Marker`, `Monitor`, `RBend`, `SBend`, `Aperture` and `Solenoid` (see #78) (@jank324)

### 🐛 Bug fixes

- Fix dependencies on readthedocs (see #54) (@jank324)
- Fix error when tracking `ParameterBeam` through segment on CPU (see #68) (@cr-xu)

### 🐆 Other

- Add `CHANGELOG.md` (see #74)
- Improved documentation on converting Marker object into cheetah (see #58) (#jank324)

## [v0.5.19](https://github.com/desy-ml/cheetah/releases/tag/v0.5.19) (2023-05-22)

### 🚀 Features

- Better error messages when beam and accelerator devices don't match (@FelixTheilen)

### 🐛 Bug fixes

- Fix BPM issue with `ParameterBeam` (@FelixTheilen)
- Fix wrong screen reading dimensions (@cr-xu)

### 🐆 Other

- Improve docstrings (@FelixTheilen)
- Implement better testing with _pytest_ (@FelixTheilen)
- Setup formatting with `black` and `isort` as well as `flake8` listing (@cr-xu)
- Add type annotations (@jank324)
- Setup Sphinx documentation on readthedocs (@jank324)

## [v0.5.18](https://github.com/desy-ml/cheetah/releases/tag/v0.5.18) (2023-02-06)

### 🐛 Bug Fixes

- Fix issue where multivariate_normal() crashes because cov is not positive-semidefinite.

## [v0.5.17](https://github.com/desy-ml/cheetah/releases/tag/v0.5.17) (2023-02-05)

### 🚀 New Features

- Faster screen reading simulation by using torch.histogramdd()

## [v0.5.16](https://github.com/desy-ml/cheetah/releases/tag/v0.5.16) (2023-02-02)

### 🐛 Bug Fixes

- Fix bug where some screens from the ARES Ocelot lattice were converted to Drift elements.

## [v0.5.15](https://github.com/desy-ml/cheetah/releases/tag/v0.5.15) (2022-10-12)

### 🚀 New Features

- Ocelot has been removed as a mandatory dependency and is now only needed when someone wants to convert from Ocelot objects.

### 🐛 Bug Fixes

- An error that only packages with all dependencies available on PyPI can be installed from PyPI has been fixed.

## [v0.5.14](https://github.com/desy-ml/cheetah/releases/tag/v0.5.14) (2022-09-28)

### 🚀 Features

- Introduce black for code formatting and isort for import organisation.
- Prepare release on PyPI as cheetah-accelerator.

## [v0.5.13](https://github.com/desy-ml/cheetah/releases/tag/v0.5.13) (2022-07-26)

### 🚀 Features

- Add caching of Screen.reading to avoid expensive recomputation

### 🐛 Bug Fixes

- Fix install dependencies

## [v0.5.12](https://github.com/desy-ml/cheetah/releases/tag/v0.5.12) (2022-06-16)

- First Release of Cheetah 🐆🎉
