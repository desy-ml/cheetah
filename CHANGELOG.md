# Changelog

## v0.7.1 [🚧 Work in Progress]

### 🚨 Breaking Changes

- The `incoming` argument of `Segment.plot_overview` is no longer optional. This change also affects the order of the arguments. Fixes an exception that was raised by an underlying plot function that requires `incoming` to be set. (see #316) (@Hespe)
- Python 3.9 is no longer supported. This does not immediately break existing code, but might cause it to break in the future. (see #325) (@jank324)
- The covariance properties of the different beam classes were renamed from names like `cor_x` and `sigma_xpx` to consistent names like `cov_xpx` (see #331) (@jank324)

### 🚀 Features

- `ParticleBeam` now supports importing from and exporting to [openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics) HDF5 files and `ParticleGroup` objects. This allows for easy conversion to and from other file formats supported by openPMD-beamphysics. (see #305, #320) (@cr-xu, @Hespe)
- Add `marker`, `quadrupole` and `csbend` element names to the Elegant converter (see #327) (@jank324)
- Add Python 3.13 support (see #275) (@jank324)
- Methods `to_parameter_beam` and `to_particle_beam` have been added for convenient conversion between `ParticleBeam` and `ParameterBeam` (see #331) (@jank324)
- Beam classes now have the `mu_tau` and `mu_p` properties on their interfaces (see #331) (@jank324)

### 🐛 Bug fixes

- Fix issue where a space before a comma could cause the Elegant and Bmad converters to fail (see #327) (@jank324)

### 🐆 Other

- The tests for backward-mode differentiation with space charge was improved, by checking the accuracy of the gradients (see #339) (@RemiLehe)
- A tests for forward-mode differentiation with space charge was added (see #339) (@RemiLehe)
- Test tolerances were adjusted reduce the chance of random test failures (see #309, #324) (@Hespe, @jank324)
- The copyright years were updated to 2025 (see #318) (@jank324)
- The broken institution logo rendering in the documentation has been fixed (see #318) (@jank324)
- Added `pyproject.toml` to conform with PEP 660 as enforced as of pip 25 for editable installs (see #334) (@jank324)
- Add TUHH logo to contributing institution logos (see #338) (@jank324)
- Doc: ImpactX Example (Space Charge) #341 (@ax3l)

### 🌟 First Time Contributors

## [v0.7.0](https://github.com/desy-ml/cheetah/releases/tag/v0.7.0) (2024-12-13)

We are proud to announce this new major release of Cheetah! This is probably the biggest release since the original Cheetah release, with many with significant upgrades under the hood. Cheetah is now fully vectorised and compatible with PyTorch broadcasting rules, while additional physics and higher fidelity options for existing physics have also been introduced. Despite extensive testing, you might still encounter a few bugs. Please report them by opening an issue, so we can fix them as soon as possible and improve the experience for everyone.

### 🚨 Breaking Changes

- Cheetah is now vectorised. This means that you can run multiple simulations in parallel by passing a batch of beams and settings, resulting a number of interfaces being changed. For Cheetah developers this means that you now have to account for an arbitrary-dimensional tensor of most of the properties of you element, rather than a single value, vector or whatever else a property was before. (see #116, #157, #170, #172, #173, #198, #208, #213, #215, #218, #229, #233, #258, #265, #284, #291) (@jank324, @cr-xu, @Hespe, @roussel-ryan)
- As part of the vectorised rewrite, the `Aperture` no longer removes particles. Instead, `ParticleBeam.survival_probabilities` tracks the probability that a particle has survived (i.e. the inverse probability that it has been lost). This also comes with the removal of `Beam.empty`. Note that particle losses in `Aperture` are currently not differentiable. This will be addressed in a future release. (see #268) (@cr-xu, @jank324)
- The fifth particle coordinate `s` is renamed to `tau`. Now Cheetah uses the canonical variables in phase space $(x,px=\frac{P_x}{p_0},y,py, \tau=c\Delta t, \delta=\Delta E/{p_0 c})$. In addition, the trailing "s" was removed from some beam property names (e.g. `beam.xs` becomes `beam.x`). (see #163, #284) (@cr-xu, @Hespe)
- `Screen` no longer blocks the beam (by default). To return to old behaviour, set `Screen.is_blocking = True`. (see #208) (@jank324, @roussel-ryan)
- The way `dtype`s are determined is now more in line with PyTorch's conventions. This may cause different-than-expected `dtype`s in old code. (see #254) (@Hespe, @jank324)
- `Beam.parameters()` no longer shadows `torch.nn.Module.parameters()`. The previously returned properties now need to be queried individually. (see #300) (@Hespe)
- `e1` and `e2` in `Dipole` and `RBend` have been renamed and made more consistent between the different magnet types. They now have prefixes `dipole_` and `rbend_` respectively. (see #289) (@Hespe, @jank324)
- The `_transfer_map` property of `CustomTransferMap` has been renamed to `predefined_transfer_map`. (see #289) (@Hespe, @jank324)

### 🚀 Features

- `CustomTransferMap` elements created by combining multiple other elements will now reflect that in their `name` attribute (see #100) (@jank324)
- Add a new class method for `ParticleBeam` to generate a 3D uniformly distributed ellipsoidal beam (see #146) (@cr-xu, @jank324)
- Add Python 3.12 support (see #161) (@jank324)
- Implement space charge using Green's function in a `SpaceChargeKick` element (see #142) (@greglenerd, @RemiLehe, @ax3l, @cr-xu, @jank324)
- `Segment`s can now be imported from Bmad to devices other than `torch.device("cpu")` and dtypes other than `torch.float32` (see #196, #206) (@jank324)
- `Screen` now offers the option to use KDE for differentiable images (see #200) (@cr-xu, @roussel-ryan)
- Moving `Element`s and `Beam`s to a different `device` and changing their `dtype` like with any `torch.nn.Module` is now possible (see #209) (@jank324)
- `Quadrupole` now supports tracking with Cheetah's matrix-based method or with Bmad's more accurate method (see #153) (@jp-ga, @jank324)
- Port Bmad-X tracking methods to Cheetah for `Quadrupole`, `Drift`, and `Dipole` (see #153, #240) (@jp-ga, @jank324)
- Add `TransverseDeflectingCavity` element (following the Bmad-X implementation) (see #240, #278 #296) (@jp-ga, @cr-xu, @jank324)
- `Dipole` and `RBend` now take a focusing moment `k1` (see #235, #247) (@Hespe)
- Implement a converter for lattice files imported from Elegant (see #222, #251, #273, #281) (@Hespe, @jank324)
- `Beam` and `Element` objects now have a `.clone()` method to create a deep copy (see #289) (@Hespe, @jank324)
- `ParticleBeam` now comes with methods for plotting the beam distribution in a variety of ways (see #292) (@roussel-ryan, @jank324)

### 🐛 Bug fixes

- Now all `Element` have a default length of `torch.zeros((1))`, fixing occasional issues with using elements without length, such as `Marker`, `BPM`, `Screen`, and `Aperture`. (see #143) (@cr-xu)
- Fix bug in `Cavity` `_track_beam` (see #150) (@jp-ga)
- Fix issue where dipoles would not get a unique name by default (see #186) (@Hespe)
- Add `name` to `Drift` element `__repr__` (see #201) (@ansantam)
- Fix bug where `dtype` was not used when creating a `ParameterBeam` from Twiss parameters (see #206) (@jank324)
- Fix bug after running `Segment.inactive_elements_as_drifts` the drifts could have the wrong `dtype` (see #206) (@jank324)
- Fix an issue where splitting elements would result in splits with a different `dtype` (see #211) (@jank324)
- Fix issue in Bmad import where collimators had no length by interpreting them as `Drift` + `Aperture` (see #249) (@jank324)
- Fix NumPy 2 compatibility issues with PyTorch on Windows (see #220, #242) (@Hespe)
- Fix issue with Dipole hgap conversion in Bmad import (see #261) (@cr-xu)
- Fix plotting for segments that contain tensors with `require_grad=True` (see #288) (@Hespe)
- Fix bug where `Element.length` could not be set as a `torch.nn.Parameter` (see #301) (@jank324, @Hespe)
- Fix registration of `torch.nn.Parameter` at initilization for elements and beams (see #303) (@Hespe)
- Fix warnings about NumPy deprecations and unintentional tensor clones (see #308) (@Hespe)

### 🐆 Other

- Update versions of some steps used by GitHub actions to handle Node.js 16 end-of-life (@jank324)
- Update versions in pre-commit config (see #148) (@jank324)
- Split `accelerator` and `beam` into separate submodules (see #158) (@jank324)
- Update reference from arXiv preprint to PRAB publication (see #166) (@jank324)
- Rename converter modules to the respective name of the accelerator code (see #167) (@jank324)
- Added imports to the code example in the README (see #188) (@jank324)
- Refactor definitions of physical constants (see #189) (@Hespe)
- Fix the quadrupole strength units in the quadrupole docstring (see #202) (@ansantam)
- Add CI runs for macOS (arm64) and Windows (see #226) (@cr-xu, @jank324, @Hespe)
- Clean up CI pipelines (see #243, #244) (@jank324)
- Fix logo display in README (see #252) (@jank324)
- Made `Beam` an abstract class (see #284) (@Hespe)
- Releases are now automatically archived on Zenodo and given a DOI (@jank324)
- The Acknowledgements section in the README has been updated to reflect new contributors (see #304) (@jank324, @AnEichler)

### 🌟 First Time Contributors

- Grégoire Charleux (@greglenerd)
- Remi Lehe (@RemiLehe)
- Axel Huebl (@ax3l)
- Juan Pablo Gonzalez-Aguilera (@jp-ga)
- Andrea Santamaria Garcia (@ansantam)
- Ryan Roussel (@roussel-ryan)
- Christian Hespe (@Hespe)
- Annika Eichler (@AnEichler)

## [v0.6.3](https://github.com/desy-ml/cheetah/releases/tag/v0.6.3) (2024-03-28)

### 🐛 Bug fixes

- Fix bug in `Cavity` transfer map bug. (see #129 and #135) (@cr-xu)

### 🐆 Other

- Add GPL 3 licence (see #131) (@jank324)

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
