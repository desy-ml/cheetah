# Changelog

## v0.7.6 [🚧 Work in Progress]

### 🚨 Breaking Changes

- `Segment.set_attrs_on_every_element_of_type` has been renamed to `Segment.set_attrs_on_every_element`, and made more general, with the `element_type` argument being optional and renamed to `filter_type`. (see #476) (@jank324, @cr-xu)
- Cheetah Modules (`Element`, `Beam`, `Species`) no longer automatically change the device and dtype of passed parameters. Instead, the user is expected to make sure that the device and dtype of parameters and Modules match. This is more in line with how Modules included in PyTorch operate. (see #538, #552) (@jank324, @Hespe)

### 🚀 Features

- Implement second-order tracking for `Drift`, `Dipole` and `Quadrupole` elements, and add a convenient method to set tracking methods for an entire segment. This comes with an overhaul of the overall tracking system. Rename the tracking method `"cheetah"` to `"linear"` and `"bmadx"` to `"drift_kick_drift"`. The existing methods `"cheetah"` and `"bmadx"` will remain supported with a `DeprecationWarning`. (see #476) (@cr-xu, @jank324, @Hespe)
- `Cavity` now supports travelling wave cavities in addition to standing wave cavities via the `cavity_type` argument (see #286) (@zihan-zh, @jank324)
- Documented PyTorch `compile` for improved speed (see #390) (@ax3l)
- Beam classes now account for dispersion. Dispersion correction is included in the Twiss and emittance computations. Dispersion arguments are added to `from_parameters` and `from_twiss` beam initialisation methods. (see #540) (@cr-xu)
- Add convenience methods to `Segment` for getting an ordered list of all element names and the index of a specific element by its name (see #534) (@roussel-ryan, @jank324)
- First- and second-order transfer maps are now cached resulting in potential speed-ups of up to 10x and more (see #532) (@jank324)
- Methods for creating `ParticleBeam` instances from distributions via stochastic sampling now make sure that the statistics of the generated particles match the desired distribution (see #546) (@cr-xu)
- `BPM` elements now support misalignments (see #533) (@roussel-ryan, @jank324)
- Speed up tracking by replacing some PyTorch operations with faster alternatives (see #538, #561) (@jank324, @Hespe)

### 🐛 Bug fixes

- Shorten `__repr__` of `Segment` for large lattices to prevent debugging slowdowns (see #529) (@Hespe)
- Fix typo saying Bmad in Elegant import method docstring (see #531) (@jank324)
- Remove division by zero in `Cavity` for off-crest phase (see #549, #550) (@Hespe)
- Fix issue with `SpaceChargeKick` where the particle species was not preserved (see #560) (@austin-hoover, @jank324)

### 🐆 Other

- Add a speed benchmarking workflow of tracking through the ARES lattice (see #527) (@Hespe)
- Add tests that track through every subclass of `Element` for all permissible `dtypes` and `device` combinations (see #499) (@Hespe)
- Fix false dtype in `Screen` documentation (see #544) (@jp-ga)
- Fix an issue where running the plot tests on Windows (most notably on the recently upgraded GitHub Actions Windows runners) would sporadically fail with a `_tkinter.TclError: Can't find a usable init.tcl in the following directories` error, by forcing the matplotlib backend to `Agg` when running tests on Windows. (see #567) (@jank324)
- Temporarily remove `flake8-black` from `format` Action because it causes issues with the latest `black` version (see #569) (@jank324)

### 🌟 First Time Contributors

- Zihan Zhu (@zihan-zh)
- Austin Hoover (@austin-hoover)

## [v0.7.5](https://github.com/desy-ml/cheetah/releases/tag/v0.7.5) (2025-08-04)

### 🚀 Features

- Add support for elements (especially `Drift`) with negative length (see #480) (@Hespe)
- Warnings are now available in the top-level namespace so that they can be referenced as e.g. `cheetah.PhysicsWarning` to shorten `filterwarnigns` code. (see #497) (@jank324)
- Add the ability to the Bmad and Elegant converters to parse expressions that access properties from other elements (see #501, #498) (@amylizzle, @jank324)
- Update the Elegant converter to allow element names with colon as well as the definition of reversed beamlines with a minus sign. (see #504) (@cr-xu, @jank324)
- `Segment`s can now conveniently be reversed with the `Segment.reversed` method (see #504) (@jank324)
- New feature for generating 3D models of lattices and viewing them (see #352, #502, #511) (@jank324, @chrisjcc, @SuchethShenoy)

### 🐛 Bug fixes

- Fix various `dtype` and `device` pertaining to `ParticleBeam`, `Species` and `SpaceChargeKick` (see #485, #486, #490, #491) (@Hespe, @jank324, @adhamrait)
- Remove incorrect implementation of `split` from `HorizontalCorrector` and `VerticalCorrector` (see #480) (@Hespe)

### 🐆 Other

- Updated contributor list and funding strings in README and on docs index page (see #487) (@jank324)
- Add a Binder and link to `cheetah-demos` (see #482) (@smartsammler, @jank324)
- PyTorch is now configured to use only deterministic algorithms during tests, preventing intermittent test failures (see #480) (@Hespe)
- Make README example more copy-paste friendly, and generally improve it and the simple intro notebook in the docs. (see #493, #496) (@jank324, @ax3l)
- Fix comparison tests to work with new PyPI release of Ocelot. Remove Twiss tests where they are not needed. Increase tolerances where Cheetah and Ocelot follow slightly different models. (see #513, #519) (@jank324, @cr-xu, @Hespe)

### 🌟 First Time Contributors

- Julian Gethmann (@smartsammler)
- Arjun Dhamrait (@adhamrait)
- Christian Contreras-Campana (@chrisjcc)
- Sucheth Shenoy (@SuchethShenoy)

## [v0.7.4](https://github.com/desy-ml/cheetah/releases/tag/v0.7.4) (2025-06-19)

### 🚀 Features

- The new warning system was extended to have more specific subclasses of `PhysicsWarning` to allow for better and easier filtering of warnings (see #415) (@Hespe, @jank324)
- Implement an infix notation parser for Bmad and Elegant converters, fixing a potential security issue where `eval()` could be called on user input. (see #412) (@amylizzle)
- Improve numerical stability of the `base_rmatrix` and `base_ttensor` functions (related to #469) (see #474) (@Hespe, @jank324)
- Minor speed improvements in `base_rmatrix` and `base_ttensor` by reducing memory allocations for constants, and skipping rotation computations when the present tilt has now effect. (see #474) (@Hespe, @jank324)

### 🐛 Bug fixes

- Fix issue that `base_rmatrix` has large error for small `k1` values even for double precision (see #469) (@cr-xu)
- Rework the covariance computation in `ParticleBeam.as_parameter_beam` to fix an issue that caused the covariance to be computed incorrectly for vectorised beams (see #471) (@cr-xu, @jank324, @Hespe)
- Unrecognised element properties in Bmad and Elegant lattice files now print a warning instead of exiting with an `AssertionError` (see #415) (@amylizzle, @jank324)
- A bug was fixed that caused the Bmad and Elegant importers to incorrectly parse `;` line endings and comments starting with `#` (see #415) (@Hespe, @jank324)
- The `santize_names` parameter is now correctly passed to `BPM` and `Marker` elements when converting from Elegant (see #473) (@amylizzle)

## [v0.7.3](https://github.com/desy-ml/cheetah/releases/tag/v0.7.3) (2025-06-11)

### 🚨 Breaking Changes

- The default resolution of all plotting functions on `Segment` is now `None`, i.e. element-wise. For most lattices this will only result in faster plotting, but note that it is possible that your plots look slightly different, especially if your lattice is short or has few elements. (see #459) (@jank324, @Hespe)
- Cheetah now requires `torch>=2.3` (see #461) (@jank324)
- Combine the `num_grid_points_{x,y,tau}` arguments of `SpaceChargeKick` into the `grid_shape` tuple. Fixes the cloning of `SpaceChargeKick`. In addition, the `grid_extend_*` properties were renamed to `grid_extent_*` (see #418) (@Hespe, @jank324)
- Warning messages, which were previously just printed are now produced using the `warnings` module, brining with it all the features of the latter. (see #450) (@Hespe, @jank324)

### 🚀 Features

- Add `KQUAD` and `CSRCSBEND` element names to Elegant converter (see #409) (@amylizzle)
- Add `Sextupole` to Bmad, Elegant, and Ocelot converters (see #430) (@Hespe)
- Implement convenience method for quickly setting attributes for all elements of a type in a `Segment` (see #431) (@jank324)
- Add a method to `ParticleBeam` that lets you subsample a particle beam with fewer particles and the same distribution (see #432, #465) (@jank324)
- `Segment` now has new functions `beam_along_segment_generator` and `get_beam_attrs_along_segment` for easily retrieving beam objects and their properties. The plot functions have been refactored to use these, and two functions `plot_beam_attrs` and `plot_beam_attrs_over_lattice` were added for straightforward plotting of different beam attributes in a single line of code. (see #436, #440) (@jank324, @amylizzle)
- `Beam` subclasses now track their `s` position along the beamline (see #436) (@jank324)
- There is a warning now when converting elements from Elegant or Bmad that have names which are invalid for use with the `segment.element_name` syntax, and add a convenience method for explicitly converting these names to valid Python variable names. (see #411) (@amylizzle, @jank324)
- Rotation matrices are no longer computed twice for forward and backward phase space rotations (see #452) (@Hespe)

### 🐛 Bug fixes

- Fix issue where `Dipole` with `tracking_method="bmadx"` and `angle=0.0` would output `NaN` values as a result of a division by zero (see #434) (@jank324)
- Fix issue in CI space-charge tests (incorrect beam duration in non-relativistic case) (see #446) (@RemiLehe)
- Fix issue that passing tensors with `requires_grad=True` does not result in gradient tracked particles when using `ParticleBeam.from_parameters` initialization (see #445) (@cr-xu)
- Fix import of `CustomTransferMap` from Elegant. The affine phase-space component was previously not carried through (see #455) (@Hespe)
- Provide more default values for parameters in the Elegant conversion, where elements without length `l` for example broke the converter. (see #442) (@cr-xu)
- Functions using `Sextupole.split` and `Sextupole.plot` no longer raise an error (see #453) (@Hespe)

### 🐆 Other

- Bmad is no longer actively run in the test workflows, and comparisons to Bmad are now done on static pre-computed results from Bmad. This also removes the use of Anaconda in the test workflow. (see #429, #431) (@jank324)
- The PyTorch pin to `<=2.6` was removed, as the issue with `abort trap: 6` was caused by Bmad is no longer actively used in the test workflow (see #429, #431) (@jank324)
- There was a temporary pin `snowballstemmer<3.0` for the docs build because of an issue with the latest release. It has since been unpinned again because the release was yanked. Refer to https://github.com/sphinx-doc/sphinx/issues/13533 and https://github.com/snowballstem/snowball/issues/229. (see #436, #438) (@jank324)
- Assert that the last row of a predefined transfer map is always correct when creating a `CustomTransferMap` element (see #462) (@jank324, @Hespe)
- Minimum compatible versions were defined for all dependencies, and tests were added to ensure that the minimum versions are compatible with Cheetah. (see #463) (@Hespe, @jank324)
- Add a `pytest` marker for running tests on all subclasses of `Element`. The marker automatically detects if an MWE has not yet been defined for a subclass and alerts the developer through a test failure. (see #418) (@Hespe, @jank324)

### 🌟 First Time Contributors

- Copilot 🤖

## [v0.7.2](https://github.com/desy-ml/cheetah/releases/tag/v0.7.2) (2025-04-28)

### 🚨 Breaking Changes

- Replace the `Segment.plot_reference_particle_traces` with a clearer visualisation in the for of `Segment.plot_mean_and_std`. This also changes the plot generated by the `Segment.plot_overview` method. (see #392) (@RemiLehe)
- The order of the pixels in `Screen.reading` was changed to start from the bottom left instead of the top left. This is now consistent with the `Screen.pixel_bin_centers` and `Screen.pixel_bin_edges` properties. (see #408) (@jank324)

### 🚀 Features

- Implement `split` method for the `Solenoid` element (see #380) (@cr-xu)
- Implement a more robust RPN parser, fixing a bug where short strings in an Elegant variable definition would cause parsing to fail. (see #387, #417) (@amylizzle, @Hespe, @jank324)
- Add a `Sextupole` element (see #406) (@jank324, @Hespe)

### 🐛 Bug fixes

- Fix issue where semicolons after an Elegant line would cause parsing to fail (see #383) (@amylizzle)
- Fix Twiss plot to plot samples also after elements in nested (see #388) (@RemiLehe)
- Fix issue where generating screen images did not work on GPU because `Screen.pixel_bin_centers` was not on the same device (see #372) (@roussel-ryan, @jank324)
- Fix issue where `Quadrupole.tracking_method` was not preserved on cloning (see #404) (@RemiLehe, @jank324)
- The vertical screen misalignment is now correctly applied to `y` instead of `px` (see #405) (@RemiLehe)
- Fix issues when generating screen images caused by the sign of particle charges (see #394) (@Hespe, @jank324)
- Fix an issue where newer versions of `torch` only accept a `torch.Tensor` as input to `torch.rad2deg` (see #417) (@jank324)
- Fix bug that caused correlations to be lost in the conversion from a `ParameterBeam` to a `ParticleBeam` (see #408) (@jank324, @Hespe)

### 🐆 Other

- Temporarily limit `torch` dependency to `2.6` or lower to avoid `abort trap: 6` error with `2.7` (at least on macOS) (see #419) (@jank324)

### 🌟 First Time Contributors

- Amelia Pollard (@amylizzle)

## [v0.7.1](https://github.com/desy-ml/cheetah/releases/tag/v0.7.1) (2025-03-21)

### 🚨 Breaking Changes

- The `incoming` argument of `Segment.plot_overview` is no longer optional. This change also affects the order of the arguments. Fixes an exception that was raised by an underlying plot function that requires `incoming` to be set. (see #316, #344) (@Hespe)
- Python 3.9 is no longer supported. This does not immediately break existing code, but might cause it to break in the future. (see #325) (@jank324)
- The covariance properties of the different beam classes were renamed from names like `cor_x` and `sigma_xpx` to consistent names like `cov_xpx` (see #331) (@jank324)
- The signature of the `transfer_map` method of all element subclasses was extended by a non-optional `species` argument (see #276) (@cr-xu, @jank324, @Hespe)
- `ParticleBeam.plot_distribution` allows for Seaborn-style passing of `axs` and returns the latter as well. In line with that change for the purpose of overlaying distributions, the `contour` argument of `ParticleBeam.plot_2d_distribution` was replaced by a `style` argument. (see #330) (@jank324)
- The default values for `total_charge` in both beam classes are no longer `0.0` but more sensible values (see #377) (@jank324)
- `ParameterBeam._mu` and `ParameterBeam._cov` were renamed to `ParameterBeam.mu` and `ParameterBeam.cov` (see #378) (@jank324)

### 🚀 Features

- `ParticleBeam` now supports importing from and exporting to [openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics) HDF5 files and `ParticleGroup` objects. This allows for easy conversion to and from other file formats supported by openPMD-beamphysics. (see #305, #320) (@cr-xu, @Hespe)
- Add `marker`, `quadrupole` and `csbend` element names to the Elegant converter (see #327) (@jank324)
- Add Python 3.13 support (see #275) (@jank324)
- Methods `to_parameter_beam` and `to_particle_beam` have been added for convenient conversion between `ParticleBeam` and `ParameterBeam` (see #331) (@jank324)
- Beam classes now have the `mu_tau` and `mu_p` properties on their interfaces (see #331) (@jank324)
- Lattice and beam converters now adhere to the default torch `dtype` when no explicit `dtype` is passed (see #340) (@Hespe, @jank324)
- Add options to include or exclude the first and last element when retrieving a `Segment.subcell` and improve error handling (see #350) (@Hespe, @jank324)
- Add support for particle species through a new `Species` class (see #276, #376) (@cr-xu, @jank324, @Hespe)
- Various optimisations for a roughly 2x speed improvement over `v0.7.0` (see #367) (@jank324, @Hespe)

### 🐛 Bug fixes

- Fix issue where a space before a comma could cause the Elegant and Bmad converters to fail (see #327) (@jank324)
- Fix issue of `BPM` and `Screen` not properly converting the `dtype` of their readings (see #335) (@Hespe)
- Fix `is_active` and `is_skippable` of some elements not being boolean properties (see #357) (@jank324)

### 🐆 Other

- Test tolerances were adjusted reduce the chance of random test failures (see #309, #324) (@Hespe, @jank324)
- The copyright years were updated to 2025 (see #318) (@jank324)
- The broken institution logo rendering in the documentation has been fixed (see #318) (@jank324)
- Added `pyproject.toml` to conform with PEP 660 as enforced as of pip 25 for editable installs (see #334) (@jank324)
- Add TUHH logo to contributing institution logos (see #338) (@jank324)
- The tests for backward-mode differentiation with space charge was improved by checking the accuracy of the gradients (see #339) (@RemiLehe)
- A tests for forward-mode differentiation with space charge was added (see #339) (@RemiLehe)
- Link to different ImpactX example in test docstring (see #341) (@ax3l)
- Add link to the new Discord server (see #355, #382) (@jank324)
- Fix typo that said "quadrupole" in a dipole docstring (see #358) (@jank324)
- Type annotations were updated to the post-PEP 585/604... style (see #360) (@jank324)
- Add badge to the README for the number of downloads from PyPI (see #364) (@jank324)

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
