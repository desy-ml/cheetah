![format](https://github.com/desy-ml/cheetah/actions/workflows/format.yaml/badge.svg)
![pytest](https://github.com/desy-ml/cheetah/actions/workflows/pytest.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/cheetah-accelerator/badge/?version=latest)](https://cheetah-accelerator.readthedocs.io/en/latest/?badge=latest)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI Downloads](https://img.shields.io/pypi/dm/cheetah-accelerator)](https://pypi.org/project/cheetah-accelerator)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/desy-ml/cheetah.git/HEAD?urlpath=%2Fdoc%2Ftree%2Fdocs%2Fexamples%2Fsimple.ipynb)

<!-- [![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master) -->

# Cheetah

<img src="https://github.com/desy-ml/cheetah/raw/master/images/logo.png" align="right" width="25%"/>

Cheetah is a high-speed differentiable beam dynamics code specifically design to support machine learning applications for particle accelerators.

Its speed helps generate data fast, for example for extremely data-hungry reinforcement learning algorithms, while its differentiability can be used for a variety of applications, including accelerator tuning, system identification and physics-informed prior means for Bayesian optimisation. Its native integration with machine learning toolchains around PyTorch also makes Cheetah an ideal candidate for coupling of physics-based and neural network beam dynamics models that remain fast and differentiable.

To learn more about what Cheetah can do, we recommend reading our [PRAB paper](https://doi.org/10.1103/PhysRevAccelBeams.27.054601). To learn how to use Cheetah, we refer to the example notebooks in the [Cheetah documentation](https://cheetah-accelerator.readthedocs.io/). We also have a public [Discord server](https://discord.gg/7k7kzSDwpn) where you can ask questions and get help.

## Installation

Simply install _Cheetah_ from PyPI by running the following command.

```bash
pip install cheetah-accelerator
```

## How To Use

**You can run the following example in Binder by clicking [here](https://mybinder.org/v2/gh/desy-ml/cheetah.git/HEAD?urlpath=%2Fdoc%2Ftree%2Fdocs%2Fexamples%2Fsimple.ipynb) or on the badge above.**

In this example, we create a custom lattice and track a beam through it. We start with some imports.

```python
import cheetah
import torch
```

Lattices in _Cheetah_ are represented by `Segments`. A `Segment` is created as follows.

```python
segment = cheetah.Segment(
    elements=[
        cheetah.Drift(length=torch.tensor(0.175)),
        cheetah.Quadrupole(length=torch.tensor(0.122), name="AREAMQZM1"),
        cheetah.Drift(length=torch.tensor(0.428)),
        cheetah.Quadrupole(length=torch.tensor(0.122), name="AREAMQZM2"),
        cheetah.Drift(length=torch.tensor(0.204)),
        cheetah.VerticalCorrector(length=torch.tensor(0.02), name="AREAMCVM1"),
        cheetah.Drift(length=torch.tensor(0.204)),
        cheetah.Quadrupole(length=torch.tensor(0.122), name="AREAMQZM3"),
        cheetah.Drift(length=torch.tensor(0.179)),
        cheetah.HorizontalCorrector(length=torch.tensor(0.02), name="AREAMCHM1"),
        cheetah.Drift(length=torch.tensor(0.45)),
        cheetah.Screen(name="AREABSCR1"),
    ]
)
```

Alternatively you can load lattices from Cheetah's variant of LatticeJSON or convert them from an Ocelot cell

```python
lattice_json_segment = cheetah.Segment.from_lattice_json("lattice_file.json")
segment = cheetah.Segment.from_ocelot(cell)
```

**Note** that many values must be passed to lattice elements as `torch.Tensor`s. This is because _Cheetah_ uses automatic differentiation to compute the gradient of the beam position at the end of the lattice with respect to the element strengths. This is necessary for gradient-based magnet setting optimisation.

Named lattice elements (i.e. elements that were given a `name` keyword argument) can be accessed by name and their parameters changed like so.

```python
segment.AREAMQZM1.k1 = torch.tensor(8.2)
segment.AREAMQZM2.k1 = torch.tensor(-14.3)
segment.AREAMCVM1.angle = torch.tensor(9e-5)
segment.AREAMQZM3.k1 = torch.tensor(3.142)
segment.AREAMCHM1.angle = torch.tensor(-1e-4)
```

Cheetah has two different beam classes: `ParticleBeam` and `ParameterBeam`. The former tracks multiple individual macroparticles for high-fidelity results, while the latter tracks the parameters of a particle distribution to save on compute time and memory.

You can create a beam manually by specifying the beam parameters of a Gaussian distributed beam

```python
parameter_beam = cheetah.ParameterBeam.from_twiss(beta_x=torch.tensor(3.14))
particle_beam = cheetah.ParticleBeam.from_twiss(
    beta_x=torch.tensor(3.14), beta_y=torch.tensor(42.0), num_particles=10_000
)
```

or load beams from other codes and standards, including openPMD, Ocelot and Astra.

```python
astra_beam = cheetah.ParticleBeam.from_astra(
    "../../tests/resources/ACHIP_EA1_2021.1351.001"
)
```

In order to track a beam through the segment, simply call the segment's `track` method.

```python
outgoing_beam = segment.track(astra_beam)
```

You may plot a segment with reference particle traces by calling

```python
segment.plot_overview(incoming=astra_beam, resolution=0.05)
```

![Overview Plot](https://github.com/desy-ml/cheetah/raw/master/images/readme_overview_plot.png)

where the keyword argument `incoming` is the incoming beam represented by the reference particles.

You can also visualise your segment in 3D. **Note** that this requires that you installed Cheetah as `pip install cheetah-accelerator[3d-visualization]`.

Use `mesh.show` to view the mesh and `mesh.export` to export it to a file.

```python
mesh, _ = segment.to_mesh(
    cuteness={cheetah.HorizontalCorrector: 2.0, cheetah.VerticalCorrector: 2.0}
)
mesh.show()
```

![Animated Mesh](https://github.com/desy-ml/cheetah/raw/master/images/animated_mesh.gif)

```python
mesh.export(file_obj="my_first_cheetah_mesh.glb", file_type="glb")
```

**For more demos and examples check out the _Examples_ section in the [Cheetah documentation](https://cheetah-accelerator.readthedocs.io/en/latest/) and the [`cheetah-demos`](https://github.com/desy-ml/cheetah-demos) repository.**

## Cite Cheetah

If you use Cheetah, please cite the two papers below.

If you use 3D meshes generated by Cheetah, please respect the licencing terms of the **[_3D Assets for Particle Accelerators_](https://github.com/desy-ml/3d-assets) repository**.

```bibtex
@article{kaiser2024cheetah,
    title        = {Bridging the gap between machine learning and particle accelerator physics with high-speed, differentiable simulations},
    author       = {Kaiser, Jan and Xu, Chenran and Eichler, Annika and Santamaria Garcia, Andrea},
    year         = 2024,
    month        = {May},
    journal      = {Phys. Rev. Accel. Beams},
    publisher    = {American Physical Society},
    volume       = 27,
    pages        = {054601},
    doi          = {10.1103/PhysRevAccelBeams.27.054601},
    url          = {https://link.aps.org/doi/10.1103/PhysRevAccelBeams.27.054601},
    issue        = 5,
    numpages     = 17
}
@inproceedings{stein2022accelerating,
    title        = {Accelerating Linear Beam Dynamics Simulations for Machine Learning Applications},
    author       = {Stein, Oliver and Kaiser, Jan and Eichler, Annika},
    year         = 2022,
    booktitle    = {Proceedings of the 13th International Particle Accelerator Conference}
}
```

## For Developers

Activate your virtual environment. (Optional)

Install the `cheetah` package as editable

```sh
pip install -e .
```

We suggest installing pre-commit hooks to automatically conform with the code formatting in commits:

```sh
pip install pre-commit
pre-commit install
```

## Acknowledgements

### Author Contributions

The following people have contributed to the development of Cheetah:

- Jan Kaiser (@jank324)
- Chenran Xu (@cr-xu)
- Annika Eichler (@AnEichler)
- Andrea Santamaria Garcia (@ansantam)
- Christian Hespe (@Hespe)
- Oliver Stein (@OliStein523)
- Grégoire Charleux (@greglenerd)
- Remi Lehe (@RemiLehe)
- Axel Huebl (@ax3l)
- Juan Pablo Gonzalez-Aguilera (@jp-ga)
- Ryan Roussel (@roussel-ryan)
- Auralee Edelen (@lee-edelen)
- Christian Contreras-Campana (@chrisjcc)
- Sucheth Shenoy (@SuchethShenoy)
- Amelia Pollard (@amylizzle)
- Julian Gethmann (@smartsammler)

### Institutions

The development of Cheetah is a joint effort by members of the following institutions:

<img src="https://github.com/desy-ml/cheetah/raw/master/images/desy.png" alt="DESY" style="width: 5em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/kit.png" alt="KIT" style="width: 7em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/lbnl.png" alt="LBNL" style="width: 11em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/university_of_chicago.png" alt="University of Chicago" style="width: 11em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/slac.png" alt="SLAC" style="width: 9em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/university_of_liverpool.png" alt="University of Liverpool" style="width: 10em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/cockcroft.png" alt="Cockcroft Institute" style="width: 7em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/tuhh.png" alt="Hamburg University of Technology" style="width: 5em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/stfc_ukri.png" alt="Science and Technology Facilities Council" style="width: 8em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/argonne.png" alt="Argonne National Laboratory" style="width: 9em;" vspace="2em"/>&nbsp;&nbsp;
<img src="https://github.com/desy-ml/cheetah/raw/master/images/hoou.png" alt="Hamburg Open Online University" style="width: 7em;" vspace="2em"/>&nbsp;&nbsp;

### Funding

The work to develop Cheetah has in part been funded by the IVF project InternLabs-0011 (HIR3X) and the Initiative and Networking Fund by the Helmholtz Association (Autonomous Accelerator, ZT-I-PF-5-6).
Further, we gratefully acknowledge funding by the EuXFEL R&D project "RP-513: Learning Based Methods".
This work is also supported by the U.S. Department of Energy, Office of Science under Contract No. DE-AC02-76SF00515, the Center for Bright Beams, NSF Award No. PHY-1549132, and the U.S. DOE Office of Science-Basic Energy Sciences, under Contract No. DE-AC02-06CH11357.
In addition, we acknowledge support from DESY (Hamburg, Germany) and KIT (Karlsruhe, Germany), members of the Helmholtz Association HGF as well as from the Hamburg Open Online University (HOOU) and the Science and Technology Facilities Council (UK).
