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

A sequence of accelerator elements (or a lattice) is called a `Segment` in _Cheetah_. You can create a `Segment` as follows

```python
import cheetah
import torch

segment = cheetah.Segment(
    elements=[
        cheetah.Drift(length=torch.tensor(0.175)),
        cheetah.Quadruple(length=torch.tensor(0.122), name="AREAMQZM1"),
        cheetah.Drift(length=torch.tensor(0.428)),
        cheetah.Quadruple(length=torch.tensor(0.122), name="AREAMQZM2"),
        cheetah.Drift(length=torch.tensor(0.204)),
        cheetah.VerticalCorrector(length=torch.tensor(0.02), name="AREAMCVM1"),
        cheetah.Drift(length=torch.tensor(0.204)),
        cheetah.Quadruple(length=torch.tensor(0.122), name="AREAMQZM3"),
        cheetah.Drift(length=torch.tensor(0.179)),
        cheetah.HorizontalCorrector(length=torch.tensor(0.02), name="AREAMCHM1"),
        cheetah.Drift(length=torch.tensor(0.45)),
        cheetah.Screen(name="AREABSCR1"),
    ]
)
```

Alternatively you can create a segment from an Ocelot cell by running

```python
segment = cheetah.Segment.from_ocelot(cell)
```

All elements can be accesses as a property of the segment via their name. The strength of a quadrupole named _AREAMQZM2_ for example, may be set by running

```python
segment.AREAMQZM2.k1 = torch.tensor(4.2)
```

You can choose to track either a beam defined by its parameters (fast) or by its particles (precise). _Cheetah_ defines two different beam classes for this purpose and beams may be created by

```python
parameter_beam = cheetah.ParameterBeam.from_twiss(beta_x=torch.tensor(3.14))
particle_beam = cheetah.ParticleBeam.from_twiss(
    beta_x=torch.tensor(3.14), num_particles=10_000
)
```

It is also possible to load beams from Ocelot `ParticleArray` or Astra particle distribution files for both types of beam

```python
ocelot_beam = cheetah.ParticleBeam.from_ocelot(parray)
astra_beam = cheetah.ParticleBeam.from_astra(filepath)
```

In order to track a beam through the segment, simply call the segment like so

```python
outgoing_beam = segment.track(astra_beam)
```

You may plot a segment with the beam position and size by calling

```python
segment.plot_overview(incoming=beam)
```

![Overview Plot](https://github.com/desy-ml/cheetah/raw/master/images/readme_overview_plot.png)

where the keyword argument `incoming` is the incoming beam represented in the plot.

**For more demos check out the [`cheetah-demos`](https://github.com/desy-ml/cheetah-demos) repository.**

## Cite Cheetah

If you use Cheetah, please cite the following two papers:

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
<img src="https://github.com/desy-ml/cheetah/raw/master/images/argonne.png" alt="Argonne National Laboratory" style="width: 9em;" vspace="2em"/>

### Funding

The work to develop Cheetah has in part been funded by the IVF project InternLabs-0011 (HIR3X) and the Initiative and Networking Fund by the Helmholtz Association (Autonomous Accelerator, ZT-I-PF-5-6).
Further, we gratefully acknowledge funding by the EuXFEL R&D project "RP-513: Learning Based Methods".
This work is also supported by the U.S. Department of Energy, Office of Science under Contract No. DE-AC02-76SF00515, the Center for Bright Beams, NSF Award No. PHY-1549132, and the U.S. DOE Office of Science-Basic Energy Sciences, under Contract No. DE-AC02-06CH11357.
In addition, we acknowledge support from DESY (Hamburg, Germany) and KIT (Karlsruhe, Germany), members of the Helmholtz Association HGF as well as from the Science and Technology Facilities Council (UK).
