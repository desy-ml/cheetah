# Cheetah

Cheetah is a particle tracking accelerator we built specifically to speed up the training of reinforcement learning models.


## Installation

Simply install *Cheetah* from PyPI by running the following command.

```bash
pip install cheetah-accelerator
```


## How To Use

A sequence of accelerator elements (or a lattice) is called a `Segment` in *Cheetah*. You can create a `Segment` as follows

```python
segment = Segment([
    BPM(name="BPM1SMATCH"),
    Drift(length=1.0),
    BPM(name="BPM6SMATCH"),
    Drift(length=1.0),
    VerticalCorrector(length=0.3, name="V7SMATCH"),
    Drift(length=0.2),
    HorizontalCorrector(length=0.3, name="H10SMATCH"),
    Drift(length=7.0),
    HorizontalCorrector(length=0.3, name="H12SMATCH"),
    Drift(length=0.05),
    BPM(name="BPM13SMATCH"),
])
```

Alternatively you can create a segment from an Ocelot cell by running

```python
segment = Segment.from_ocelot(cell)
```

All elements can be accesses as a property of the segment via their name. The strength of a quadrupole named *AREAMQZM2* for example, may be set by running

```python
segment.AREAMQZM2.k1 = 4.2
```

In order to track a beam through the segment, simply call the segment like so

```python
outgoing_beam = segment(incoming_beam)
````

You can choose to track either a beam defined by its parameters (fast) or by its particles (precise). *Cheetah* defines two different beam classes for this purpose and beams may be created by

```python
beam1 = ParameterBeam.from_parameters()
beam2 = ParticleBeam.from_parameters()
```

It is also possible to load beams from Ocelot `ParticleArray` or Astra particle distribution files for both types of beam

```python
ocelot_beam = ParticleBeam.from_ocelot(parray)
astra_beam = ParticleBeam.from_astra(filepath)
```

You may plot a segment with reference particle traces bay calling

```python
segment.plot_overview(beam=beam)
```

![Overview Plot](images/misalignment.png)

where the optional keyword argument `beam` is the incoming beam represented by the reference particles. Cheetah will use a default incoming beam, if no beam is passed.


## Cite Cheetah

To cite Cheetah in publications:

```bibtex
@inproceedings{stein2022accelerating,
    author = {Stein, Oliver and
              Kaiser, Jan and
              Eichler, Annika},
    title = {Accelerating Linear Beam Dynamics Simulations for Machine Learning Applications},
    booktitle = {Proceedings of the 13th International Particle Accelerator Conference},
    year = {2022},
    url = {https://github.com/desy-ml/cheetah},
}
```
