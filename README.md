__UNDER CONSTRUCTION:__ This project is currently undergoing a major refit. Formely known as _JOSS_, it will emerge as _Cheetah_. __Note__, that code breaking changes will occur in the meantime.

# JOSS

JOSS (Jan and Oliver's Simulation Software) is a particle tracking accelerator we built specifically to speed up the training of reinforcement learning models.

## Installation

First, you need to install the [_Ocelot_](https://github.com/ocelot-collab/ocelot) package manually as it is not currently available on PyPI.

Then, simply `git clone` this repository to your machine, change into the directory and run

```bash
pip3 install .
```

to install JOSS.

## How To Use

It is unlikely that you will need to use JOSS by itself, as its main purpose in life is being the default particle tracking backend for the [_Accelerator-Environments_](https://github.com/desy-ml/accelerator-environments) project. Nonetheless, here is a quick example of how JOSS is currently used in our RL environments.

To create a JOSS `Segment` by defining a cell and creating a segment from it as follows

```python
segment = Segment([[BPM(name="BPM1SMATCH"),
                    Drift(length=1.0),
                    BPM(name="BPM6SMATCH"),
                    Drift(length=1.0),
                    VerticalCorrector(length=0.3, name="V7SMATCH"),
                    Drift(length=0.2),
                    HorizontalCorrector(length=0.3, name="H10SMATCH"),
                    Drift(length=7.0),
                    HorizontalCorrector(length=0.3, name="H12SMATCH"),
                    Drift(length=0.05),
                    BPM(name="BPM13SMATCH")])
```

Alternatively you can create a segment from an Ocelot cell by running

```python
segment = joss.Segment(cell)
```

Assuming in `cell` there exists a quadrupole that goes by the ID *AREAMQZM2*, the quadrupole's strength *k* can be changed by calling

```python
segment.AREAMQZM2.k1 = 4.2
```

In order to track a beam through the segment, simply call it like so

```python
outgoing_beam = segment(incoming_beam)
````

You can create particles a random beam using JOSS' `Beam` class by running

```python
beam = Beam.make_random()
```

or by converting an Ocelot `ParticleArray`

```python
beam = Beam.from_ocelot(parray)
```

You may plot a segment with reference particle traces bay calling

```python
segment.plot_overview(beam=beam)
```

![Overview Plot](images/misalignment.png)

where the optional keyword argument `particles` is the incoming particles from which the reference particles are created. JOSS will use its own incoming particles, if you do not pass any.
