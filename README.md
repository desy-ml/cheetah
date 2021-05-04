# JOSS

JOSS (Jan and Oliver's Simulation Software) is a particle tracking accelerator we built specifically to speed up the training of reinforcement learning models.

## Installation

Simply `git clone` this repository to your machine, change into the directory and run

```bash
pip3 install .
```

to install JOSS. Once the installation is finished, you will need to install the [_Ocelot_](https://github.com/ocelot-collab/ocelot) package manually as it is not currently available on PyPI.

## How To Use

It is unlikely that you will need to use JOSS by itself, as its main purpose in life is being the default particle tracking backend for the [_Accelerator-Environments_](https://github.com/desy-ml/accelerator-environments) project. Nonetheless, here is a quick example of how JOSS is currently used in our RL environments.

At this point in development, JOSS is rather integrated with Ocelot, because of the number of Ocelot-defined accelerator sections we use. You will therefore need an Ocelot cell defined as a list of Ocelot elements. We call this variable `cell`. You can create a JOSS `Segment` from this cell by running

```python
segment = joss.Segment(cell)
```

Assuming in `cell` there exists a quadrupole that goes by the ID *AREAMQZM2*, the quadrupole's strength *k* can be changed by calling

```python
segment.AREAMQZM2.k1 = 4.2
```

In order to track particles through the segment, simply call it like so

```python
particles = segment(particles)
````

You can create particles either by calling one of JOSS's particle creation functions

```python
particles1 = joss.random_particles()
particles2 = joss.linspaced_particles()
```

or by converting an Ocelot `ParticleArray`

```python
particles = joss.ocelot_parray_to_joss_particles(parray)
```

You may plot a segment with reference particle traces bay calling

```python
segment.plot_overview(particles=particles)
```

![Overview Plot](images/misalignment.png)

where the optional keyword argument `particles` is the incoming particles from which the reference particles are created. JOSS will use its own incoming particles, if you do not pass any.
