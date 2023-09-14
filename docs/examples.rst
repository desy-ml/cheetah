.. Documents error.py

Examples
========

This section contains a few examples of how to use *Cheetah*. These are meant to give you a quick overview of the most important features provided by *Cheetah* and how to use them.

Please feel free to open an issue on GitHub if you feel like another example should be included here.


Tracking through a custom lattice
---------------------------------

In this example, we create a custom lattice and track a beam through it. We start with some imports

.. code-block:: python

    import torch
    from cheetah import ( 
        BPM, Drift, HorizontalCorrector, ParticleBeam, Segment, VerticalCorrector
    )

Lattices in *Cheetah* are represented by `Segments`. A `Segment` is created as follows

.. code-block:: python

    segment = Segment(elements=[
        BPM(name="BPM1SMATCH"),
        Drift(length=torch.tensor(1.0)),
        BPM(name="BPM6SMATCH"),
        Drift(length=torch.tensor(1.0)),
        VerticalCorrector(length=torch.tensor(0.3), name="V7SMATCH"),
        Drift(length=torch.tensor(0.2)),
        HorizontalCorrector(length=torch.tensor(0.3), name="H10SMATCH"),
        Drift(length=torch.tensor(7.0)),
        HorizontalCorrector(length=torch.tensor(0.3), name="H12SMATCH"),
        Drift(length=torch.tensor(0.05)),
        BPM(name="BPM13SMATCH"),
    ])

**Note** that many values must be passed to lattice elements as `torch.Tensor`s. This is because *Cheetah* uses automatic differentiation to compute the gradient of the beam position at the end of the lattice with respect to the element strengths. This is necessary for gradient-based magnet setting optimisation.

Named lattice elements (i.e. elements that were given a `name` keyword argument) can be accessed by name and their parameters changed like so

.. code-block:: python

    segment.V7SMATCH.angle = torch.tensor(3.142e-3)

Next, we create a beam to track through the lattice. In this particular example, we import a beam from an Astra particle distribution file. Note that we are using a `ParticleBeam` here, which is a beam defined by individual particles. This is the most precise way to track a beam through a lattice, but also slower than the alternative `ParameterBeam` which is defined by the beam's parameters. Instead of importing beams from other simulation codes, you can also create beams from scratch, either using their parameters or their Twiss parameters.

.. code-block:: python

    beam = ParticleBeam.from_astra("benchmark/astra/ACHIP_EA1_2021.1351.001")


In order to track a beam through the segment, simply call the segment's `track` method

.. code-block:: python

    outgoing_beam = segment.track(incoming_beam)

You may plot a segment with reference particle traces bay calling

.. code-block:: python

    segment.plot_overview(beam=beam)

.. image:: _static/misalignment.png

where the optional keyword argument `beam` is the incoming beam represented by the reference particles. Cheetah will use a default incoming beam, if no beam is passed.


Tracking an Ocelot beam through an Ocelot lattice
-------------------------------------------------

In this example, we demonstrate 


Gradient-based magnet setting optimisation
------------------------------------------


Normalising parameters in gradient-based optimisation
-----------------------------------------------------
