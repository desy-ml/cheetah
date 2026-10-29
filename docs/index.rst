.. Cheetah documentation master file, created by
   sphinx-quickstart on Fri May 19 10:20:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Cheetah's documentation!
===================================

|pypi| |downloads| |pyversions| |ci| |license| |discord|

.. |pypi| image:: https://img.shields.io/pypi/v/cheetah-accelerator.svg
    :target: https://pypi.org/project/cheetah-accelerator/
    :alt: PyPI Version

.. |downloads| image:: https://img.shields.io/pypi/dm/cheetah-accelerator.svg
    :target: https://pypi.org/project/cheetah-accelerator/
    :alt: PyPI Downloads

.. |pyversions| image:: https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg?logo=python&logoColor=white
    :target: https://pypi.org/project/cheetah-accelerator/
    :alt: Python Version

.. |ci| image:: https://github.com/desy-ml/cheetah/actions/workflows/pytest.yaml/badge.svg
    :target: https://github.com/desy-ml/cheetah/actions/workflows/pytest.yaml
    :alt: CI Status

.. |license| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://www.gnu.org/licenses/gpl-3.0
    :alt: License: GPL v3

.. |discord| image:: https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white
    :target: https://discord.gg/7k7kzSDwpn
    :alt: Discord


*Cheetah* is a high-speed, differentiable particle tracking engine designed specifically for machine learning applications in accelerator physics, such as reinforcement learning, system identification, and magnet tuning.

Resources & Links
-----------------

* **Code**: `GitHub Repository <https://github.com/desy-ml/cheetah>`_
* **Paper**: `Physical Review Accelerators and Beams <https://doi.org/10.1103/PhysRevAccelBeams.27.054601>`_
* **Community**: `Discord Server <https://discord.gg/hrwYPC3a>`_
* **Demos**: `Cheetah Demos <https://github.com/desy-ml/cheetah-demos>`_ (examples including system identification, magnet tuning, and reinforcement learning)

Key Features
------------

* **Differentiable Physics**: Fully integrated with PyTorch's automatic differentiation (autograd) to allow backpropagation of gradients directly through lattice elements and parameters.
* **High-Speed Execution**: Tailored algorithms optimised to speed up the data generation for training machine learning algorithms and gradient-based parameter optimisation loops.
* **Vectorised Simulations**: Perform parallel parameter sweeps over beam distributions and magnet settings without slow CPU Python loops.
* **Native GPU/CPU Support**: Seamlessly transition computations between CPU and GPU devices (CUDA or MPS) using PyTorch's native ``.to()`` method.
* **3D Visualisation**: Automatic 3D model rendering and mesh export of your lattice geometry for visualisation.

Installation
------------

Simply install *Cheetah* from PyPI by running the following command.

.. code-block:: bash

    pip install cheetah-accelerator


Quickstart
----------

Define a beamline, generate a beam, and track it with PyTorch:

.. code-block:: python

    import torch
    import cheetah

    # Define a simple lattice segment
    lattice = cheetah.Segment(
        elements=[
            cheetah.Quadrupole(length=torch.tensor(0.1), k1=torch.tensor(10.0)),
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Quadrupole(length=torch.tensor(0.1), k1=torch.tensor(-10.0)),
            cheetah.Screen(name="screen"),
        ]
    )

    # Sample a particle beam from Twiss parameters
    beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14),
        beta_y=torch.tensor(4.2),
        energy=torch.tensor(100e6),
        num_particles=10_000,
    )

    # Track through the lattice
    outgoing_beam = lattice.track(beam)



Supported Elements
------------------

Cheetah currently supports the following accelerator elements:

* **Drifts**: :class:`~cheetah.Drift`
* **Magnets**: :class:`~cheetah.Quadrupole`, :class:`~cheetah.Sextupole`, :class:`~cheetah.Solenoid`, :class:`~cheetah.Dipole` (sector bend), and :class:`~cheetah.RBend` (rectangular bend)
* **Correctors**: :class:`~cheetah.HorizontalCorrector`, :class:`~cheetah.VerticalCorrector`, :class:`~cheetah.CombinedCorrector`
* **RF Cavities**: :class:`~cheetah.Cavity`, :class:`~cheetah.TransverseDeflectingCavity`
* **Diagnostics**: :class:`~cheetah.BPM` (Beam Position Monitor), :class:`~cheetah.Screen`
* **Other**: :class:`~cheetah.Aperture`, :class:`~cheetah.Undulator`, :class:`~cheetah.Marker`, :class:`~cheetah.SpaceChargeKick`

If you need an element that is not yet supported, you can open an issue on the `GitHub repository <https://github.com/desy-ml/cheetah>`_ or implement a custom element by subclassing :class:`~cheetah.Element` and implementing the required methods. See :doc:`/getting_started/custom_elements` for more information.

For the full element API documentation, see the :doc:`/accelerator` API Reference.

User Guide
----------

These guides explain how to use *Cheetah* and provide an overview of its features, from basics to advanced topics.

.. grid:: 1 2 2 3
    :gutter: 2

    .. grid-item-card:: 🚀 Getting Started
        :link: /getting_started/simple
        :link-type: doc

        Essential walkthroughs for defining lattices, choosing beam representations, and understanding coordinates.

        * :doc:`/getting_started/simple`
        * :doc:`/getting_started/beam_representations`
        * :doc:`/getting_started/plotting_visualization`
        * :doc:`/getting_started/coordinate_system`

    .. grid-item-card:: ⚛️ Physics
        :link: /getting_started/tracking_methods
        :link-type: doc

        Learn about first- and second-order transfer maps, thin/thick lens modeling, 3D space charge kicks, and virtual diagnostics.

        * :doc:`/getting_started/tracking_methods`
        * :doc:`/getting_started/space_charge`
        * :doc:`/getting_started/diagnostics`

    .. grid-item-card:: 🧠 Differentiable Physics & ML
        :link: /getting_started/gradientbased
        :link-type: doc

        Combine Cheetah with PyTorch automatic differentiation (`autograd`) for gradient-based optimisation and ML workflows.

        * :doc:`/getting_started/gradientbased`

    .. grid-item-card:: ⚡ Optimising Simulation Speed
        :link: /getting_started/vectorized_simulations
        :link-type: doc

        Techniques for vectorised parameter sweeps, GPU acceleration, mixed precision, and PyTorch compilation.

        * :doc:`/getting_started/vectorized_simulations`
        * :doc:`/getting_started/optimising_speed`

    .. grid-item-card:: 🔌 Interoperability
        :link: /getting_started/importing_from_other_codes
        :link-type: doc

        Convert lattices from Bmad, Elegant, and Ocelot, or integrate with live control systems like DOOCS.

        * :doc:`/getting_started/importing_from_other_codes`
        * :doc:`/getting_started/control_systems`

    .. grid-item-card:: 🧩 Extending Cheetah
        :link: /getting_started/custom_elements
        :link-type: doc

        Implement custom accelerator elements by subclassing `Element` with transfer maps or tracking logic.

        * :doc:`/getting_started/custom_elements`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    getting_started/simple
    getting_started/beam_representations.md
    getting_started/plotting_visualization
    getting_started/coordinate_system.md

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Physics

    getting_started/tracking_methods.md
    getting_started/space_charge.md
    getting_started/diagnostics.md

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Differentiable Physics & ML

    getting_started/gradientbased

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Optimising Simulation Speed

    getting_started/vectorized_simulations
    getting_started/optimising_speed

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Interoperability

    getting_started/importing_from_other_codes.md
    getting_started/control_systems.md

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Extending Cheetah

    getting_started/custom_elements

API Reference
-------------

For a complete overview of the *Cheetah* API, please refer to the following pages.

.. toctree::
    :maxdepth: 1
    :caption: API Reference

    accelerator
    converters
    latticejson
    particles
    track_methods
    utils


Cite Cheetah
------------

If you use Cheetah, please cite the two papers below.

If you use the 3D meshes generated by Cheetah, please respect the licensing terms of the `3D Assets for Particle Accelerators <https://github.com/desy-ml/3d-assets>`_ repository.

.. code-block:: bibtex

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


For Developers
--------------

Activate your virtual environment. (Optional)

Install the cheetah package as editable

.. code-block:: sh

    pip install -e .

We suggest installing pre-commit hooks to automatically conform with the code formatting in commits:

.. code-block:: sh

    pip install pre-commit
    pre-commit install


Acknowledgements
----------------

Author Contributions
~~~~~~~~~~~~~~~~~~~~~

The following people have contributed to the development of Cheetah:

- Jan Kaiser (`@jank324 <https://github.com/jank324>`_)
- Chenran Xu (`@cr-xu <https://github.com/cr-xu>`_)
- Annika Eichler (`@AnEichler <https://github.com/AnEichler>`_)
- Andrea Santamaria Garcia (`@ansantam <https://github.com/ansantam>`_)
- Christian Hespe (`@Hespe <https://github.com/Hespe>`_)
- Oliver Stein (`@OliStein523 <https://github.com/OliStein523>`_)
- Grégoire Charleux (`@greglenerd <https://github.com/greglenerd>`_)
- Remi Lehe (`@RemiLehe <https://github.com/RemiLehe>`_)
- Axel Huebl (`@ax3l <https://github.com/ax3l>`_)
- Juan Pablo Gonzalez-Aguilera (`@jp-ga <https://github.com/jp-ga>`_)
- Ryan Roussel (`@roussel-ryan <https://github.com/roussel-ryan>`_)
- Auralee Edelen (`@lee-edelen <https://github.com/lee-edelen>`_)
- Zihan Zhu (`@zihan-zh <https://github.com/zihan-zh>`_)
- Christian Contreras-Campana (`@chrisjcc <https://github.com/chrisjcc>`_)
- Sucheth Shenoy (`@SuchethShenoy <https://github.com/SuchethShenoy>`_)
- Amelia Pollard (`@amylizzle <https://github.com/amylizzle>`_)
- Julian Gethmann (`@smartsammler <https://github.com/smartsammler>`_)

Institutions
~~~~~~~~~~~~

The development of Cheetah is a joint effort by members of the following institutions:

.. container:: institution-logos

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/desy.png
        :alt: DESY

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/kit.png
        :alt: KIT

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/lbnl.png
        :alt: LBNL

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/university_of_chicago.png
        :alt: University of Chicago

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/slac.png
        :alt: SLAC

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/university_of_liverpool.png
        :alt: University of Liverpool

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/cockcroft.png
        :alt: Cockcroft Institute

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/tuhh.png
        :alt: Hamburg University of Technology

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/stfc_ukri.png
        :alt: Science and Technology Facilities Council

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/argonne.png
        :alt: Argonne National Laboratory

    .. image:: https://github.com/desy-ml/cheetah/raw/master/images/hoou.png
        :alt: Hamburg Open Online University

Funding
~~~~~~~

The work to develop Cheetah has in part been funded by the IVF project InternLabs-0011 (HIR3X) and the Initiative and Networking Fund by the Helmholtz Association (Autonomous Accelerator, ZT-I-PF-5-6).
Further, we gratefully acknowledge funding by the EuXFEL R&D project "RP-513: Learning Based Methods".
This work is also supported by the U.S. Department of Energy, Office of Science under Contract No. DE-AC02-76SF00515, the Center for Bright Beams, NSF Award No. PHY-1549132, and the U.S. DOE Office of Science-Basic Energy Sciences, under Contract No. DE-AC02-06CH11357.
In addition, we acknowledge support from DESY (Hamburg, Germany) and KIT (Karlsruhe, Germany), members of the Helmholtz Association HGF as well as from the Hamburg Open Online University (HOOU) and the Science and Technology Facilities Council (UK).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
