.. Cheetah documentation master file, created by
   sphinx-quickstart on Fri May 19 10:20:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Cheetah's documentation!
===================================

`Cheetah <https://github.com/desy-ml/cheetah>`_ is a particle tracking accelerator we built specifically to speed up the training of reinforcement learning models.

GitHub repository: https://github.com/desy-ml/cheetah

Paper: https://doi.org/10.1103/PhysRevAccelBeams.27.054601

Discord server: https://discord.gg/hrwYPC3a


Installation
------------

Simply install *Cheetah* from PyPI by running the following command.

.. code-block:: bash

    pip install cheetah-accelerator


Examples
--------

We provide some examples to demonstrate some features of *Cheetah* and show how to use them. They provide a good entry point to using *Cheetah*, but they do not represent its full functionality. To move beyond the examples, please refer to the in-depth documentation. If you feel like other examples should be added, feel free to open an issue on GitHub.

.. toctree::
    :maxdepth: 2
    :caption: Examples

    examples/simple
    examples/convert
    examples/optimize_speed
    examples/gradientbased

Getting Started
---------------

These pages explain how to get started with *Cheetah*.

.. toctree::
    :maxdepth: 1
    :caption: Getting Started
    
    coordinate_system.md

Documentation
-------------

For more advanced usage, please refer to the in-depth documentation.

.. toctree::
    :maxdepth: 1
    :caption: Documentation

    accelerator
    converters
    latticejson
    particles
    track_methods
    utils


Cite Cheetah
------------

If you use Cheetah, please cite the following two papers:

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

Institutions
~~~~~~~~~~~~

The development of Cheetah is a joint effort by members of the following institutions:

.. image:: https://github.com/desy-ml/cheetah/raw/master/images/desy.png
    :alt: DESY
    :width: 5em

.. image:: https://github.com/desy-ml/cheetah/raw/master/images/kit.png
    :alt: KIT
    :width: 7em

.. image:: https://github.com/desy-ml/cheetah/raw/master/images/lbnl.png
    :alt: LBNL
    :width: 11em

.. image:: https://github.com/desy-ml/cheetah/raw/master/images/university_of_chicago.png
    :alt: University of Chicago
    :width: 11em

.. image:: https://github.com/desy-ml/cheetah/raw/master/images/slac.png
    :alt: SLAC
    :width: 9em

.. image:: https://github.com/desy-ml/cheetah/raw/master/images/university_of_liverpool.png
    :alt: University of Liverpool
    :width: 10em

.. image:: https://github.com/desy-ml/cheetah/raw/master/images/cockcroft.png
    :alt: Cockcroft Institute
    :width: 7em

.. image:: https://github.com/desy-ml/cheetah/raw/master/images/tuhh.png
    :alt: Hamburg University of Technology
    :width: 5em

Funding
~~~~~~~

The work to develop Cheetah has in part been funded by the IVF project InternLabs-0011 (HIR3X) and the Initiative and Networking Fund by the Helmholtz Association (Autonomous Accelerator, ZT-I-PF-5-6).
Further, we gratefully acknowledge funding by the EuXFEL R&D project "RP-513: Learning Based Methods".
This work was also supported by the U.S. National Science Foundation under Award PHY-1549132, the Center for Bright Beams.
In addition, we acknowledge support from DESY (Hamburg, Germany) and KIT (Karlsruhe, Germany), members of the Helmholtz Association HGF.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
