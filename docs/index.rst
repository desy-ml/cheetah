.. Cheetah documentation master file, created by
   sphinx-quickstart on Fri May 19 10:20:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Cheetah's documentation!
===================================

`Cheetah <https://github.com/desy-ml/cheetah>`_ is a particle tracking accelerator we built specifically to speed up the training of reinforcement learning models.

GitHub repository: https://github.com/desy-ml/cheetah

Paper: https://arxiv.org/abs/2401.05815


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


Documentation
-------------

For more advanced usage, please refer to the in-depth documentation.

.. toctree::
    :maxdepth: 1
    :caption: Documentation

    accelerator
    astralavista
    dontbmad
    error
    latticejson
    nocelot
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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
