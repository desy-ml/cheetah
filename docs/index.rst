.. Cheetah documentation master file, created by
   sphinx-quickstart on Fri May 19 10:20:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Cheetah's documentation!
===================================

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    examples

`Cheetah <https://github.com/desy-ml/cheetah>`_ is a particle tracking accelerator we built specifically to speed up the training of reinforcement learning models.

GitHub repository: https://github.com/desy-ml/cheetah

Paper: https://accelconf.web.cern.ch/ipac2022/papers/wepoms036.pdf


Installation
------------

Simply install *Cheetah* from PyPI by running the following command.

.. code-block:: bash

    pip install cheetah-accelerator


Cite Cheetah
------------

To cite Cheetah in publications:

.. code-block:: bibtex

    @inproceedings{stein2022accelerating,
        author = {Stein, Oliver and
                Kaiser, Jan and
                Eichler, Annika},
        title = {Accelerating Linear Beam Dynamics Simulations for Machine Learning Applications},
        booktitle = {Proceedings of the 13th International Particle Accelerator Conference},
        year = {2022},
        url = {https://github.com/desy-ml/cheetah},
    }

For Developers
--------------

Activate your virtual envrionment. (Optional)

Install the cheetah package as editable

.. code-block:: sh

    pip install -e .

We suggest to install pre-commit hooks to automatically conform with the code formatting in commits:

.. code-block:: sh

    pip install pre-commit
    pre-commit install


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
