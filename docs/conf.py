# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# Source code directory, relative to this file, for sphinx-autobuild
sys.path.insert(0, os.path.abspath("../cheetah"))

from _version import __version__  # noqa: E402

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Cheetah"
copyright = "2021-2025 Jan Kaiser, Chenran Xu, Christian Hespe"
author = "Jan Kaiser, Chenran Xu"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",  # To render Jupyter notebooks
    "sphinx.ext.autodoc",  # To generate documentation from docstrings
    "sphinx.ext.autosummary",  # To automatically generate recursively
    "myst_parser",  # To parse markdown files
]
myst_enable_extensions = ["dollarmath", "amsmath"]  # Enable math typesetting

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",  # Add markdown support
}
autosummary_generate = True  # Turn on autosummary

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_logo = "_static/logo.png"

html_static_path = ["_static"]
