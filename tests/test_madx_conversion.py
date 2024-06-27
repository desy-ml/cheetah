import importlib
import os
import sys

import numpy as np
import pandas as pd

import cheetah
from cheetah.converters.madx import *


def test_madx_tfs():
    madx_tfs_file_path = "tests/resources/twiss_tt43_nom.tfs"
    converted_segment = convert_madx_lattice(madx_tfs_file_path)
    assert isinstance(converted_segment.elements[0], cheetah.RBend)

test_madx_tfs()