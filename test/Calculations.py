import sys

import cheetah
import numpy as np
import torch

sys.path.append(
    "c:/users/ftheilen/appdata/local/packages/pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0/localcache/local-packages/python310/site-packages"
)

"""
Test Beam, which can be found in GitHub in the folder benchmark/cheetah/ACHIP_EA1_2021.1351.001
"""

beam1 = cheetah.ParameterBeam.from_astra(
    "D:/Fachpraktikum_DESY/GitHub/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)
beam2 = cheetah.ParticleBeam.from_astra(
    "D:/Fachpraktikum_DESY/GitHub/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)

segment = cheetah.Segment(
    [cheetah.HorizontalCorrector(length=0.02, name="quad"), cheetah.Drift(length=2.0)]
)
segment.quad.angle = 2e-3

result1 = segment(beam1)
result2 = segment(beam2)

print("ParameterBeam")
print(result1._mu)

print("ParticleBeam")
print(result2.particles.mean(axis=0))
