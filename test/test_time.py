import cheetah
import timeit
import numpy as np

import ARESlatticeStage3v1_9 as lattice

beam1 = cheetah.ParameterBeam.from_astra("C:/Users/ftheilen/Source/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001")
beam2 = cheetah.ParticleBeam.from_astra("C:/Users/ftheilen/Source/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001")

segment = cheetah.Segment.from_ocelot(lattice.cell, warnings=False).subcell("AREASOLA1", "AREABSCR1")
segment.AREABSCR1.binning = 4
segment.AREABSCR1.is_active = False
segment.AREAMQZM1.k1 = 3.1
segment.AREAMQZM2.k1 = -3.1
segment.AREAMCVM1.angle = 1e-3
segment.AREAMQZM3.k1 = 4.2
segment.AREAMCHM1.angle = 2e-3

def test_timeit1():
    measured_time = timeit.timeit(lambda: segment(beam1), number = 10000) / 10000
    expected_time = 86.6e-06
    assert np.isclose(measured_time, expected_time, rtol=0, atol=1e-05, equal_nan=False)

def test_timeit2():
    measured_time = timeit.timeit(lambda: segment(beam2), number = 10000) / 10000
    expected_time = 955e-06
    assert np.isclose(measured_time, expected_time, rtol=0, atol=1e-04, equal_nan=False)
