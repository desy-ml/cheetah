from copy import deepcopy

import numpy as np
import ocelot

import cheetah

"""
Test implementation of transfer maps of the cheetah elements,
compare tracking results with OCELOT: https://github.com/ocelot-collab/ocelot
"""

PARTICLEBEAM_CHEETAH = cheetah.ParticleBeam.from_astra(
    "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)

PARRAY_OCELOT = ocelot.astraBeam2particleArray(
    "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)


def test_benchmark_ocelot_dipole():
    length = 0.1
    angle = 0.1
    cheetah_bend = cheetah.Dipole(length=length, angle=angle)
    ocelot_bend = ocelot.Bend(l=length, angle=angle)
    p_array = deepcopy(PARRAY_OCELOT)
    p_in_cheetah = deepcopy(PARTICLEBEAM_CHEETAH)
    pb_out_cheetah = cheetah_bend(p_in_cheetah)

    lat = ocelot.MagneticLattice([ocelot_bend], stop=None)
    navi = ocelot.Navigator(lat)
    _, p_array = ocelot.track(lat, p_array, navi)

    assert np.allclose(
        p_array.rparticles,
        pb_out_cheetah.particles[:, :6].t().numpy(),
        rtol=1e-4,
        atol=1e-10,
        equal_nan=False,
    )


def test_benchmark_ocelot_dipole_with_fringe_field():
    length = 0.1
    angle = 0.1
    fint = 0.1
    gap = 0.2
    cheetah_bend = cheetah.Dipole(length=length, angle=angle, fint=fint, gap=gap)
    ocelot_bend = ocelot.Bend(l=length, angle=angle, fint=fint, gap=gap)
    p_array = deepcopy(PARRAY_OCELOT)
    p_in_cheetah = deepcopy(PARTICLEBEAM_CHEETAH)
    pb_out_cheetah = cheetah_bend(p_in_cheetah)

    lat = ocelot.MagneticLattice([ocelot_bend], stop=None)
    navi = ocelot.Navigator(lat)
    _, p_array = ocelot.track(lat, p_array, navi)

    assert np.allclose(
        p_array.rparticles,
        pb_out_cheetah.particles[:, :6].t().numpy(),
        rtol=1e-4,
        atol=1e-10,
        equal_nan=False,
    )


def test_benchmark_ocelot_aperture():
    xmax = 2e-4
    ymax = 2e-4
    drift_length = 0.1  # so that ocelot starts tracking
    ocelot_aperture = ocelot.Aperture(xmax=xmax, ymax=xmax)
    cheetah_aperture = cheetah.Aperture(xmax=xmax, ymax=ymax)
    cheetah_aperture.is_active = True
    p_array = deepcopy(PARRAY_OCELOT)
    p_in_cheetah = deepcopy(PARTICLEBEAM_CHEETAH)
    # Cheetah Tracking
    segment = cheetah.Segment([cheetah_aperture, cheetah.Drift(length=drift_length)])
    p_out_cheetah = segment(p_in_cheetah)
    # Ocelot Tracking
    lat = ocelot.MagneticLattice(
        [ocelot_aperture, ocelot.Drift(drift_length)], stop=None
    )
    navi = ocelot.Navigator(lat)
    navi.activate_apertures()
    _, p_array = ocelot.track(lat, p_array, navi)

    assert p_out_cheetah.n == p_array.rparticles.shape[1]


def test_benchmark_ocelot_aperture_elliptical():
    xmax = 2e-4
    ymax = 2e-4
    drift_length = 0.1  # so that ocelot starts tracking
    ocelot_aperture = ocelot.Aperture(xmax=xmax, ymax=xmax, type="ellipt")
    cheetah_aperture = cheetah.Aperture(xmax=xmax, ymax=ymax, type="ellipt")
    cheetah_aperture.is_active = True
    p_array = deepcopy(PARRAY_OCELOT)
    p_in_cheetah = deepcopy(PARTICLEBEAM_CHEETAH)
    # Cheetah Tracking
    segment = cheetah.Segment([cheetah_aperture, cheetah.Drift(length=drift_length)])
    p_out_cheetah = segment(p_in_cheetah)
    # Ocelot Tracking
    lat = ocelot.MagneticLattice(
        [ocelot_aperture, ocelot.Drift(drift_length)], stop=None
    )
    navi = ocelot.Navigator(lat)
    navi.activate_apertures()
    _, p_array = ocelot.track(lat, p_array, navi)

    assert p_out_cheetah.n == p_array.rparticles.shape[1]
