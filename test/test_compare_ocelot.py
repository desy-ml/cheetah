from copy import deepcopy

import numpy as np
import ocelot

import cheetah


def test_dipole():
    """
    Test that the tracking results through a Cheeath `Dipole` element match those
    through an Oclet `Bend` element.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_dipole = cheetah.Dipole(length=0.1, angle=0.1)
    outgoing_beam = cheetah_dipole.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    ocelot_bend = ocelot.Bend(l=0.1, angle=0.1)
    lattice = ocelot.MagneticLattice([ocelot_bend])
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)

    assert np.allclose(
        outgoing_beam.particles[:, :6], outgoing_p_array.rparticles.transpose()
    )


def test_dipole_with_fringe_field():
    """
    Test that the tracking results through a Cheeath `Dipole` element match those
    through an Oclet `Bend` element when there are fringe fields.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_dipole = cheetah.Dipole(length=0.1, angle=0.1, fringe_integral=0.1, gap=0.2)
    outgoing_beam = cheetah_dipole.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    ocelot_bend = ocelot.Bend(l=0.1, angle=0.1, fint=0.1, gap=0.2)
    lattice = ocelot.MagneticLattice([ocelot_bend])
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)

    assert np.allclose(
        outgoing_beam.particles[:, :6], outgoing_p_array.rparticles.transpose()
    )


def test_aperture():
    """
    Test that the tracking results through a Cheeath `Aperture` element match those
    through an Oclet `Aperture` element.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_segment = cheetah.Segment(
        [
            cheetah.Aperture(
                x_max=2e-4,
                y_max=2e-4,
                shape="rectangular",
                name="aperture",
                is_active=True,
            ),
            cheetah.Drift(length=0.1),
        ]
    )
    outgoing_beam = cheetah_segment.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    ocelot_cell = [ocelot.Aperture(xmax=2e-4, ymax=2e-4), ocelot.Drift(l=0.1)]
    lattice = ocelot.MagneticLattice([ocelot_cell])
    navigator = ocelot.Navigator(lattice)
    navigator.activate_apertures()
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)

    assert outgoing_beam.num_particles == outgoing_p_array.rparticles.shape[1]


def test_aperture_elliptical():
    """
    Test that the tracking results through an elliptical Cheeath `Aperture` element
    match those through an elliptical Oclet `Aperture` element.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_segment = cheetah.Segment(
        [
            cheetah.Aperture(
                x_max=2e-4,
                y_max=2e-4,
                shape="elliptical",
                name="aperture",
                is_active=True,
            ),
            cheetah.Drift(length=0.1),
        ]
    )
    outgoing_beam = cheetah_segment.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    ocelot_cell = [
        ocelot.Aperture(xmax=2e-4, ymax=2e-4, type="ellipt"),
        ocelot.Drift(l=0.1),
    ]
    lattice = ocelot.MagneticLattice([ocelot_cell])
    navigator = ocelot.Navigator(lattice)
    navigator.activate_apertures()
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)

    assert outgoing_beam.num_particles == outgoing_p_array.rparticles.shape[1]


def test_solenoid():
    """
    Test that the tracking results through a Cheeath `Solenoid` element match those
    through an Oclet `Solenoid` element.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_solenoid = cheetah.Solenoid(length=0.5, k=5)
    outgoing_beam = cheetah_solenoid.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    ocelot_solenoid = ocelot.Solenoid(l=0.5, k=5)
    lattice = ocelot.MagneticLattice([ocelot_solenoid])
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)

    assert np.allclose(
        outgoing_beam.particles[:, :6], outgoing_p_array.rparticles.transpose()
    )
