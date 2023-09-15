from copy import deepcopy

import numpy as np
import ocelot
import torch

import cheetah

from .resources import ARESlatticeStage3v1_9 as ares


def test_dipole():
    """
    Test that the tracking results through a Cheeath `Dipole` element match those
    through an Oclet `Bend` element.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_dipole = cheetah.Dipole(length=torch.tensor(0.1), angle=torch.tensor(0.1))
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
    cheetah_dipole = cheetah.Dipole(
        length=torch.tensor(0.1),
        angle=torch.tensor(0.1),
        fringe_integral=torch.tensor(0.1),
        gap=torch.tensor(0.2),
    )
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
                x_max=torch.tensor(2e-4),
                y_max=torch.tensor(2e-4),
                shape="rectangular",
                name="aperture",
                is_active=True,
            ),
            cheetah.Drift(length=torch.tensor(0.1)),
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
                x_max=torch.tensor(2e-4),
                y_max=torch.tensor(2e-4),
                shape="elliptical",
                name="aperture",
                is_active=True,
            ),
            cheetah.Drift(length=torch.tensor(0.1)),
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
    cheetah_solenoid = cheetah.Solenoid(length=torch.tensor(0.5), k=torch.tensor(5.0))
    outgoing_beam = cheetah_solenoid.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    ocelot_solenoid = ocelot.Solenoid(l=0.5, k=5.0)
    lattice = ocelot.MagneticLattice([ocelot_solenoid])
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)

    assert np.allclose(
        outgoing_beam.particles[:, :6], outgoing_p_array.rparticles.transpose()
    )


def test_ares_ea():
    """
    Test that the tracking results through a Experimental Area (EA) lattice of the ARES
    accelerator at DESY match those using Ocelot.
    """
    cell = cheetah.converters.nocelot.subcell_of_ocelot(
        ares.cell, "AREASOLA1", "AREABSCR1"
    )
    ares.areamqzm1.k1 = 5.0
    ares.areamqzm2.k1 = -5.0
    ares.areamcvm1.k1 = 1e-3
    ares.areamqzm3.k1 = 5.0
    ares.areamchm1.k1 = -2e-3

    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_segment = cheetah.Segment.from_ocelot(cell)
    outgoing_beam = cheetah_segment.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001", print_params=False
    )
    lattice = ocelot.MagneticLattice(cell)
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)

    assert np.isclose(outgoing_beam.mu_x, outgoing_p_array.x().mean())
    assert np.isclose(outgoing_beam.mu_xp, outgoing_p_array.px().mean())
    assert np.isclose(outgoing_beam.mu_y, outgoing_p_array.y().mean())
    assert np.isclose(outgoing_beam.mu_yp, outgoing_p_array.py().mean())
    assert np.isclose(outgoing_beam.mu_s, outgoing_p_array.tau().mean(), atol=1e-7)
    assert np.isclose(outgoing_beam.mu_p, outgoing_p_array.p().mean())

    assert np.allclose(outgoing_beam.xs, outgoing_p_array.x())
    assert np.allclose(outgoing_beam.xps, outgoing_p_array.px())
    assert np.allclose(outgoing_beam.ys, outgoing_p_array.y())
    assert np.allclose(outgoing_beam.yps, outgoing_p_array.py())
    assert np.allclose(
        outgoing_beam.ss, outgoing_p_array.tau(), atol=1e-7, rtol=1e-1
    )  # TODO: Why do we need such large tolerances?
    assert np.allclose(outgoing_beam.ps, outgoing_p_array.p())


def test_twiss_particle_beam():
    """
    Test that the twiss parameters computed by Cheetah for a `ParticleBeam` loaded from
    an Astra beam are the same as those computed by Ocelot for the `ParticleArray`
    loaded from that same Astra beam.
    """
    # Cheetah
    particle_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )

    # Ocelot
    p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001", print_params=False
    )
    ocelot_twiss = ocelot.cpbd.beam.get_envelope(p_array)

    # Compare
    assert np.isclose(particle_beam.emittance_x, ocelot_twiss.emit_x)
    assert np.isclose(particle_beam.normalized_emittance_x, ocelot_twiss.emit_xn)
    assert np.isclose(
        particle_beam.beta_x, ocelot_twiss.beta_x, rtol=1e-4
    )  # TODO: Is tolerance okay?
    assert np.isclose(particle_beam.alpha_x, ocelot_twiss.alpha_x, rtol=1e-4)
    assert np.isclose(particle_beam.emittance_y, ocelot_twiss.emit_y)
    assert np.isclose(particle_beam.normalized_emittance_y, ocelot_twiss.emit_yn)
    assert np.isclose(
        particle_beam.beta_y, ocelot_twiss.beta_y, rtol=1e-4
    )  # TODO: Is tolerance okay?
    assert np.isclose(particle_beam.alpha_y, ocelot_twiss.alpha_y, rtol=1e-4)


def test_twiss_parameter_beam():
    """
    Test that the twiss parameters computed by Cheetah for a `ParameterBeam` loaded from
    an Astra beam are the same as those computed by Ocelot for the `ParticleArray`
    loaded from that same Astra beam.
    """
    # Cheetah
    parameter_beam = cheetah.ParameterBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )

    # Ocelot
    p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001", print_params=False
    )
    ocelot_twiss = ocelot.cpbd.beam.get_envelope(p_array)

    # Compare
    assert np.isclose(parameter_beam.emittance_x, ocelot_twiss.emit_x)
    assert np.isclose(parameter_beam.normalized_emittance_x, ocelot_twiss.emit_xn)
    assert np.isclose(
        parameter_beam.beta_x, ocelot_twiss.beta_x, rtol=1e-4
    )  # TODO: Is tolerance okay?
    assert np.isclose(parameter_beam.alpha_x, ocelot_twiss.alpha_x, rtol=1e-4)
    assert np.isclose(parameter_beam.emittance_y, ocelot_twiss.emit_y)
    assert np.isclose(parameter_beam.normalized_emittance_y, ocelot_twiss.emit_yn)
    assert np.isclose(
        parameter_beam.beta_y, ocelot_twiss.beta_y, rtol=1e-4
    )  # TODO: Is tolerance okay?
    assert np.isclose(parameter_beam.alpha_y, ocelot_twiss.alpha_y, rtol=1e-4)


def test_astra_import():
    """
    Test if the beam imported from Astra in Cheetah matches the beam imported from Astra
    in Ocelot.
    """
    beam = cheetah.ParticleBeam.from_astra("benchmark/astra/ACHIP_EA1_2021.1351.001")
    p_array = ocelot.astraBeam2particleArray("benchmark/astra/ACHIP_EA1_2021.1351.001")

    assert np.allclose(beam.particles[:, :6], p_array.rparticles.transpose())
    assert np.isclose(beam.energy, (p_array.E * 1e9))


def test_quadrupole():
    """
    Test if the tracking results through a Cheeath `Quadrupole` element match those
    through an Ocelot `Quadrupole` element.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_quadrupole = cheetah.Quadrupole(
        length=torch.tensor(0.23), k1=torch.tensor(5.0)
    )
    cheetah_segment = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(0.1)),
            cheetah_quadrupole,
            cheetah.Drift(length=torch.tensor(0.1)),
        ]
    )
    outgoing_beam = cheetah_segment.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    ocelot_quadrupole = ocelot.Quadrupole(l=0.23, k1=5.0)
    lattice = ocelot.MagneticLattice(
        [ocelot.Drift(l=0.1), ocelot_quadrupole, ocelot.Drift(l=0.1)]
    )
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)

    # Split in order to allow for different tolerances for each particle dimension
    assert np.allclose(
        outgoing_beam.particles[:, 0], outgoing_p_array.rparticles.transpose()[:, 0]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 1], outgoing_p_array.rparticles.transpose()[:, 1]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 2], outgoing_p_array.rparticles.transpose()[:, 2]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 3], outgoing_p_array.rparticles.transpose()[:, 3]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 4],
        outgoing_p_array.rparticles.transpose()[:, 4],
        atol=1e-6,  # TODO: Is this tolerance already too large?
    )
    assert np.allclose(
        outgoing_beam.particles[:, 5], outgoing_p_array.rparticles.transpose()[:, 5]
    )


def test_tilted_quadrupole():
    """
    Test if the tracking results through a tilted Cheeath `Quadrupole` element match
    those through a tilted Ocelot `Quadrupole` element.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_quadrupole = cheetah.Quadrupole(
        length=torch.tensor(0.23), k1=torch.tensor(5.0), tilt=torch.tensor(0.79)
    )
    cheetah_segment = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(0.1)),
            cheetah_quadrupole,
            cheetah.Drift(length=torch.tensor(0.1)),
        ]
    )
    outgoing_beam = cheetah_segment.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    ocelot_quadrupole = ocelot.Quadrupole(l=0.23, k1=5.0, tilt=0.79)
    lattice = ocelot.MagneticLattice(
        [ocelot.Drift(l=0.1), ocelot_quadrupole, ocelot.Drift(l=0.1)]
    )
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)

    # Split in order to allow for different tolerances for each particle dimension
    assert np.allclose(
        outgoing_beam.particles[:, 0], outgoing_p_array.rparticles.transpose()[:, 0]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 1], outgoing_p_array.rparticles.transpose()[:, 1]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 2], outgoing_p_array.rparticles.transpose()[:, 2]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 3], outgoing_p_array.rparticles.transpose()[:, 3]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 4],
        outgoing_p_array.rparticles.transpose()[:, 4],
        atol=1e-6,  # TODO: Is this tolerance already too large?
    )
    assert np.allclose(
        outgoing_beam.particles[:, 5], outgoing_p_array.rparticles.transpose()[:, 5]
    )


def test_sbend():
    """
    Test if the tracking results through a Cheeath `Dipole` element match those through
    an Ocelot `SBend` element.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_dipole = cheetah.Dipole(length=torch.tensor(0.1), angle=torch.tensor(2e-5))
    cheetah_segment = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(0.1)),
            cheetah_dipole,
            cheetah.Drift(length=torch.tensor(0.1)),
        ]
    )
    outgoing_beam = cheetah_segment.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001", print_params=False
    )
    ocelot_sbend = ocelot.SBend(l=0.1, angle=2e-5)
    lattice = ocelot.MagneticLattice(
        [ocelot.Drift(l=0.1), ocelot_sbend, ocelot.Drift(l=0.1)]
    )
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(
        lattice, deepcopy(incoming_p_array), navigator, print_progress=False
    )

    # Split in order to allow for different tolerances for each particle dimension
    assert np.allclose(
        outgoing_beam.particles[:, 0],
        outgoing_p_array.rparticles.transpose()[:, 0],
        rtol=1e-3,
    )
    assert np.allclose(
        outgoing_beam.particles[:, 1], outgoing_p_array.rparticles.transpose()[:, 1]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 2], outgoing_p_array.rparticles.transpose()[:, 2]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 3], outgoing_p_array.rparticles.transpose()[:, 3]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 4],
        outgoing_p_array.rparticles.transpose()[:, 4],
        atol=1e-6,  # TODO: Is this tolerance already too large?
    )
    assert np.allclose(
        outgoing_beam.particles[:, 5], outgoing_p_array.rparticles.transpose()[:, 5]
    )


def test_rbend():
    """
    Test if the tracking results through a Cheeath `RBend` element match those through
    an Ocelot `RBend` element.
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_dipole = cheetah.RBend(length=torch.tensor(0.1), angle=torch.tensor(2e-5))
    cheetah_segment = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(0.1)),
            cheetah_dipole,
            cheetah.Drift(length=torch.tensor(0.1)),
        ]
    )
    outgoing_beam = cheetah_segment.track(incoming_beam)

    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001", print_params=False
    )
    ocelot_rbend = ocelot.RBend(l=0.1, angle=2e-5)
    lattice = ocelot.MagneticLattice(
        [ocelot.Drift(l=0.1), ocelot_rbend, ocelot.Drift(l=0.1)]
    )
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(
        lattice, deepcopy(incoming_p_array), navigator, print_progress=False
    )

    # Split in order to allow for different tolerances for each particle dimension
    assert np.allclose(
        outgoing_beam.particles[:, 0],
        outgoing_p_array.rparticles.transpose()[:, 0],
        rtol=1e-3,
    )
    assert np.allclose(
        outgoing_beam.particles[:, 1], outgoing_p_array.rparticles.transpose()[:, 1]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 2], outgoing_p_array.rparticles.transpose()[:, 2]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 3], outgoing_p_array.rparticles.transpose()[:, 3]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 4],
        outgoing_p_array.rparticles.transpose()[:, 4],
        atol=1e-6,  # TODO: Is this tolerance already too large?
    )
    assert np.allclose(
        outgoing_beam.particles[:, 5], outgoing_p_array.rparticles.transpose()[:, 5]
    )


def test_convert_rbend():
    """
    Test if the tracking results through a ocelot-converted Cheetah segment match
    those through an Ocelot section with an `RBend` element.
    """
    # Ocelot
    incoming_p_array = ocelot.astraBeam2particleArray(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001", print_params=False
    )
    ocelot_rbend = ocelot.RBend(l=0.1, angle=2e-5, gap=0.05, fint=0.1, eid="rbend")
    lattice = ocelot.MagneticLattice(
        [
            ocelot.Drift(l=0.1, eid="drfit_1"),
            ocelot_rbend,
            ocelot.Drift(l=0.1, eid="drift_2"),
        ]
    )
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(
        lattice, deepcopy(incoming_p_array), navigator, print_progress=False
    )

    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
    )
    cheetah_segment = cheetah.Segment.from_ocelot(lattice.sequence)
    outgoing_beam = cheetah_segment.track(incoming_beam)

    # Split in order to allow for different tolerances for each particle dimension
    assert np.allclose(
        outgoing_beam.particles[:, 0],
        outgoing_p_array.rparticles.transpose()[:, 0],
        rtol=1e-3,
    )
    assert np.allclose(
        outgoing_beam.particles[:, 1], outgoing_p_array.rparticles.transpose()[:, 1]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 2], outgoing_p_array.rparticles.transpose()[:, 2]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 3], outgoing_p_array.rparticles.transpose()[:, 3]
    )
    assert np.allclose(
        outgoing_beam.particles[:, 4],
        outgoing_p_array.rparticles.transpose()[:, 4],
        atol=1e-6,  # TODO: Is this tolerance already too large?
    )
    assert np.allclose(
        outgoing_beam.particles[:, 5], outgoing_p_array.rparticles.transpose()[:, 5]
    )


def test_cavity():
    """
    Compare tracking through a cavity that is on.

    The particular settings tested here also get to the same result in Bmad/Tao. The
    below lattice was used for the Bmad test:
    ```
    ! Lattice file: simple.bmad
    beginning[beta_a] = 5.91253677 ! m a-mode beta function
    beginning[beta_b] = 5.91253677 ! m b-mode beta function
    beginning[alpha_a] = 3.55631308 ! a-mode alpha function
    beginning[alpha_b] = 3.55631308 ! b-mode alpha function
    beginning[e_tot] = 6e6 ! eV    Or can set beginning[p0c]

    parameter[geometry] = open  ! Or closed
    parameter[particle] = electron  ! Reference particle.

    c: lcavity, rf_frequency = 1.3e9, l = 1.0377, voltage = 0.01815975e9, phi0 = 0.0

    lat: line = (c) ! List of lattice elements
    use, lat ! Line used to construct the lattice
    ```
    The twiss parameters at the end of the lattice according to Bmad should be:
     - beta_x  = 0.23847352510683092
     - beta_y  = 0.23847352512430994
     - alpha_x = -1.0160687592932345
     - alpha_y = -1.0160687593664295
    """
    # Cheetah
    incoming_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(5.91253677),
        alpha_x=torch.tensor(3.55631308),
        beta_y=torch.tensor(5.91253677),
        alpha_y=torch.tensor(3.55631308),
        emittance_x=torch.tensor(3.494768647122823e-09),
        emittance_y=torch.tensor(3.497810737006068e-09),
        energy=torch.tensor(6e6),
    )
    cheetah_cavity = cheetah.Cavity(
        length=torch.tensor(1.0377),
        voltage=torch.tensor(0.01815975e9),
        frequency=torch.tensor(1.3e9),
        phase=torch.tensor(0.0),
    )
    outgoing_beam = cheetah_cavity.track(incoming_beam)

    # Ocelot
    tws = ocelot.Twiss()
    tws.beta_x = 5.91253677
    tws.alpha_x = 3.55631308
    tws.beta_y = 5.91253677
    tws.alpha_y = 3.55631308
    tws.emit_x = 3.494768647122823e-09
    tws.emit_y = 3.497810737006068e-09
    tws.gamma_x = (1 + tws.alpha_x**2) / tws.beta_x
    tws.gamma_y = (1 + tws.alpha_y**2) / tws.beta_y
    tws.E = 6e-3

    p_array = ocelot.generate_parray(tws=tws)

    cell = [ocelot.Cavity(l=1.0377, v=0.01815975, freq=1.3e9, phi=0.0)]
    lattice = ocelot.MagneticLattice(cell)
    navigator = ocelot.Navigator(lattice=lattice)

    _, outgoing_parray = ocelot.track(lattice, deepcopy(p_array), navigator)
    derived_twiss = ocelot.cpbd.beam.get_envelope(outgoing_parray)

    # Compare
    assert np.isclose(outgoing_beam.beta_x, derived_twiss.beta_x, rtol=1e-2)
    assert np.isclose(outgoing_beam.alpha_x, derived_twiss.alpha_x, rtol=1e-2)
    assert np.isclose(outgoing_beam.beta_y, derived_twiss.beta_y, rtol=1e-2)
    assert np.isclose(outgoing_beam.alpha_y, derived_twiss.alpha_y, rtol=1e-2)
