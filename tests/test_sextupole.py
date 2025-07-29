from copy import deepcopy

import ocelot
import torch

import cheetah


def test_compare_sextupole_to_ocelot_particle():
    """
    Compare the results of tracking through a sextupole in Cheetah and Ocelot. For a
    `ParticleBeam` with second order effects in Ocelot.
    """
    length = 0.11
    k2 = 87.0
    tilt = torch.pi / 2

    # Track through a sextupole in Cheetah
    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    cheetah_sextupole = cheetah.Sextupole(
        length=torch.tensor(length),
        k2=torch.tensor(k2),
        tilt=torch.tensor(tilt),
        tracking_method="second_order",
    )
    outgoing_cheetah = cheetah_sextupole.track(incoming)

    # Convert to Ocelot sextupole
    incoming_p_array = ocelot.astraBeam2particleArray(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    lattice = ocelot.MagneticLattice(
        [ocelot.Sextupole(l=length, k2=k2, tilt=tilt)],
        method={"global": ocelot.SecondTM},
    )
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)
    outgoing_ocelot = cheetah.ParticleBeam.from_ocelot(outgoing_p_array)

    # Compare the results
    assert torch.allclose(
        outgoing_cheetah.particles, outgoing_ocelot.particles, atol=1e-5, rtol=1e-6
    )


def test_compare_sextupole_to_ocelot_particle_vectorized():
    """
    Compare the results of tracking through a sextupole in Cheetah and Ocelot. For a
    `ParticleBeam` with second order effects in Ocelot.

    Vectorised version of the test.
    """
    length = 0.11
    k2 = 87.0
    tilt = torch.pi / 2

    # Track through a sextupole in Cheetah
    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    cheetah_sextupole = cheetah.Sextupole(
        length=torch.tensor(length),
        k2=torch.tensor(k2).repeat([2]),
        tilt=torch.tensor(tilt).repeat([3, 1]),
        tracking_method="second_order",
    )
    outgoing_cheetah = cheetah_sextupole.track(incoming)

    # Convert to Ocelot sextupole
    incoming_p_array = ocelot.astraBeam2particleArray(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    lattice = ocelot.MagneticLattice(
        [ocelot.Sextupole(l=length, k2=k2, tilt=tilt)],
        method={"global": ocelot.SecondTM},
    )
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)
    outgoing_ocelot = cheetah.ParticleBeam.from_ocelot(outgoing_p_array)

    # Compare the results
    assert torch.allclose(
        outgoing_cheetah.particles, outgoing_ocelot.particles, atol=1e-5, rtol=1e-6
    )


def test_compare_sextupole_to_ocelot_parameter_linear():
    """
    Compare the results of tracking through a sextupole in Cheetah and Ocelot for a
    `ParameterBeam` with only first order effects in Ocelot.
    """
    length = 0.11
    k2 = 87.0
    tilt = torch.pi / 2

    # Track through a sextupole in Cheetah
    incoming = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    cheetah_sextupole = cheetah.Sextupole(
        length=torch.tensor(length), k2=torch.tensor(k2), tilt=torch.tensor(tilt)
    )
    outgoing_cheetah = cheetah_sextupole.track(incoming)

    # Convert to Ocelot sextupole
    incoming_p_array = ocelot.astraBeam2particleArray(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    lattice = ocelot.MagneticLattice(
        [ocelot.Sextupole(l=length, k2=k2, tilt=tilt)],
        method={"global": ocelot.TransferMap},
    )
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)
    outgoing_ocelot = cheetah.ParameterBeam.from_ocelot(outgoing_p_array)

    # Compare the results
    assert torch.allclose(outgoing_cheetah.mu, outgoing_ocelot.mu, atol=1e-5, rtol=1e-6)
    assert torch.allclose(
        outgoing_cheetah.cov, outgoing_ocelot.cov, atol=1e-5, rtol=1e-6
    )


def test_sextupole_as_drift():
    """Test that a sextupole with k2=0 is equivalent to a drift."""
    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    sextupole = cheetah.Sextupole(length=torch.tensor(0.11), k2=torch.tensor(0.0))
    drift = cheetah.Drift(length=torch.tensor(0.11))

    # Track through the sextupole and drift
    sextupole_outgoing = sextupole.track(incoming)
    drift_outgoing = drift.track(incoming)

    # Check that the results are the same
    assert torch.allclose(
        sextupole_outgoing.particles, drift_outgoing.particles, atol=1e-5, rtol=1e-6
    )


def test_sextupole_with_misalignments():
    """Test that a sextupole with misalignments behaves as expected."""

    centered_sextupole = cheetah.Sextupole(
        length=torch.tensor(1.0), k2=torch.tensor(0.5)
    )
    misaligned_sextupole = cheetah.Sextupole(
        length=torch.tensor(1.0),
        k2=torch.tensor(0.5),
        misalignment=torch.tensor([1e-3, 0.0]),
    )

    centered_incoming_beam = cheetah.ParticleBeam.from_parameters(
        mu_x=torch.tensor(0.0),
        sigma_px=torch.tensor(2e-7),
        sigma_py=torch.tensor(2e-7),
        sigma_p=torch.tensor(1e-2),
    )
    misaligned_incoming_beam = centered_incoming_beam.clone()
    misaligned_incoming_beam.particles[:, 0] -= 1e-3

    outgoing_beam_misaligned_sextupole = misaligned_sextupole.track(
        centered_incoming_beam
    )
    outgoing_beam_misaligned_incoming = centered_sextupole.track(
        misaligned_incoming_beam
    )

    outgoing_beam_misaligned_incoming.particles[:, 0] += 1e-3

    assert torch.allclose(
        outgoing_beam_misaligned_sextupole.particles,
        outgoing_beam_misaligned_incoming.particles,
    )
