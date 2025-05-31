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
        length=torch.tensor(length), k2=torch.tensor(k2), tilt=torch.tensor(tilt)
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


def test_compare_sextupole_to_ocelot_parameter():
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
