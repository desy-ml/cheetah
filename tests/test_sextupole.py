from copy import deepcopy

import ocelot
import torch

import cheetah


def test_compare_sextupole_to_ocelot():
    """Compare the results of tracking through a sextupole in Cheetah and Ocelot."""
    length = 0.34
    k2 = 0.5
    tilt = 0.1

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


def test_sextupole_as_drift():
    """Test that a sextupole with k2=0 is equivalent to a drift."""
    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    sextupole = cheetah.Sextupole(length=torch.tensor(0.34), k2=torch.tensor(0.0))
    drift = cheetah.Drift(length=torch.tensor(0.34))

    # Track through the sextupole and drift
    sextupole_outgoing = sextupole.track(incoming)
    drift_outgoing = drift.track(incoming)

    # Check that the results are the same
    assert torch.allclose(
        sextupole_outgoing.particles, drift_outgoing.particles, atol=1e-5, rtol=1e-6
    )


def test_sextupole_parameter_beam_particle_beam_agreement():
    """
    Test that the results of tracking an `ParameterBeam` and a `ParticleBeam` through a
    sextupole agree.
    """
    # Create a sextupole
    length = 0.34
    k2 = 0.5
    tilt = 0.1
    misalignment = (1e-4, 2e-4)
    sextupole = cheetah.Sextupole(
        length=torch.tensor(length),
        k2=torch.tensor(k2),
        tilt=torch.tensor(tilt),
        misalignment=torch.tensor(misalignment),
    )

    # Create an incoming ParticleBeam
    incoming_particle_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    # Create an incoming ParameterBeam
    incoming_parameter_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    # Track through the sextupole
    outgoing_particle_beam = sextupole.track(incoming_particle_beam)
    outgoing_parameter_beam = sextupole.track(incoming_parameter_beam)

    outgoing_particle_beam_as_parameter_beam = (
        outgoing_particle_beam.as_parameter_beam()
    )

    # Check that the results are the same
    assert torch.allclose(
        outgoing_particle_beam_as_parameter_beam.mu,
        outgoing_parameter_beam.mu,
        atol=1e-5,
        rtol=1e-6,
    )
    assert torch.allclose(
        outgoing_particle_beam_as_parameter_beam.cov,
        outgoing_parameter_beam.cov,
        atol=1e-5,
        rtol=1e-6,
    )
