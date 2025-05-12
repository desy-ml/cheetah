from copy import deepcopy

import ocelot
import torch

import cheetah


def test_compare_octupole_to_ocelot():
    """Compare the results of tracking through a octupole in Cheetah and Ocelot."""
    length = 0.34
    k3 = 0.5
    tilt = 0.1

    # Track through a octupole in Cheetah
    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    cheetah_octupole = cheetah.Octupole(
        length=torch.tensor(length), k3=torch.tensor(k3), tilt=torch.tensor(tilt)
    )
    outgoing_cheetah = cheetah_octupole.track(incoming)

    # Convert to Ocelot octupole
    incoming_p_array = ocelot.astraBeam2particleArray(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    lattice = ocelot.MagneticLattice(
        [ocelot.Octupole(l=length, k3=k3, tilt=tilt)],
        method={"global": ocelot.SecondTM},
    )
    navigator = ocelot.Navigator(lattice)
    _, outgoing_p_array = ocelot.track(lattice, deepcopy(incoming_p_array), navigator)
    outgoing_ocelot = cheetah.ParticleBeam.from_ocelot(outgoing_p_array)

    # Compare the results
    assert torch.allclose(
        outgoing_cheetah.particles, outgoing_ocelot.particles, atol=1e-5, rtol=1e-6
    )


def test_octupole_as_drift():
    """Test that a octupole with k3=0 is equivalent to a drift."""
    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    octupole = cheetah.Octupole(length=torch.tensor(0.34), k3=torch.tensor(0.0))
    drift = cheetah.Drift(length=torch.tensor(0.34))

    # Track through the octupole and drift
    octupole_outgoing = octupole.track(incoming)
    drift_outgoing = drift.track(incoming)

    # Check that the results are the same
    assert torch.allclose(
        octupole_outgoing.particles, drift_outgoing.particles, atol=1e-5, rtol=1e-6
    )


def test_octupole_parameter_beam_particle_beam_agreement():
    """
    Test that the results of tracking an `ParameterBeam` and a `ParticleBeam` through a
    octupole agree.
    """
    # Create a octupole
    length = 0.34
    k3 = 0.5
    tilt = 0.1
    octupole = cheetah.Octupole(
        length=torch.tensor(length), k3=torch.tensor(k3), tilt=torch.tensor(tilt)
    )

    # Create an incoming ParticleBeam
    incoming_particle_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    # Create an incoming ParameterBeam
    incoming_parameter_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    # Track through the octupole
    outgoing_particle_beam = octupole.track(incoming_particle_beam)
    outgoing_parameter_beam = octupole.track(incoming_parameter_beam)

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
