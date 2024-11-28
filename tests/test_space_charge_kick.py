import torch
from scipy import constants
from scipy.constants import physical_constants
from torch import nn

import cheetah


def test_cold_uniform_beam_expansion():
    """
    Tests that that a cold uniform beam doubles in size in both dimensions when
    travelling through a drift section with space_charge. (cf ImpactX test:
    https://impactx.readthedocs.io/en/latest/usage/examples/cfchannel/README.html#constant-focusing-channel-with-space-charge)
    See Free Expansion of a Cold Uniform Bunch in
    https://accelconf.web.cern.ch/hb2023/papers/thbp44.pdf.
    """

    # Random fluctuations in the initial density can cause the tests to fail
    torch.manual_seed(42)

    # Simulation parameters
    R0 = torch.tensor(0.001)
    energy = torch.tensor(2.5e8)
    rest_energy = torch.tensor(
        constants.electron_mass
        * constants.speed_of_light**2
        / constants.elementary_charge
    )
    elementary_charge = torch.tensor(constants.elementary_charge)
    electron_radius = torch.tensor(physical_constants["classical electron radius"][0])
    gamma = energy / rest_energy
    beta = torch.sqrt(1 - 1 / gamma**2)

    incoming = cheetah.ParticleBeam.uniform_3d_ellipsoid(
        num_particles=torch.tensor(10_000),
        total_charge=torch.tensor(1e-9),
        energy=energy,
        radius_x=R0,
        radius_y=R0,
        radius_tau=R0 / gamma,  # Radius of the beam in s direction in the lab frame
        sigma_px=torch.tensor(1e-15),
        sigma_py=torch.tensor(1e-15),
        sigma_p=torch.tensor(1e-15),
    )

    # Compute section length
    kappa = 1 + (torch.sqrt(torch.tensor(2)) / 4) * torch.log(
        3 + 2 * torch.sqrt(torch.tensor(2))
    )
    Nb = incoming.total_charge / elementary_charge
    section_length = beta * gamma * kappa * torch.sqrt(R0**3 / (Nb * electron_radius))

    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(section_length / 6),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 6),
        ]
    )
    outgoing = segment.track(incoming)

    assert torch.isclose(outgoing.sigma_x, 2 * incoming.sigma_x, rtol=2e-2)
    assert torch.isclose(outgoing.sigma_y, 2 * incoming.sigma_y, rtol=2e-2)
    assert torch.isclose(outgoing.sigma_tau, 2 * incoming.sigma_tau, rtol=2e-2)


def test_vectorized():
    """
    Tests that the space charge kick can be applied to a vectorized beam.
    """

    # Simulation parameters
    section_length = torch.tensor(0.42)
    R0 = torch.tensor(0.001)
    energy = torch.tensor(2.5e8)
    rest_energy = torch.tensor(
        constants.electron_mass
        * constants.speed_of_light**2
        / constants.elementary_charge
    )
    gamma = energy / rest_energy

    incoming = cheetah.ParticleBeam.uniform_3d_ellipsoid(
        num_particles=torch.tensor(10_000),
        total_charge=torch.tensor([[1e-9, 2e-9], [3e-9, 4e-9], [5e-9, 6e-9]]),
        energy=energy.expand([3, 2]),
        radius_x=R0.expand([3, 2]),
        radius_y=R0.expand([3, 2]),
        radius_tau=R0.expand([3, 2]) / gamma,
        # Radius of the beam in s direction in the lab frame
        sigma_px=torch.tensor(1e-15).expand([3, 2]),
        sigma_py=torch.tensor(1e-15).expand([3, 2]),
        sigma_p=torch.tensor(1e-15).expand([3, 2]),
    )

    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(section_length / 6),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 6),
        ]
    )

    outgoing = segment.track(incoming)

    assert outgoing.particles.shape == (3, 2, 10_000, 7)


def test_vectorized_cold_uniform_beam_expansion():
    """
    Same as `test_cold_uniform_beam_expansion` but testing that all results in a
    vectorised setup are correct.
    """

    # Random fluctuations in the initial density can cause the tests to fail
    torch.manual_seed(42)

    # Simulation parameters
    R0 = torch.tensor(0.001)
    energy = torch.tensor(2.5e8)
    rest_energy = torch.tensor(
        constants.electron_mass
        * constants.speed_of_light**2
        / constants.elementary_charge
    )
    elementary_charge = torch.tensor(constants.elementary_charge)
    electron_radius = torch.tensor(physical_constants["classical electron radius"][0])
    gamma = energy / rest_energy
    beta = torch.sqrt(1 - 1 / gamma**2)

    incoming = cheetah.ParticleBeam.uniform_3d_ellipsoid(
        num_particles=torch.tensor(10_000),
        total_charge=torch.tensor(1e-9),
        energy=energy,
        radius_x=R0,
        radius_y=R0,
        radius_tau=R0 / gamma,  # Radius of the beam in s direction in the lab frame
        sigma_px=torch.tensor(1e-15),
        sigma_py=torch.tensor(1e-15),
        sigma_p=torch.tensor(1e-15),
    )

    # Compute section length
    kappa = 1 + (torch.sqrt(torch.tensor(2)) / 4) * torch.log(
        3 + 2 * torch.sqrt(torch.tensor(2))
    )
    Nb = incoming.total_charge / elementary_charge
    section_length = beta * gamma * kappa * torch.sqrt(R0**3 / (Nb * electron_radius))

    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(section_length / 6),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 6),
        ]
    )
    outgoing = segment.track(incoming)

    assert torch.allclose(outgoing.sigma_x, 2 * incoming.sigma_x, rtol=2e-2)
    assert torch.allclose(outgoing.sigma_y, 2 * incoming.sigma_y, rtol=2e-2)
    assert torch.allclose(outgoing.sigma_tau, 2 * incoming.sigma_tau, rtol=2e-2)


def test_incoming_beam_not_modified():
    """
    Tests that the incoming beam is not modified when calling the track method.
    """

    # Random fluctuations in the initial density can cause the tests to fail
    torch.manual_seed(42)

    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=torch.tensor(10_000),
        sigma_px=torch.tensor(2e-7),
        sigma_py=torch.tensor(2e-7),
    )
    # Initial beam properties
    incoming_beam_before = incoming_beam.particles

    section_length = torch.tensor(1.0)
    segment_space_charge = cheetah.Segment(
        elements=[
            cheetah.Drift(section_length / 6),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 6),
        ]
    )
    # Calling the track method
    segment_space_charge.track(incoming_beam)

    # Final beam properties
    incoming_beam_after = incoming_beam.particles

    assert torch.allclose(incoming_beam_before, incoming_beam_after)


def test_gradient():
    """
    Tests that the gradient of the track method is computed withouth throwing an error.
    """
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=torch.tensor(10_000),
        sigma_px=torch.tensor(2e-7),
        sigma_py=torch.tensor(2e-7),
    )

    segment_length = nn.Parameter(torch.tensor(1.0))
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(segment_length / 6),
            cheetah.SpaceChargeKick(segment_length / 3),
            cheetah.Drift(segment_length / 3),
            cheetah.SpaceChargeKick(segment_length / 3),
            cheetah.Drift(segment_length / 3),
            cheetah.SpaceChargeKick(segment_length / 3),
            cheetah.Drift(segment_length / 6),
        ]
    )

    # Track the beam
    outgoing_beam = segment.track(incoming_beam)

    # Compute the gradient ... would throw an error if in-place operations are used
    outgoing_beam.sigma_x.mean().backward()


def test_does_not_break_segment_length():
    """
    Test that the computation of a `Segment`'s length does not break when
    `SpaceChargeKick` is used.
    """
    section_length = torch.tensor(1.0)
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(section_length / 6),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick(section_length / 3),
            cheetah.Drift(section_length / 6),
        ]
    )

    assert segment.length.shape == torch.Size([])
    assert torch.allclose(segment.length, torch.tensor(1.0))


def test_space_charge_with_ares_astra_beam():
    """
    Tests running space charge through a 1m drift with an Astra beam from the ARES
    linac. This test is added because running this code would throw an error:
    `IndexError: index -38 is out of bounds for dimension 3 with size 32`.
    """
    segment = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.SpaceChargeKick(effect_length=torch.tensor(1.0)),
        ]
    )
    beam = cheetah.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    _ = segment.track(beam)


def test_space_charge_with_aperture_cutoff():
    """
    Tests that the space charge kick is correctly applied only to surviving particles,
    by comparing the results with and without an aperture that results in beam losses.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.2)),
            cheetah.Aperture(
                x_max=torch.tensor(1e-4),
                y_max=torch.tensor(1e-4),
                shape="rectangular",
                is_active="False",
                name="aperture",
            ),
            cheetah.Drift(length=torch.tensor(0.25)),
            cheetah.SpaceChargeKick(effect_length=torch.tensor(0.5)),
            cheetah.Drift(length=torch.tensor(0.25)),
        ]
    )
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=torch.tensor(10_000),
        total_charge=torch.tensor(1e-9),
        mu_x=torch.tensor(5e-5),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
    )

    # Track with inactive aperture
    outgoing_beam_without_aperture = segment.track(incoming_beam)

    # Activate the aperture and track the beam
    segment.aperture.is_active = True
    outgoing_beam_with_aperture = segment.track(incoming_beam)

    # Check that with particle loss the space charge kick is different
    assert not torch.allclose(
        outgoing_beam_with_aperture.particles, outgoing_beam_without_aperture.particles
    )
    # Check that the number of surviving particles is less than the initial number
    assert outgoing_beam_with_aperture.survival_probabilities.sum(dim=-1).max() < 10_000
