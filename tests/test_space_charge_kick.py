import pytest
import torch
from scipy import constants
from scipy.constants import physical_constants

import cheetah

# For reproducibility, we set the seed for the random number generator.
# This is used when the particles positions are initialized randomly,
# Random fluctuations in the initial density can cause the tests to fail.
torch.manual_seed(0)


def test_cold_uniform_beam_expansion():
    """
    Tests that that a cold uniform beam doubles in size in both dimensions when
    travelling through a drift section with space_charge. (cf ImpactX test:
    https://impactx.readthedocs.io/en/latest/usage/examples/cfchannel/README.html#constant-focusing-channel-with-space-charge)
    See Free Expansion of a Cold Uniform Bunch in
    https://accelconf.web.cern.ch/hb2023/papers/thbp44.pdf.
    """

    # Simulation parameters
    num_particles = 10000
    total_charge = torch.tensor([1e-9])
    R0 = torch.tensor([0.001])
    energy = torch.tensor([2.5e8])
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
        num_particles=torch.tensor(num_particles),
        total_charge=total_charge,
        energy=energy,
        radius_x=R0,
        radius_y=R0,
        radius_s=R0 / gamma,  # radius of the beam in s direction, in the lab frame.
        sigma_xp=torch.tensor([1e-15]),
        sigma_yp=torch.tensor([1e-15]),
        sigma_p=torch.tensor([1e-15]),
    )

    # Compute section lenght
    kappa = 1 + (torch.sqrt(torch.tensor(2)) / 4) * torch.log(
        3 + 2 * torch.sqrt(torch.tensor(2))
    )
    Nb = total_charge / elementary_charge
    section_length = beta * gamma * kappa * torch.sqrt(R0**3 / (Nb * electron_radius))

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
    outgoing_beam = segment_space_charge.track(incoming)

    assert torch.isclose(outgoing_beam.sigma_x, 2 * incoming.sigma_x, rtol=2e-2, atol=0.0)
    assert torch.isclose(outgoing_beam.sigma_y, 2 * incoming.sigma_y, rtol=2e-2, atol=0.0)
    assert torch.isclose(outgoing_beam.sigma_s, 2 * incoming.sigma_s, rtol=2e-2, atol=0.0)


def test_incoming_beam_not_modified():
    """
    Tests that the incoming beam is not modified when calling the track method.
    """

    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=torch.tensor([10000]),
        sigma_xp=torch.tensor([2e-7]),
        sigma_yp=torch.tensor([2e-7]),
    )
    # Initial beam properties
    incoming_beam_before = incoming_beam.particles

    section_length = torch.tensor([1.0])
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
