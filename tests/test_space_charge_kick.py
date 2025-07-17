import pytest
import torch
import torch.autograd.forward_ad as fwAD
from scipy import constants
from scipy.constants import physical_constants
from torch import nn

import cheetah
from cheetah.utils import compute_relativistic_factors, is_mps_available_and_functional


# Run the test below for both the ultra-relativistic case
# (250 MeV) and the non-relativistic case (1 MeV).
@pytest.mark.parametrize("energy", [torch.tensor(2.5e8), torch.tensor(1e6)])
def test_cold_uniform_beam_expansion(energy):
    """
    Tests that that a cold uniform beam doubles in size in both dimensions when
    travelling through a drift section with space_charge. (cf ImpactX test:
    https://impactx.readthedocs.io/en/latest/usage/examples/expanding_beam/README.html)
    See Free Expansion of a Cold Uniform Bunch in
    https://accelconf.web.cern.ch/hb2023/papers/thbp44.pdf
    """
    # Simulation parameters
    R0 = torch.tensor(0.001)
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
        num_particles=100_000,
        total_charge=torch.tensor(1e-8),
        energy=energy,
        radius_x=R0,
        radius_y=R0,
        radius_tau=R0 / gamma / beta,  # Duration of the beam in in the lab frame
        sigma_px=torch.tensor(1e-15),
        sigma_py=torch.tensor(1e-15),
        sigma_p=torch.tensor(1e-15),
    )

    # Compute section length that results in a doubling of the beam size
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


def test_vectorized_cold_uniform_beam_expansion():
    """
    Same as `test_cold_uniform_beam_expansion` but testing that all results in a
    vectorised setup are correct.
    """
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
        num_particles=100_000,
        total_charge=torch.tensor(1e-8).repeat(3, 2),
        energy=energy,
        radius_x=R0,
        radius_y=R0,
        radius_tau=R0 / gamma / beta,  # Duration of the beam in in the lab frame
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


def test_vectorized():
    """Tests that the space charge kick can be applied to a vectorized beam."""
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
    beta = torch.sqrt(1 - 1 / gamma**2)

    incoming = cheetah.ParticleBeam.uniform_3d_ellipsoid(
        num_particles=10_000,
        total_charge=torch.tensor([[1e-9, 2e-9], [3e-9, 4e-9], [5e-9, 6e-9]]),
        energy=energy.expand([3, 2]),
        radius_x=R0.expand([3, 2]),
        radius_y=R0.expand([3, 2]),
        radius_tau=R0.expand([3, 2]) / gamma / beta,
        # Duration of the beam in the lab frame
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


def test_incoming_beam_not_modified():
    """
    Tests that the incoming beam is not modified when calling the track method.
    """
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000, sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
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


def test_gradient_value_backward_ad():
    """
    Tests that the gradient of the track method is computed accurately. Using PyTorch's
    default backward mode automatic differentiation.
    """
    # Simulation parameters
    R0 = torch.tensor(0.001)
    energy = torch.tensor(2.5e8)
    species = cheetah.Species("electron")
    gamma, _, beta = compute_relativistic_factors(energy, species.mass_eV)

    incoming_beam = cheetah.ParticleBeam.uniform_3d_ellipsoid(
        num_particles=100_000,
        total_charge=torch.tensor(1e-8),
        energy=energy,
        radius_x=R0,
        radius_y=R0,
        radius_tau=R0 / gamma / beta,  # Duration of the beam in the lab frame
        sigma_px=torch.tensor(1e-15),
        sigma_py=torch.tensor(1e-15),
        sigma_p=torch.tensor(1e-15),
        species=species,
    )

    # Compute section length that results in a doubling of the beam size
    electron_radius = torch.tensor(physical_constants["classical electron radius"][0])
    kappa = 1 + (torch.sqrt(torch.tensor(2)) / 4) * torch.log(
        3 + 2 * torch.sqrt(torch.tensor(2))
    )
    Nb = incoming_beam.total_charge / constants.elementary_charge
    segment_length = beta * gamma * kappa * torch.sqrt(R0**3 / (Nb * electron_radius))

    segment_length = nn.Parameter(segment_length)
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
    outgoing_beam.sigma_x.backward()

    # Check that the gradient is correct by comparing the derivative of the beam radius
    # as a function of the segment length
    dsigma_dlength = segment_length.grad
    # For a sphere, the radius is sqrt(5) bigger than sigma_x
    dradius_dlength = 5**0.5 * dsigma_dlength
    # Theoretical formula obtained by conservation of energy in the beam frame
    expected_dradius_dlength = torch.sqrt((Nb * electron_radius) / R0) / gamma

    assert torch.allclose(dradius_dlength, expected_dradius_dlength, rtol=0.1)


def test_gradient_value_forward_ad():
    """
    Tests that the gradient of the track method is computed accurately. Using PyTorch's
    forward mode automatic differentiation.

    See: https://pytorch.org/tutorials/intermediate/forward_ad_usage.html
    """
    # Simulation parameters
    R0 = torch.tensor(0.001)
    energy = torch.tensor(2.5e8)
    species = cheetah.Species("electron")
    gamma, _, beta = compute_relativistic_factors(energy, species.mass_eV)

    incoming_beam = cheetah.ParticleBeam.uniform_3d_ellipsoid(
        num_particles=100_000,
        total_charge=torch.tensor(1e-8),
        energy=energy,
        radius_x=R0,
        radius_y=R0,
        radius_tau=R0 / gamma / beta,  # Duration of the beam in the lab frame
        sigma_px=torch.tensor(1e-15),
        sigma_py=torch.tensor(1e-15),
        sigma_p=torch.tensor(1e-15),
        species=species,
    )

    # Compute section length that results in a doubling of the beam size
    electron_radius = torch.tensor(physical_constants["classical electron radius"][0])
    kappa = 1 + (torch.sqrt(torch.tensor(2)) / 4) * torch.log(
        3 + 2 * torch.sqrt(torch.tensor(2))
    )
    Nb = incoming_beam.total_charge / constants.elementary_charge
    segment_length = beta * gamma * kappa * torch.sqrt(R0**3 / (Nb * electron_radius))

    tangent = torch.ones_like(segment_length)

    with fwAD.dual_level():
        segment_length = fwAD.make_dual(segment_length, tangent)

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
        beam_size = outgoing_beam.sigma_x

        # Check that the gradient is correct by comparing the derivative of the beam
        # radius as a function of the segment length
        dsigma_dlength = fwAD.unpack_dual(beam_size).tangent
        # For a sphere, the radius is sqrt(5) bigger than sigma_x
        dradius_dlength = 5**0.5 * dsigma_dlength
        # Theoretical formula obtained by conservation of energy in the beam frame
        expected_dradius_dlength = torch.sqrt((Nb * electron_radius) / R0) / gamma

        assert torch.allclose(dradius_dlength, expected_dradius_dlength, rtol=0.1)


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
                is_active=False,
                name="aperture",
            ),
            cheetah.Drift(length=torch.tensor(0.25)),
            cheetah.SpaceChargeKick(effect_length=torch.tensor(0.5)),
            cheetah.Drift(length=torch.tensor(0.25)),
        ]
    )
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
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


@pytest.mark.parametrize(
    "device, dtype",
    [
        (torch.device("cpu"), torch.float32),
        (torch.device("cpu"), torch.float64),
        pytest.param(
            torch.device("cuda"),
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            torch.float32,
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_device_and_dtype(device, dtype):
    """
    Test that `SpaceChargeKick` works correctly on various devices and with various
    dtypes.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.25)),
            cheetah.SpaceChargeKick(effect_length=torch.tensor(0.5)),
            cheetah.Drift(length=torch.tensor(0.25)),
        ]
    ).to(device=device, dtype=dtype)
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        total_charge=torch.tensor(1e-9),
        mu_x=torch.tensor(5e-5),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
    ).to(device=device, dtype=dtype)

    # Run in part to see if errors are raised
    outgoing_beam = segment.track(incoming_beam)

    # Check device and dtype of the output
    for attribute in outgoing_beam.UNVECTORIZED_NUM_ATTR_DIMS.keys():
        assert getattr(outgoing_beam, attribute).device.type == device.type
        assert getattr(outgoing_beam, attribute).dtype == dtype
