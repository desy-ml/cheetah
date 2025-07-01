import numpy as np
import pytest
import scipy.stats
import torch

import cheetah
from cheetah.utils import is_mps_available_and_functional


def test_create_from_parameters():
    """
    Test that a `ParticleBeam` created from parameters actually has those parameters.
    """
    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=1_000_000,
        mu_x=torch.tensor(1e-5),
        mu_px=torch.tensor(1e-7),
        mu_y=torch.tensor(2e-5),
        mu_py=torch.tensor(2e-7),
        sigma_x=torch.tensor(1.75e-7),
        sigma_px=torch.tensor(2e-7),
        sigma_y=torch.tensor(1.75e-7),
        sigma_py=torch.tensor(2e-7),
        sigma_tau=torch.tensor(0.000001),
        sigma_p=torch.tensor(0.000001),
        cov_xpx=torch.tensor(0.0),
        cov_ypy=torch.tensor(0.0),
        cov_taup=torch.tensor(0.0),
        energy=torch.tensor(1e7),
        total_charge=torch.tensor(1e-9),
    )

    assert beam.num_particles == 1_000_000
    assert np.isclose(beam.mu_x.cpu().numpy(), 1e-5)
    assert np.isclose(beam.mu_px.cpu().numpy(), 1e-7)
    assert np.isclose(beam.mu_y.cpu().numpy(), 2e-5)
    assert np.isclose(beam.mu_py.cpu().numpy(), 2e-7)
    assert np.isclose(beam.sigma_x.cpu().numpy(), 1.75e-7)
    assert np.isclose(beam.sigma_px.cpu().numpy(), 2e-7)
    assert np.isclose(beam.sigma_y.cpu().numpy(), 1.75e-7)
    assert np.isclose(beam.sigma_py.cpu().numpy(), 2e-7)
    assert np.isclose(beam.sigma_tau.cpu().numpy(), 0.000001)
    assert np.isclose(beam.sigma_p.cpu().numpy(), 0.000001)
    assert np.isclose(beam.energy.cpu().numpy(), 1e7)
    assert np.isclose(beam.total_charge.cpu().numpy(), 1e-9)


def test_transform_to():
    """
    Test that a `ParticleBeam` transformed to new parameters actually has those new
    parameters.
    """
    original_beam = cheetah.ParticleBeam.from_parameters()
    transformed_beam = original_beam.transformed_to(
        mu_x=torch.tensor(1e-5),
        mu_px=torch.tensor(1e-7),
        mu_y=torch.tensor(2e-5),
        mu_py=torch.tensor(2e-7),
        sigma_x=torch.tensor(1.75e-7),
        sigma_px=torch.tensor(2e-7),
        sigma_y=torch.tensor(1.75e-7),
        sigma_py=torch.tensor(2e-7),
        sigma_tau=torch.tensor(0.000001),
        sigma_p=torch.tensor(0.000001),
        energy=torch.tensor(1e7),
        total_charge=torch.tensor(1e-9),
    )

    assert isinstance(transformed_beam, cheetah.ParticleBeam)
    assert original_beam.num_particles == transformed_beam.num_particles

    assert np.isclose(transformed_beam.mu_x.cpu().numpy(), 1e-5)
    assert np.isclose(transformed_beam.mu_px.cpu().numpy(), 1e-7)
    assert np.isclose(transformed_beam.mu_y.cpu().numpy(), 2e-5)
    assert np.isclose(transformed_beam.mu_py.cpu().numpy(), 2e-7)
    assert np.isclose(transformed_beam.sigma_x.cpu().numpy(), 1.75e-7)
    assert np.isclose(transformed_beam.sigma_px.cpu().numpy(), 2e-7)
    assert np.isclose(transformed_beam.sigma_y.cpu().numpy(), 1.75e-7)
    assert np.isclose(transformed_beam.sigma_py.cpu().numpy(), 2e-7)
    assert np.isclose(transformed_beam.sigma_tau.cpu().numpy(), 0.000001)
    assert np.isclose(transformed_beam.sigma_p.cpu().numpy(), 0.000001)
    assert np.isclose(transformed_beam.energy.cpu().numpy(), 1e7)
    assert np.isclose(transformed_beam.total_charge.cpu().numpy(), 1e-9)


def test_from_twiss_to_twiss():
    """
    Test that a `ParameterBeam` created from twiss parameters actually has those
    parameters.
    """
    beam = cheetah.ParticleBeam.from_twiss(
        num_particles=10_000_000,
        beta_x=torch.tensor(5.91253676811640894),
        alpha_x=torch.tensor(3.55631307633660354),
        emittance_x=torch.tensor(3.494768647122823e-09),
        beta_y=torch.tensor(5.91253676811640982),
        alpha_y=torch.tensor(1.0),  # TODO: set realistic value
        emittance_y=torch.tensor(3.497810737006068e-09),
        energy=torch.tensor(6e6),
    )
    # rather loose rtol is needed here due to the random sampling of the beam
    assert np.isclose(beam.beta_x.cpu().numpy(), 5.91253676811640894, rtol=1e-2)
    assert np.isclose(beam.alpha_x.cpu().numpy(), 3.55631307633660354, rtol=1e-2)
    assert np.isclose(beam.emittance_x.cpu().numpy(), 3.494768647122823e-09, rtol=1e-2)
    assert np.isclose(beam.beta_y.cpu().numpy(), 5.91253676811640982, rtol=1e-2)
    assert np.isclose(beam.alpha_y.cpu().numpy(), 1.0, rtol=1e-2)
    assert np.isclose(beam.emittance_y.cpu().numpy(), 3.497810737006068e-09, rtol=1e-2)
    assert np.isclose(beam.energy.cpu().numpy(), 6e6)


def test_generate_uniform_ellipsoid_dtype():
    """
    Test that a `ParticleBeam` generated from a uniform 3D ellipsoid has the manually
    specified dtype.
    """
    beam_attributes = cheetah.ParticleBeam.UNVECTORIZED_NUM_ATTR_DIMS.keys()

    # Check that the dtype is float32 by default
    default_beam = cheetah.ParticleBeam.uniform_3d_ellipsoid()
    for attribute in beam_attributes:
        assert getattr(default_beam, attribute).dtype == torch.float32

    # Verify that all attributes have been changed to float64
    double_beam = cheetah.ParticleBeam.uniform_3d_ellipsoid(dtype=torch.float64)
    for attribute in beam_attributes:
        assert getattr(double_beam, attribute).dtype == torch.float64


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_generate_uniform_ellipsoid_device(device):
    """
    Test that a `ParticleBeam` generated from a uniform 3D ellipsoid is created on the
    correct device if manually specified.
    """
    beam_attributes = cheetah.ParticleBeam.UNVECTORIZED_NUM_ATTR_DIMS.keys()

    default_beam = cheetah.ParticleBeam.uniform_3d_ellipsoid(device=device)
    for attribute in beam_attributes:
        assert getattr(default_beam, attribute).device.type == device.type


def test_generate_uniform_ellipsoid_vectorized():
    """
    Test that a `ParticleBeam` generated from a uniform 3D ellipsoid has the correct
    parameters, i.e. the all particles are within the ellipsoid, and that the other beam
    parameters are as they would be for a Gaussian beam.
    """
    radius_x = torch.tensor([1e-3, 2e-3])
    radius_y = torch.tensor([1e-4, 2e-4])
    radius_tau = torch.tensor([1e-5, 2e-5])

    num_particles = torch.tensor(1_000_000)
    sigma_px = torch.tensor([2e-7, 1e-7])
    sigma_py = torch.tensor([3e-7, 2e-7])
    sigma_p = torch.tensor([0.000001, 0.000002])
    energy = torch.tensor([1e7, 2e7])
    total_charge = torch.tensor([1e-9, 3e-9])

    num_particles = 1_000_000
    beam = cheetah.ParticleBeam.uniform_3d_ellipsoid(
        num_particles=num_particles,
        radius_x=radius_x,
        radius_y=radius_y,
        radius_tau=radius_tau,
        sigma_px=sigma_px,
        sigma_py=sigma_py,
        sigma_p=sigma_p,
        energy=energy,
        total_charge=total_charge,
    )

    assert beam.num_particles == num_particles
    assert torch.all(beam.x.abs().transpose(0, 1) <= radius_x)
    assert torch.all(beam.y.abs().transpose(0, 1) <= radius_y)
    assert torch.all(beam.tau.abs().transpose(0, 1) <= radius_tau)
    assert torch.allclose(beam.sigma_px, sigma_px)
    assert torch.allclose(beam.sigma_py, sigma_py)
    assert torch.allclose(beam.sigma_p, sigma_p)
    assert torch.allclose(beam.energy, energy)
    assert torch.allclose(beam.total_charge, total_charge)


def test_only_sigma_vectorized():
    """
    Test that particle beam works correctly when only a vectorised sigma is given and
    all else is scalar.
    """
    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        mu_x=torch.tensor(1e-5),
        sigma_x=torch.tensor([1.75e-7, 2.75e-7]),
    )
    assert beam.particles.shape == (2, 10_000, 7)


def test_indexing_with_vectorized_beamline():
    """
    Test that indexing into a vectorised outgoing beam works when the vectorisation
    originates in the beamline.
    """
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor(0.2).unsqueeze(0), k1=torch.rand((5, 2))
    )
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=1_000, sigma_x=torch.tensor(1e-5)
    )

    outgoing = quadrupole.track(incoming)
    sub_beam = outgoing[:3]

    assert sub_beam.particles.shape == torch.Size([3, 2, 1_000, 7])
    assert sub_beam.energy.shape == torch.Size([3, 2])
    assert sub_beam.particle_charges.shape == torch.Size([3, 2, 1_000])
    assert sub_beam.survival_probabilities.shape == torch.Size([3, 2, 1_000])

    assert torch.all(sub_beam.particles == outgoing.particles[:3])
    assert torch.all(sub_beam.energy == outgoing.energy)
    assert torch.all(sub_beam.particle_charges == outgoing.particle_charges)
    assert torch.all(sub_beam.survival_probabilities == outgoing.survival_probabilities)


def test_indexing_with_vectorized_incoming_beam():
    """
    Test that indexing into a vectorised outgoing beam works when the vectorisation
    originates in the incoming beam.
    """
    quadrupole = cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(0.1))
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=1_000,
        sigma_x=torch.tensor(1e-5),
        energy=torch.rand((5, 2)) * 154e6,
    )

    outgoing = quadrupole.track(incoming)
    sub_beam = outgoing[:3]

    assert sub_beam.particles.shape == torch.Size([3, 2, 1_000, 7])
    assert sub_beam.energy.shape == torch.Size([3, 2])
    assert sub_beam.particle_charges.shape == torch.Size([3, 2, 1_000])
    assert sub_beam.survival_probabilities.shape == torch.Size([3, 2, 1_000])

    assert torch.allclose(sub_beam.particles, outgoing.particles[:3])
    assert torch.allclose(sub_beam.energy, outgoing.energy[:3])
    assert torch.allclose(sub_beam.particle_charges, outgoing.particle_charges)
    assert torch.allclose(
        sub_beam.survival_probabilities, outgoing.survival_probabilities
    )


def test_indexing_fails_for_inconsitent_vectorization():
    """
    Test that indexing into a vectorised beam fails when the vectorisation is
    inconsistent, i.e. not broadcastable.
    """
    beam = cheetah.ParticleBeam.from_parameters(
        sigma_x=torch.rand((5, 2)), energy=torch.rand((4, 2)) * 154e6
    )

    with pytest.raises(RuntimeError):
        _ = beam[:3]


def test_indexing_fails_for_invalid_index():
    """Test that indexing into a vectorised beam fails when the index is invalid."""
    beam = cheetah.ParticleBeam.from_parameters(energy=torch.rand((5, 2)) * 154e6)

    with pytest.raises(IndexError):
        _ = beam[6]


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_random_subsample_gaussian_properties(device: torch.device):
    """
    Test that a random subsample of a beam has the correct number of particles and
    similar parameters as the original.
    """
    original_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001", device=device, dtype=torch.float32
    )
    subsampled_beam = original_beam.randomly_subsampled(50_000)

    assert subsampled_beam.num_particles == 50_000

    assert torch.isclose(subsampled_beam.energy, original_beam.energy)
    assert torch.isclose(
        subsampled_beam.total_charge, original_beam.total_charge, rtol=1e-5
    )
    assert subsampled_beam.species.name == original_beam.species.name
    assert torch.isclose(
        subsampled_beam.species.charge_coulomb, original_beam.species.charge_coulomb
    )
    assert torch.isclose(subsampled_beam.species.mass_kg, original_beam.species.mass_kg)

    assert torch.isclose(subsampled_beam.mu_x, original_beam.mu_x, rtol=1e-5, atol=1e-5)
    assert torch.isclose(
        subsampled_beam.mu_px, original_beam.mu_px, rtol=1e-5, atol=1e-5
    )
    assert torch.isclose(subsampled_beam.mu_y, original_beam.mu_y, rtol=1e-5, atol=1e-5)
    assert torch.isclose(
        subsampled_beam.mu_py, original_beam.mu_py, rtol=1e-5, atol=1e-5
    )
    assert torch.isclose(
        subsampled_beam.mu_tau, original_beam.mu_tau, rtol=1e-5, atol=1e-5
    )
    assert torch.isclose(subsampled_beam.mu_p, original_beam.mu_p, rtol=1e-5, atol=1e-5)
    assert torch.isclose(
        subsampled_beam.sigma_x, original_beam.sigma_x, rtol=1e-5, atol=1e-5
    )
    assert torch.isclose(
        subsampled_beam.sigma_px, original_beam.sigma_px, rtol=1e-5, atol=1e-5
    )
    assert torch.isclose(
        subsampled_beam.sigma_y, original_beam.sigma_y, rtol=1e-5, atol=1e-5
    )
    assert torch.isclose(
        subsampled_beam.sigma_py, original_beam.sigma_py, rtol=1e-5, atol=1e-5
    )
    assert torch.isclose(
        subsampled_beam.sigma_tau, original_beam.sigma_tau, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_random_subsample_energy_distance_better_than_gaussian(device: torch.device):
    """
    Test that on a non-Gaussian beam, the energy distance from the random subsample to
    the original beam is much (5x) lower than the energy distance from a Gaussian
    subsample (via conversion to `ParameterBeam` and back) to the original beam.
    """
    original_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001", device=device, dtype=torch.float32
    )
    randomly_subsampled_beam = original_beam.randomly_subsampled(50_000)
    gaussian_subsampled_beam = original_beam.as_parameter_beam().as_particle_beam(
        num_particles=50_000
    )

    for dim_name in ["x", "px", "y", "py", "tau", "p"]:
        original_dim = getattr(original_beam, dim_name)
        randomly_subsampled_dim = getattr(randomly_subsampled_beam, dim_name)
        gaussian_subsampled_dim = getattr(gaussian_subsampled_beam, dim_name)

        energy_distance_to_random_subsample = scipy.stats.energy_distance(
            original_dim.cpu(), randomly_subsampled_dim.cpu()
        )
        energy_distance_to_gaussian_subsample = scipy.stats.energy_distance(
            original_dim.cpu(), gaussian_subsampled_dim.cpu()
        )

        assert (
            5 * energy_distance_to_random_subsample
            < energy_distance_to_gaussian_subsample
        )


def test_vectorized_conversion_to_parameter_beam_and_back():
    """
    Test that converting a vectorised `ParticleBeam` to a `ParameterBeam` and back does
    not throw errors and results in a beam with the same parameters.
    """
    original_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,  # Does not need as many particles as reconstructed beam
        mu_x=torch.tensor((2e-4, 3e-4)),
        sigma_x=torch.tensor((2e-5, 3e-5)),
        energy=torch.tensor((1e7, 2e7)),
    )

    # Vectorise survival probabilities make make them slightly different
    original_beam.survival_probabilities = original_beam.survival_probabilities.repeat(
        3, 1, 1
    )
    original_beam.survival_probabilities[0, 0, : int(10_000 / 3)] = 0.3
    original_beam.survival_probabilities[1, 0, : int(10_000 / 3)] = 0.6

    roundtrip_converted_beam = original_beam.as_parameter_beam().as_particle_beam(
        num_particles=10_000_000
    )

    assert isinstance(roundtrip_converted_beam, cheetah.ParticleBeam)
    assert torch.allclose(
        original_beam.mu_x, roundtrip_converted_beam.mu_x, rtol=1e-3, atol=1e-6
    )
    assert torch.allclose(
        original_beam.mu_px, roundtrip_converted_beam.mu_px, rtol=1e-3, atol=1e-6
    )
    assert torch.allclose(
        original_beam.mu_y, roundtrip_converted_beam.mu_y, rtol=1e-3, atol=1e-6
    )
    assert torch.allclose(
        original_beam.mu_py, roundtrip_converted_beam.mu_py, rtol=1e-3, atol=1e-6
    )
    assert torch.allclose(
        original_beam.mu_tau, roundtrip_converted_beam.mu_tau, rtol=1e-3, atol=1e-6
    )
    assert torch.allclose(
        original_beam.mu_p, roundtrip_converted_beam.mu_p, rtol=1e-3, atol=1e-5
    )
    assert torch.allclose(
        original_beam.sigma_x, roundtrip_converted_beam.sigma_x, rtol=1e-3
    )
    assert torch.allclose(
        original_beam.sigma_px, roundtrip_converted_beam.sigma_px, rtol=1e-3
    )
    assert torch.allclose(
        original_beam.sigma_y, roundtrip_converted_beam.sigma_y, rtol=1e-3
    )
    assert torch.allclose(
        original_beam.sigma_py, roundtrip_converted_beam.sigma_py, rtol=1e-3
    )
    assert torch.allclose(
        original_beam.sigma_tau, roundtrip_converted_beam.sigma_tau, rtol=1e-3
    )
    assert torch.allclose(
        original_beam.sigma_p, roundtrip_converted_beam.sigma_p, rtol=1e-3
    )
    assert torch.allclose(
        original_beam.energy, roundtrip_converted_beam.energy, rtol=1e-3
    )
    assert torch.allclose(
        original_beam.total_charge, roundtrip_converted_beam.total_charge, rtol=1e-3
    )
    assert torch.allclose(original_beam.s, roundtrip_converted_beam.s, rtol=1e-3)
