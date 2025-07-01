import numpy as np
import torch

from cheetah import ParameterBeam


def test_create_from_parameters():
    """
    Test that a `ParameterBeam` created from parameters actually has those parameters.
    """
    beam = ParameterBeam.from_parameters(
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
    )

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


def test_transform_to():
    """
    Test that a `ParameterBeam` transformed to new parameters actually has those new
    parameters.
    """
    original_beam = ParameterBeam.from_parameters()
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

    assert isinstance(transformed_beam, ParameterBeam)
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
    beam = ParameterBeam.from_twiss(
        beta_x=torch.tensor(5.91253676811640894),
        alpha_x=torch.tensor(3.55631307633660354),
        emittance_x=torch.tensor(3.494768647122823e-09),
        beta_y=torch.tensor(5.91253676811640982),
        alpha_y=torch.tensor(2e-7),
        emittance_y=torch.tensor(3.497810737006068e-09),
        energy=torch.tensor(6e6),
    )

    assert np.isclose(beam.beta_x.cpu().numpy(), 5.91253676811640894)
    assert np.isclose(beam.alpha_x.cpu().numpy(), 3.55631307633660354)
    assert np.isclose(beam.emittance_x.cpu().numpy(), 3.494768647122823e-09)
    assert np.isclose(beam.beta_y.cpu().numpy(), 5.91253676811640982)
    assert np.isclose(beam.alpha_y.cpu().numpy(), 2e-7)
    assert np.isclose(beam.emittance_y.cpu().numpy(), 3.497810737006068e-09)
    assert np.isclose(beam.energy.cpu().numpy(), 6e6)


def test_from_twiss_dtype():
    """
    Test that a `ParameterBeam` created from twiss parameters has the requested `dtype`.
    """
    beam = ParameterBeam.from_twiss(
        beta_x=torch.tensor(5.91253676811640894),
        alpha_x=torch.tensor(3.55631307633660354),
        emittance_x=torch.tensor(3.494768647122823e-09),
        beta_y=torch.tensor(5.91253676811640982),
        alpha_y=torch.tensor(2e-7),
        emittance_y=torch.tensor(3.497810737006068e-09),
        energy=torch.tensor(6e6),
        dtype=torch.float64,
    )

    assert np.isclose(beam.beta_x.cpu().numpy(), 5.91253676811640894)
    assert np.isclose(beam.alpha_x.cpu().numpy(), 3.55631307633660354)
    assert np.isclose(beam.emittance_x.cpu().numpy(), 3.494768647122823e-09)
    assert np.isclose(beam.beta_y.cpu().numpy(), 5.91253676811640982)
    assert np.isclose(beam.alpha_y.cpu().numpy(), 2e-7)
    assert np.isclose(beam.emittance_y.cpu().numpy(), 3.497810737006068e-09)
    assert np.isclose(beam.energy.cpu().numpy(), 6e6)

    assert beam.mu.dtype == torch.float64
    assert beam.cov.dtype == torch.float64


def test_conversion_to_and_from_particle_beam():
    """
    Test that converting a `ParameterBeam` to a `ParticleBeam` and back results in the
    an equivalent `ParameterBeam`.

    NOTE: Runs in double precision to avoid numerical issues.
    """
    original_parameter_beam = ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001", dtype=torch.float64
    )
    particle_beam = original_parameter_beam.as_particle_beam(num_particles=10_000_000)
    reconstructed_parameter_beam = particle_beam.as_parameter_beam()

    # Side check that the `ParticleBeam` has the correct number of particles
    assert particle_beam.num_particles == 10_000_000

    # Check that reconstructed `ParameterBeam` has the same parameters as the original
    assert torch.isclose(
        original_parameter_beam.mu_x, reconstructed_parameter_beam.mu_x, atol=1e-6
    )
    assert torch.isclose(
        original_parameter_beam.mu_y, reconstructed_parameter_beam.mu_y, atol=1e-6
    )
    assert torch.isclose(
        original_parameter_beam.sigma_x, reconstructed_parameter_beam.sigma_x, rtol=1e-3
    )
    assert torch.isclose(
        original_parameter_beam.sigma_y, reconstructed_parameter_beam.sigma_y, rtol=1e-3
    )
    assert torch.isclose(
        original_parameter_beam.mu_px, reconstructed_parameter_beam.mu_px, atol=1e-6
    )
    assert torch.isclose(
        original_parameter_beam.mu_py, reconstructed_parameter_beam.mu_py, atol=1e-6
    )
    assert torch.isclose(
        original_parameter_beam.sigma_px,
        reconstructed_parameter_beam.sigma_px,
        rtol=1e-3,
    )
    assert torch.isclose(
        original_parameter_beam.sigma_py,
        reconstructed_parameter_beam.sigma_py,
        rtol=1e-3,
    )
    assert torch.isclose(
        original_parameter_beam.mu_tau, reconstructed_parameter_beam.mu_tau, atol=1e-6
    )
    assert torch.isclose(
        original_parameter_beam.sigma_tau,
        reconstructed_parameter_beam.sigma_tau,
        rtol=1e-3,
    )
    assert torch.isclose(
        original_parameter_beam.mu_p, reconstructed_parameter_beam.mu_p, atol=1e-5
    )
    assert torch.isclose(
        original_parameter_beam.sigma_p, reconstructed_parameter_beam.sigma_p, rtol=1e-3
    )
    assert torch.isclose(
        original_parameter_beam.energy, reconstructed_parameter_beam.energy
    )
    assert torch.isclose(
        original_parameter_beam.total_charge, reconstructed_parameter_beam.total_charge
    )
