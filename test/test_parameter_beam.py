import numpy as np
import torch

from cheetah import ParameterBeam


def test_create_from_parameters():
    """
    Test that a `ParameterBeam` created from parameters actually has those parameters.
    """
    beam = ParameterBeam.from_parameters(
        mu_x=torch.tensor(1e-5),
        mu_xp=torch.tensor(1e-7),
        mu_y=torch.tensor(2e-5),
        mu_yp=torch.tensor(2e-7),
        sigma_x=torch.tensor(1.75e-7),
        sigma_xp=torch.tensor(2e-7),
        sigma_y=torch.tensor(1.75e-7),
        sigma_yp=torch.tensor(2e-7),
        sigma_s=torch.tensor(0.000001),
        sigma_p=torch.tensor(0.000001),
        cor_x=torch.tensor(0.0),
        cor_y=torch.tensor(0.0),
        cor_s=torch.tensor(0.0),
        energy=torch.tensor(1e7),
    )

    assert np.isclose(beam.mu_x, 1e-5)
    assert np.isclose(beam.mu_xp, 1e-7)
    assert np.isclose(beam.mu_y, 2e-5)
    assert np.isclose(beam.mu_yp, 2e-7)
    assert np.isclose(beam.sigma_x, 1.75e-7)
    assert np.isclose(beam.sigma_xp, 2e-7)
    assert np.isclose(beam.sigma_y, 1.75e-7)
    assert np.isclose(beam.sigma_yp, 2e-7)
    assert np.isclose(beam.sigma_s, 0.000001)
    assert np.isclose(beam.sigma_p, 0.000001)
    assert np.isclose(beam.energy, 1e7)


def test_transform_to():
    """
    Test that a `ParameterBeam` transformed to new parameters actually has those new
    parameters.
    """
    original_beam = ParameterBeam.from_parameters()
    transformed_beam = original_beam.transformed_to(
        mu_x=torch.tensor(1e-5),
        mu_xp=torch.tensor(1e-7),
        mu_y=torch.tensor(2e-5),
        mu_yp=torch.tensor(2e-7),
        sigma_x=torch.tensor(1.75e-7),
        sigma_xp=torch.tensor(2e-7),
        sigma_y=torch.tensor(1.75e-7),
        sigma_yp=torch.tensor(2e-7),
        sigma_s=torch.tensor(0.000001),
        sigma_p=torch.tensor(0.000001),
        energy=torch.tensor(1e7),
    )

    assert isinstance(transformed_beam, ParameterBeam)
    assert np.isclose(transformed_beam.mu_x, 1e-5)
    assert np.isclose(transformed_beam.mu_xp, 1e-7)
    assert np.isclose(transformed_beam.mu_y, 2e-5)
    assert np.isclose(transformed_beam.mu_yp, 2e-7)
    assert np.isclose(transformed_beam.sigma_x, 1.75e-7)
    assert np.isclose(transformed_beam.sigma_xp, 2e-7)
    assert np.isclose(transformed_beam.sigma_y, 1.75e-7)
    assert np.isclose(transformed_beam.sigma_yp, 2e-7)
    assert np.isclose(transformed_beam.sigma_s, 0.000001)
    assert np.isclose(transformed_beam.sigma_p, 0.000001)
    assert np.isclose(transformed_beam.energy, 1e7)


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

    assert np.isclose(beam.beta_x, 5.91253676811640894)
    assert np.isclose(beam.alpha_x, 3.55631307633660354)
    assert np.isclose(beam.emittance_x, 3.494768647122823e-09)
    assert np.isclose(beam.beta_y, 5.91253676811640982)
    assert np.isclose(beam.alpha_y, 2e-7)
    assert np.isclose(beam.emittance_y, 3.497810737006068e-09)
    assert np.isclose(beam.energy, 6e6)
