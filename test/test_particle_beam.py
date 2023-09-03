import numpy as np

from cheetah import ParticleBeam


def test_create_from_parameters():
    """
    Test that a `ParticleBeam` created from parameters actually has those parameters.
    """
    beam = ParticleBeam.from_parameters(
        n=1_000_000,
        mu_x=1e-5,
        mu_xp=1e-7,
        mu_y=2e-5,
        mu_yp=2e-7,
        sigma_x=1.75e-7,
        sigma_xp=2e-7,
        sigma_y=1.75e-7,
        sigma_yp=2e-7,
        sigma_s=0.000001,
        sigma_p=0.000001,
        cor_x=0,
        cor_y=0,
        cor_s=0,
        energy=1e7,
    )

    assert beam.n == 1_000_000
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
    Test that a `ParticleBeam` transformed to new parameters actually has those new
    parameters.
    """
    original_beam = ParticleBeam.from_parameters()
    transformed_beam = original_beam.transformed_to(
        mu_x=1e-5,
        mu_xp=1e-7,
        mu_y=2e-5,
        mu_yp=2e-7,
        sigma_x=1.75e-7,
        sigma_xp=2e-7,
        sigma_y=1.75e-7,
        sigma_yp=2e-7,
        sigma_s=0.000001,
        sigma_p=0.000001,
        energy=1e7,
    )

    assert isinstance(transformed_beam, ParticleBeam)
    assert original_beam.n == transformed_beam.n

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
