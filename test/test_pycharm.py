import cheetah
import numpy as np
import torch

beam = cheetah.ParameterBeam.from_parameters(sigma_x=175e-9, sigma_y=175e-9)


def test_beam_mu():
    actual = beam._mu
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_beam_cov():
    actual = beam._cov
    expected = torch.tensor(
        [
            [
                3.0625e-14,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                4.0000e-14,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                3.0625e-14,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                4.0000e-14,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.0000e-12,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.0000e-12,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_beam_energy():
    actual = beam.energy
    expected = 100000000.000
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


beam2 = beam.transformed_to(sigma_x=123e-6)


def test_beam2_mu():
    actual = beam2._mu
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_beam2_cov():
    actual = beam2._cov
    expected = torch.tensor(
        [
            [
                1.5129e-08,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.0000e-12,
                0.0000e00,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.0000e-12,
                0.0000e00,
            ],
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ],
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_beam2_energy():
    actual = beam2.energy
    expected = 100000000.000
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


beam3 = cheetah.ParticleBeam.from_parameters(sigma_x=175e-9, sigma_y=175e-9)


def test_beam3_n():
    assert beam3.n == 100000


def test_beam3_mean():
    actual = beam3.particles.mean(axis=0)
    expected = torch.tensor(
        [
            -9.6960e-11,
            -5.3534e-11,
            4.4916e-11,
            9.2226e-10,
            5.1672e-09,
            1.9966e-09,
            1.0000e00,
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_beam3_cov():
    actual = np.cov(beam3.particles.t().numpy())
    expected = torch.tensor(
        [
            [
                3.07948938e-14,
                -1.83413905e-16,
                -1.68399788e-16,
                4.25610929e-17,
                -7.98809553e-16,
                4.54601961e-16,
                0.00000000e00,
            ],
            [
                -1.83413905e-16,
                3.99321871e-14,
                1.52709870e-16,
                5.61528738e-19,
                9.21760507e-17,
                2.64697265e-16,
                0.00000000e00,
            ],
            [
                -1.68399788e-16,
                1.52709870e-16,
                3.05111147e-14,
                -1.70158009e-16,
                1.68751952e-16,
                -4.03071599e-16,
                0.00000000e00,
            ],
            [
                4.25610929e-17,
                5.61528738e-19,
                -1.70158009e-16,
                4.02518627e-14,
                -6.30815231e-17,
                2.41067632e-17,
                0.00000000e00,
            ],
            [
                -7.98809553e-16,
                9.21760507e-17,
                1.68751952e-16,
                -6.30815231e-17,
                9.89274528e-13,
                1.88611486e-15,
                0.00000000e00,
            ],
            [
                4.54601961e-16,
                2.64697265e-16,
                -4.03071599e-16,
                2.41067632e-17,
                1.88611486e-15,
                9.99455081e-13,
                0.00000000e00,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
        ]
    )
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)


def test_beam3_energy():
    actual = beam3.energy
    expected = 100000000.000
    assert np.allclose(actual, expected, atol=1e-05, rtol=1e-08, equal_nan=False)
