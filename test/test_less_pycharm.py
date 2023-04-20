import sys

import numpy as np
import torch

import cheetah

"""
Test Beam, which can be found in GitHub in the 
folder benchmark/cheetah/ACHIP_EA1_2021.1351.001
"""


beam1 = cheetah.ParameterBeam.from_astra("../benchmark/cheetah/ACHIP_EA1_2021.1351.001")
beam2 = cheetah.ParticleBeam.from_astra("../benchmark/cheetah/ACHIP_EA1_2021.1351.001")


def test_ParticleBeam_n():
    assert beam2.n == 100000


def test_ParameterBeam_energy():
    actual = beam1.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParticleBeam_energy():
    actual = beam2.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParameterBeam_mu():
    actual = beam1._mu
    expected = torch.tensor(
        [
            8.2413e-07,
            5.9885e-08,
            -1.7276e-06,
            -1.1746e-07,
            5.7250e-06,
            3.8292e-04,
            1.0000e00,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParticleBeam_particles_mean():
    actual = beam2.particles.mean(axis=0)
    expected = torch.tensor(
        [
            8.2413e-07,
            5.9885e-08,
            -1.7276e-06,
            -1.1746e-07,
            5.7250e-06,
            3.8292e-04,
            1.0000e00,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParameterBeam_ParticleBeam_mu_dif():
    assert torch.allclose(
        beam1._mu, beam2.particles.mean(axis=0), rtol=1e-5, atol=1e-8, equal_nan=False
    )


def test_ParameterBeam_cov():
    actual = beam1._cov
    expected = torch.tensor(
        [
            [
                3.0589e-08,
                5.5679e-10,
                8.0207e-13,
                2.2111e-13,
                -6.3953e-13,
                -7.6916e-12,
                0.0000e00,
            ],
            [
                5.5679e-10,
                1.3538e-11,
                9.8643e-14,
                6.4855e-15,
                -3.6896e-14,
                -8.0708e-14,
                0.0000e00,
            ],
            [
                8.0207e-13,
                9.8643e-14,
                3.0693e-08,
                5.6076e-10,
                6.0425e-13,
                3.3948e-11,
                0.0000e00,
            ],
            [
                2.2111e-13,
                6.4855e-15,
                5.6076e-10,
                1.3646e-11,
                6.4452e-14,
                5.3652e-12,
                0.0000e00,
            ],
            [
                -6.3953e-13,
                -3.6896e-14,
                6.0425e-13,
                6.4452e-14,
                6.4185e-11,
                3.0040e-09,
                0.0000e00,
            ],
            [
                -7.6916e-12,
                -8.0708e-14,
                3.3948e-11,
                5.3652e-12,
                3.0040e-09,
                5.2005e-06,
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
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParticleBeam_cov():
    actual = np.cov(beam2.particles.t().numpy())
    expected = np.array(
        [
            [
                3.05892720e-08,
                5.56790623e-10,
                8.02068059e-13,
                2.21114027e-13,
                -6.39527178e-13,
                -7.69157365e-12,
                0.00000000e00,
            ],
            [
                5.56790623e-10,
                1.35380005e-11,
                9.86434299e-14,
                6.48545528e-15,
                -3.68956168e-14,
                -8.07087941e-14,
                0.00000000e00,
            ],
            [
                8.02068059e-13,
                9.86434299e-14,
                3.06934414e-08,
                5.60758572e-10,
                6.04253726e-13,
                3.39481957e-11,
                0.00000000e00,
            ],
            [
                2.21114027e-13,
                6.48545528e-15,
                5.60758572e-10,
                1.36463749e-11,
                6.44515124e-14,
                5.36516751e-12,
                0.00000000e00,
            ],
            [
                -6.39527178e-13,
                -3.68956168e-14,
                6.04253726e-13,
                6.44515124e-14,
                6.41849709e-11,
                3.00400242e-09,
                0.00000000e00,
            ],
            [
                -7.69157365e-12,
                -8.07087941e-14,
                3.39481957e-11,
                5.36516751e-12,
                3.00400242e-09,
                5.20046739e-06,
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
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParameterBeam_ParticleBeam_cov_dif():
    assert np.allclose(
        beam1._cov,
        np.cov(beam2.particles.t().numpy()),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


segment = cheetah.Segment(
    [cheetah.HorizontalCorrector(length=0.02, name="quad"), cheetah.Drift(length=2.0)]
)
segment.quad.angle = 2e-3

result1 = segment(beam1)
result2 = segment(beam2)


def test_ParticleBeam_result2_n():
    assert result2.n == 100000


def test_ParameterBeam_result1_energy():
    actual = result1.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParticleBeam_result2_energy():
    actual = result1.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParameterBeam_result1_mu():
    actual = result1._mu
    expected = torch.tensor(
        [
            4.0009e-03,
            2.0001e-03,
            -1.9649e-06,
            -1.1746e-07,
            5.7424e-06,
            3.8292e-04,
            1.0000e00,
        ]
    )
    assert np.allclose(actual, expected, rtol=1e-3, atol=1e-8, equal_nan=False)


def test_ParticleBeam_result2_particles_mean():
    actual = result2.particles.mean(axis=0)
    expected = torch.tensor(
        [
            4.0009e-03,
            2.0001e-03,
            -1.9649e-06,
            -1.1746e-07,
            5.7424e-06,
            3.8292e-04,
            1.0000e00,
        ]
    )
    assert np.allclose(actual, expected, rtol=1e-3, atol=1e-8, equal_nan=False)


def test_ParameterBeam_result1_ParticleBeam_result2_mu_dif():
    assert torch.allclose(
        result1._mu,
        result2.particles.mean(axis=0),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_ParameterBeam_result1_cov():
    actual = result1._cov
    expected = torch.tensor(
        [
            [
                3.2894e-08,
                5.8414e-10,
                1.4744e-12,
                2.3421e-13,
                -7.1441e-13,
                -7.8546e-12,
                0.0000e00,
            ],
            [
                5.8414e-10,
                1.3538e-11,
                1.1174e-13,
                6.4855e-15,
                -3.6899e-14,
                -8.0708e-14,
                0.0000e00,
            ],
            [
                1.4744e-12,
                1.1174e-13,
                3.3015e-08,
                5.8832e-10,
                7.3648e-13,
                4.4786e-11,
                0.0000e00,
            ],
            [
                2.3421e-13,
                6.4855e-15,
                5.8832e-10,
                1.3646e-11,
                6.4695e-14,
                5.3652e-12,
                0.0000e00,
            ],
            [
                -7.1441e-13,
                -3.6899e-14,
                7.3648e-13,
                6.4695e-14,
                6.4468e-11,
                3.2398e-09,
                0.0000e00,
            ],
            [
                -7.8546e-12,
                -8.0708e-14,
                4.4786e-11,
                5.3652e-12,
                3.2398e-09,
                5.2005e-06,
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
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_ParticleBeam_result2_cov():
    actual = np.cov(result2.particles.t().numpy())
    expected = np.array(
        [
            [
                3.28939465e-08,
                5.84137324e-10,
                1.47452394e-12,
                2.34217086e-13,
                -7.14414075e-13,
                -7.85582424e-12,
                0.00000000e00,
            ],
            [
                5.84137324e-10,
                1.35379983e-11,
                1.11718174e-13,
                6.48437255e-15,
                -3.68973966e-14,
                -8.13532478e-14,
                0.00000000e00,
            ],
            [
                1.47452394e-12,
                1.11718174e-13,
                3.30145887e-08,
                5.88324249e-10,
                7.36476744e-13,
                4.47858536e-11,
                0.00000000e00,
            ],
            [
                2.34217086e-13,
                6.48437255e-15,
                5.88324249e-10,
                1.36463749e-11,
                6.46948015e-14,
                5.36516751e-12,
                0.00000000e00,
            ],
            [
                -7.14414075e-13,
                -3.68973966e-14,
                7.36476744e-13,
                6.46948015e-14,
                6.44681053e-11,
                3.23982437e-09,
                0.00000000e00,
            ],
            [
                -7.85582424e-12,
                -8.13532478e-14,
                4.47858536e-11,
                5.36516751e-12,
                3.23982437e-09,
                5.20046739e-06,
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
    assert np.allclose(actual, expected, rtol=1e-05, atol=1e-14, equal_nan=False)


def test_ParameterBeam_result1_ParticleBeam_result2_cov_dif():
    assert np.allclose(
        result1._cov,
        np.cov(result2.particles.t().numpy()),
        rtol=1e-3,
        atol=1e-8,
        equal_nan=True,
    )
