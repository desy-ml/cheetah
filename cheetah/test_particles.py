import numpy as np
import torch

import cheetah

ParameterBeam_parameters = cheetah.ParameterBeam.from_parameters(
    sigma_x=175e-9, sigma_y=175e-9
)
ParticleBeam_parameters = cheetah.ParticleBeam.from_parameters(
    sigma_x=175e-9, sigma_y=175e-9
)

ParameterBeam_transformed = ParameterBeam_parameters.transformed_to(
    mu_x=8.6423e-06,
    mu_xp=5.9384e-07,
    mu_y=-2.7276e-07,
    mu_yp=-3.1776e-08,
    sigma_x=123e-06,
    sigma_xp=7e-07,
    sigma_y=8e-08,
    sigma_yp=2e-06,
    sigma_s=2e-05,
    sigma_p=2e-06,
    energy=130089263.44785302,
)
ParticleBeam_transformed = ParticleBeam_parameters.transformed_to(
    mu_x=8.6423e-06,
    mu_xp=5.9384e-07,
    mu_y=-2.7276e-07,
    mu_yp=-3.1776e-08,
    sigma_x=123e-06,
    sigma_xp=7e-07,
    sigma_y=8e-08,
    sigma_yp=2e-06,
    sigma_s=2e-05,
    sigma_p=2e-06,
    energy=130089263.44785302,
)

ParameterBeam_astra = cheetah.ParameterBeam.from_astra(
    "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)
ParticleBeam_astra = cheetah.ParticleBeam.from_astra(
    "benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)

# Expected Results
ParticleBeam_parameters_n = 100000
ParameterBeam_parameters_Energy = 100000000.0
ParticleBeam_parameters_Energy = 100000000.0
ParameterBeam_parameters_mu = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
ParticleBeam_parameters_mu = torch.tensor(
    [
        4.9239e-10,
        -8.5083e-10,
        -1.3031e-10,
        -4.3553e-10,
        3.5803e-09,
        -9.5884e-10,
        1.0000e00,
    ]
)
ParameterBeam_parameters_cov = torch.tensor(
    [
        [3.0625e-14, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 4.0000e-14, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 3.0625e-14, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 4.0000e-14, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 1.0000e-12, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 1.0000e-12, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
ParticleBeam_parameters_cov = [
    [
        3.07914111e-14,
        5.88738347e-17,
        4.57326862e-17,
        -1.26756988e-16,
        -4.72561871e-16,
        -1.19928956e-16,
        0.00000000e00,
    ],
    [
        5.88738347e-17,
        3.99582590e-14,
        -2.60470840e-17,
        2.20780170e-16,
        -4.48037915e-16,
        -9.33762811e-16,
        0.00000000e00,
    ],
    [
        4.57326862e-17,
        -2.60470840e-17,
        3.06947419e-14,
        -8.46318977e-17,
        -2.09777114e-16,
        1.39406364e-16,
        0.00000000e00,
    ],
    [
        -1.26756988e-16,
        2.20780170e-16,
        -8.46318977e-17,
        3.99782337e-14,
        8.78611323e-18,
        4.48218607e-16,
        0.00000000e00,
    ],
    [
        -4.72561871e-16,
        -4.48037915e-16,
        -2.09777114e-16,
        8.78611323e-18,
        9.93149970e-13,
        -2.61481282e-15,
        0.00000000e00,
    ],
    [
        -1.19928956e-16,
        -9.33762811e-16,
        1.39406364e-16,
        4.48218607e-16,
        -2.61481282e-15,
        1.00019479e-12,
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

ParticleBeam_astra_n = 100000
ParameterBeam_astra_Energy = 107315902.44355084
ParticleBeam_astra_Energy = 107315902.44355084
ParameterBeam_astra_mu = torch.tensor(
    [
        8.2413e-07,
        5.9885e-08,
        -1.7276e-06,
        -1.1746e-07,
        5.7250e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
ParticleBeam_astra_mu = torch.tensor(
    [
        8.2413e-07,
        5.9885e-08,
        -1.7276e-06,
        -1.1746e-07,
        5.7250e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
ParameterBeam_astra_cov = torch.tensor(
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
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
ParticleBeam_astra_cov = [
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

ParticleBeam_transformed_n = 100000
ParameterBeam_transformed_Energy = 130089263.44785301
ParticleBeam_transformed_Energy = 130089263.44785301
ParameterBeam_transformed_mu = torch.tensor(
    [8.6423e-06, 5.9384e-07, -2.7276e-07, -3.1776e-08, 0.0000e00, 0.0000e00, 1.0000e00]
)

ParticleBeam_transformed_mu = torch.tensor(
    [
        8.6423e-06,
        5.9384e-07,
        -2.7276e-07,
        -3.1776e-08,
        -2.7210e-08,
        -2.4420e-09,
        1.0000e00,
    ]
)

ParameterBeam_transformed_cov = torch.tensor(
    [
        [1.5129e-08, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 4.9000e-13, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 6.4000e-15, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 4.0000e-12, 0.0000e00, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 4.0000e-10, 0.0000e00, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 4.0000e-12, 0.0000e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
ParticleBeam_transformed_cov = [
    [
        1.51290015e-08,
        -4.86927066e-13,
        2.33863504e-14,
        -5.19859903e-13,
        -5.85372522e-12,
        -1.02460980e-12,
        0.00000000e00,
    ],
    [
        -4.86927066e-13,
        4.90000004e-13,
        1.39565416e-16,
        -1.98664945e-15,
        -3.72587989e-14,
        -1.92397243e-15,
        0.00000000e00,
    ],
    [
        2.33863504e-14,
        1.39565416e-16,
        6.40000034e-15,
        -2.18124634e-17,
        -8.34253985e-15,
        3.54562001e-16,
        0.00000000e00,
    ],
    [
        -5.19859903e-13,
        -1.98664945e-15,
        -2.18124634e-17,
        4.00000023e-12,
        -9.30197489e-14,
        2.97484121e-14,
        0.00000000e00,
    ],
    [
        -5.85372522e-12,
        -3.72587989e-14,
        -8.34253985e-15,
        -9.30197489e-14,
        3.99999961e-10,
        -1.11188776e-13,
        0.00000000e00,
    ],
    [
        -1.02460980e-12,
        -1.92397243e-15,
        3.54562001e-16,
        2.97484121e-14,
        -1.11188776e-13,
        4.00000025e-12,
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


# From parameters
def test_ParticleBeam_parameters_n():
    assert ParticleBeam_parameters.n == ParticleBeam_parameters_n


def test_ParameterBeam_parameters_energy():
    assert np.allclose(
        ParameterBeam_parameters.energy,
        ParameterBeam_parameters_Energy,
        rtol=1e-04,
        atol=1e-08,
        equal_nan=False,
    )


def test_ParticleBeam_parameters_energy():
    assert np.allclose(
        ParticleBeam_parameters.energy,
        ParticleBeam_parameters_Energy,
        rtol=1e-04,
        atol=1e-08,
        equal_nan=False,
    )


def test_ParameterBeam_parameters_mu():
    assert torch.allclose(
        ParameterBeam_parameters._mu,
        ParameterBeam_parameters_mu,
        rtol=1e-04,
        atol=1e-09,
        equal_nan=False,
    )


def test_ParticleBeam_parameters_mu():
    assert torch.allclose(
        ParticleBeam_parameters.particles.mean(axis=0),
        ParticleBeam_parameters_mu,
        rtol=1e-04,
        atol=1e-08,
        equal_nan=False,
    )


def test_ParameterBeam_parameters_ParticleBeam_parameters_mu_dif():
    assert torch.allclose(
        ParameterBeam_parameters._mu,
        ParticleBeam_parameters.particles.mean(axis=0),
        rtol=1e-04,
        atol=1e-08,
        equal_nan=False,
    )


def test_ParameterBeam_parameters_cov():
    assert torch.allclose(
        ParameterBeam_parameters._cov,
        ParameterBeam_parameters_cov,
        rtol=1e-04,
        atol=1e-14,
        equal_nan=False,
    )


def test_ParticleBeam_parameters_cov():
    assert np.allclose(
        np.cov(ParticleBeam_parameters.particles.t().numpy()),
        ParticleBeam_parameters_cov,
        rtol=1e-04,
        atol=1e-14,
        equal_nan=False,
    )


def test_ParameterBeam_parameters_ParticleBeam_parameters_cov_dif():
    assert np.allclose(
        ParameterBeam_parameters._cov,
        np.cov(ParticleBeam_parameters.particles.t().numpy()),
        rtol=1e-03,
        atol=1e-14,
        equal_nan=False,
    )


# Astra


def test_ParticleBeam_astra_n():
    assert ParticleBeam_astra.n == ParticleBeam_astra_n


def test_ParameterBeam_astra_Energy():
    assert np.isclose(
        ParameterBeam_astra.energy,
        ParameterBeam_astra_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_ParticleBeam_astra_Energy():
    assert np.isclose(
        ParticleBeam_astra.energy,
        ParticleBeam_astra_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_ParameterBeam_astra_mu():
    assert torch.allclose(
        ParameterBeam_astra._mu,
        ParameterBeam_astra_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_ParticleBeam_astra_mu():
    assert torch.allclose(
        ParticleBeam_astra.particles.mean(axis=0),
        ParticleBeam_astra_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_ParameterBeam_astra_ParticleBeam_astra_mu_dif():
    assert torch.allclose(
        ParameterBeam_astra._mu,
        ParticleBeam_astra.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_ParameterBeam_astra_cov():
    assert torch.allclose(
        ParameterBeam_astra._cov,
        ParameterBeam_astra_cov,
        rtol=1e-04,
        atol=1e-14,
        equal_nan=False,
    )


def test_ParticleBeam_astra_cov():
    assert np.allclose(
        np.cov(ParticleBeam_astra.particles.t().numpy()),
        ParticleBeam_astra_cov,
        rtol=1e-04,
        atol=1e-14,
        equal_nan=False,
    )


def test_ParameterBeam_astra_ParticleBeam_astra_cov_dif():
    assert np.allclose(
        ParameterBeam_astra._cov,
        np.cov(ParticleBeam_astra.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-14,
        equal_nan=False,
    )


# Transformed_to
def test_ParticleBeam_transformed_n():
    assert ParticleBeam_transformed.n == ParticleBeam_transformed_n


def test_ParameterBeam_transformed_energy():
    assert np.allclose(
        ParameterBeam_transformed.energy,
        ParameterBeam_transformed_Energy,
        rtol=1e-04,
        atol=1e-08,
        equal_nan=False,
    )


def test_ParticleBeam_transformed_energy():
    assert np.allclose(
        ParticleBeam_transformed.energy,
        ParticleBeam_transformed_Energy,
        rtol=1e-04,
        atol=1e-08,
        equal_nan=False,
    )


def test_ParameterBeam_transformed_mu():
    assert torch.allclose(
        ParameterBeam_transformed._mu,
        ParameterBeam_transformed_mu,
        rtol=1e-04,
        atol=1e-08,
        equal_nan=False,
    )


def test_ParticleBeam_transformed_mu():
    assert torch.allclose(
        ParticleBeam_transformed.particles.mean(axis=0),
        ParticleBeam_transformed_mu,
        rtol=1e-04,
        atol=1e-06,
        equal_nan=False,
    )


def test_ParameterBeam_transformed_ParticleBeam_transformed_mu_dif():
    assert torch.allclose(
        ParameterBeam_transformed._mu,
        ParticleBeam_transformed.particles.mean(axis=0),
        rtol=1e-03,
        atol=1e-06,
        equal_nan=False,
    )


def test_ParameterBeam_transformed_cov():
    assert torch.allclose(
        ParameterBeam_transformed._cov,
        ParameterBeam_transformed_cov,
        rtol=1e-03,
        atol=1e-06,
        equal_nan=False,
    )


def test_ParticleBeam_transformed_cov():
    assert np.allclose(
        np.cov(ParticleBeam_transformed.particles.t().numpy()),
        ParticleBeam_transformed_cov,
        rtol=1e-03,
        atol=1e-08,
        equal_nan=False,
    )


def test_ParameterBeam_transformed_ParticleBeam_transformed_cov_dif():
    assert np.allclose(
        ParameterBeam_transformed._cov,
        np.cov(ParticleBeam_transformed.particles.t().numpy()),
        rtol=1e-05,
        atol=1e-08,
        equal_nan=False,
    )
