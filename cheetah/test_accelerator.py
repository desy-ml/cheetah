import cheetah
import numpy as np
import torch

"""
Test Beam, which can be found in GitHub in the folder
benchmark/cheetah/ACHIP_EA1_2021.1351.001
"""


ParameterBeam = cheetah.ParameterBeam.from_astra(
    "../benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)
ParticleBeam = cheetah.ParticleBeam.from_astra(
    "../benchmark/cheetah/ACHIP_EA1_2021.1351.001"
)

# Segments
Drift = cheetah.Segment([cheetah.Drift(length=0.02, name="element")])

Quadrupole = cheetah.Segment(
    [cheetah.Quadrupole(length=0.02, misalignment=(1, 1), name="element")]
)
Quadrupole.element.k1 = 200

HorizontalCorrector = cheetah.Segment(
    [cheetah.HorizontalCorrector(length=0.02, name="element")]
)
HorizontalCorrector.element.angle = 2e-3

VerticalCorrector = cheetah.Segment(
    [cheetah.VerticalCorrector(length=0.02, name="element")]
)
VerticalCorrector.element.angle = 2e-3

Cavity = cheetah.Segment([cheetah.Cavity(length=0.02, name="element")])
Cavity.element.delta_energy = 1000

BPM = cheetah.Segment([cheetah.BPM(name="element")])

Screen = cheetah.Segment(
    [cheetah.Screen(resolution=(1000, 1000), pixel_size=1, name="element")]
)

Undulator = cheetah.Segment([cheetah.Undulator(length=0.02, name="element")])

# Segments applied on Beams
HorizontalCorrector_ParameterBeam = HorizontalCorrector(ParameterBeam)
HorizontalCorrector_ParticleBeam = HorizontalCorrector(ParticleBeam)

Quadrupole_ParameterBeam = Quadrupole(ParameterBeam)
Quadrupole_ParticleBeam = Quadrupole(ParticleBeam)

VerticalCorrector_ParameterBeam = VerticalCorrector(ParameterBeam)
VerticalCorrector_ParticleBeam = VerticalCorrector(ParticleBeam)

Cavity_ParameterBeam = Cavity(ParameterBeam)
Cavity_ParticleBeam = Cavity(ParticleBeam)

BPM_ParameterBeam = BPM(ParameterBeam)
BPM_ParticleBeam = BPM(ParticleBeam)

Screen_ParameterBeam = Screen(ParameterBeam)
Screen_ParticleBeam = Screen(ParticleBeam)

Undulator_ParameterBeam = Undulator(ParameterBeam)
Undulator_ParticleBeam = Undulator(ParticleBeam)

# Expected Results
Quadrupole_ParticleBeam_n = 100000
Quadrupole_ParameterBeam_Energy = 107315902.44355084
Quadrupole_ParticleBeam_Energy = 107315902.44355084
Quadrupole_ParameterBeam_mu = torch.tensor(
    [3.9735e-02, 3.9469e00, -4.0269e-02, -4.0536e00, 5.7248e-06, 3.8292e-04, 1.0000e00]
)
Quadrupole_ParticleBeam_mu = torch.tensor(
    [3.9735e-02, 3.9469e00, -4.0269e-02, -4.0536e00, 5.7248e-06, 3.8292e-04, 1.0000e00]
)
Quadrupole_ParameterBeam_cov = torch.tensor(
    [
        [
            2.8228e-08,
            -1.1546e-07,
            8.0754e-13,
            3.3509e-12,
            -6.1484e-13,
            -7.3875e-12,
            0.0000e00,
        ],
        [
            -1.1546e-07,
            4.7231e-07,
            -3.2122e-12,
            -1.3350e-11,
            2.4887e-12,
            3.0280e-11,
            0.0000e00,
        ],
        [
            8.0754e-13,
            -3.2122e-12,
            3.3239e-08,
            1.3008e-07,
            6.2988e-13,
            3.5424e-11,
            0.0000e00,
        ],
        [
            3.3509e-12,
            -1.3350e-11,
            1.3008e-07,
            5.0908e-07,
            2.5164e-12,
            1.4319e-10,
            0.0000e00,
        ],
        [
            -6.1484e-13,
            2.4887e-12,
            6.2988e-13,
            2.5164e-12,
            6.4182e-11,
            3.0016e-09,
            0.0000e00,
        ],
        [
            -7.3875e-12,
            3.0280e-11,
            3.5424e-11,
            1.4319e-10,
            3.0016e-09,
            5.2005e-06,
            0.0000e00,
        ],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
Quadrupole_ParticleBeam_cov = [
    [
        2.82278032e-08,
        -1.15464668e-07,
        8.08104574e-13,
        3.36715289e-12,
        -6.14818644e-13,
        -7.38039434e-12,
        0.00000000e00,
    ],
    [
        -1.15464668e-07,
        4.72307166e-07,
        -3.25763334e-12,
        -1.35833965e-11,
        2.48985351e-12,
        2.99345454e-11,
        0.00000000e00,
    ],
    [
        8.08104574e-13,
        -3.25763334e-12,
        3.32387494e-08,
        1.30080308e-07,
        6.29891413e-13,
        3.54257524e-11,
        0.00000000e00,
    ],
    [
        3.36715289e-12,
        -1.35833965e-11,
        1.30080308e-07,
        5.09074253e-07,
        2.52182982e-12,
        1.42098403e-10,
        0.00000000e00,
    ],
    [
        -6.14818644e-13,
        2.48985351e-12,
        6.29891413e-13,
        2.52182982e-12,
        6.41822474e-11,
        3.00164415e-09,
        0.00000000e00,
    ],
    [
        -7.38039434e-12,
        2.99345454e-11,
        3.54257524e-11,
        1.42098403e-10,
        3.00164415e-09,
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
HorizontalCorrector_ParticleBeam_n = 100000
HorizontalCorrector_ParameterBeam_Energy = 107315902.44355084
HorizontalCorrector_ParticleBeam_Energy = 107315902.44355084
HorizontalCorrector_ParameterBeam_mu = torch.tensor(
    [
        8.2532e-07,
        2.0001e-03,
        -1.7300e-06,
        -1.1746e-07,
        5.7250e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
HorizontalCorrector_ParticleBeam_mu = torch.tensor(
    [
        8.2532e-07,
        2.0001e-03,
        -1.7300e-06,
        -1.1746e-07,
        5.7250e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
HorizontalCorrector_ParameterBeam_cov = torch.tensor(
    [
        [
            3.0612e-08,
            5.5706e-10,
            8.0846e-13,
            2.2124e-13,
            -6.4027e-13,
            -7.6932e-12,
            0.0000e00,
        ],
        [
            5.5706e-10,
            1.3538e-11,
            9.8773e-14,
            6.4855e-15,
            -3.6896e-14,
            -8.0708e-14,
            0.0000e00,
        ],
        [
            8.0846e-13,
            9.8773e-14,
            3.0716e-08,
            5.6103e-10,
            6.0554e-13,
            3.4055e-11,
            0.0000e00,
        ],
        [
            2.2124e-13,
            6.4855e-15,
            5.6103e-10,
            1.3646e-11,
            6.4452e-14,
            5.3652e-12,
            0.0000e00,
        ],
        [
            -6.4027e-13,
            -3.6896e-14,
            6.0554e-13,
            6.4452e-14,
            6.4185e-11,
            3.0040e-09,
            0.0000e00,
        ],
        [
            -7.6932e-12,
            -8.0708e-14,
            3.4055e-11,
            5.3652e-12,
            3.0040e-09,
            5.2005e-06,
            0.0000e00,
        ],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
HorizontalCorrector_ParticleBeam_cov = [
    [
        3.06115490e-08,
        5.57061327e-10,
        8.08462999e-13,
        2.21243689e-13,
        -6.40264932e-13,
        -7.69321934e-12,
        0.00000000e00,
    ],
    [
        5.57061327e-10,
        1.35379983e-11,
        9.87493703e-14,
        6.48437255e-15,
        -3.68937077e-14,
        -8.13532478e-14,
        0.00000000e00,
    ],
    [
        8.08462999e-13,
        9.87493703e-14,
        3.07158772e-08,
        5.61031499e-10,
        6.05542772e-13,
        3.40555441e-11,
        0.00000000e00,
    ],
    [
        2.21243689e-13,
        6.48437255e-15,
        5.61031499e-10,
        1.36463749e-11,
        6.44515124e-14,
        5.36516751e-12,
        0.00000000e00,
    ],
    [
        -6.40264932e-13,
        -3.68937077e-14,
        6.05542772e-13,
        6.44515124e-14,
        6.41849709e-11,
        3.00400242e-09,
        0.00000000e00,
    ],
    [
        -7.69321934e-12,
        -8.13532478e-14,
        3.40555441e-11,
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
VerticalCorrector_ParticleBeam_n = 100000
VerticalCorrector_ParameterBeam_Energy = 107315902.44355084
VerticalCorrector_ParticleBeam_Energy = 107315902.44355084
VerticalCorrector_ParameterBeam_mu = torch.tensor(
    [8.2532e-07, 5.9885e-08, -1.7300e-06, 1.9999e-03, 5.7250e-06, 3.8292e-04, 1.0000e00]
)
VerticalCorrector_ParticleBeam_mu = torch.tensor(
    [8.2532e-07, 5.9885e-08, -1.7300e-06, 1.9999e-03, 5.7250e-06, 3.8292e-04, 1.0000e00]
)
VerticalCorrector_ParameterBeam_cov = torch.tensor(
    [
        [
            3.0612e-08,
            5.5706e-10,
            8.0846e-13,
            2.2124e-13,
            -6.4027e-13,
            -7.6932e-12,
            0.0000e00,
        ],
        [
            5.5706e-10,
            1.3538e-11,
            9.8773e-14,
            6.4855e-15,
            -3.6896e-14,
            -8.0708e-14,
            0.0000e00,
        ],
        [
            8.0846e-13,
            9.8773e-14,
            3.0716e-08,
            5.6103e-10,
            6.0554e-13,
            3.4055e-11,
            0.0000e00,
        ],
        [
            2.2124e-13,
            6.4855e-15,
            5.6103e-10,
            1.3646e-11,
            6.4452e-14,
            5.3652e-12,
            0.0000e00,
        ],
        [
            -6.4027e-13,
            -3.6896e-14,
            6.0554e-13,
            6.4452e-14,
            6.4185e-11,
            3.0040e-09,
            0.0000e00,
        ],
        [
            -7.6932e-12,
            -8.0708e-14,
            3.4055e-11,
            5.3652e-12,
            3.0040e-09,
            5.2005e-06,
            0.0000e00,
        ],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
VerticalCorrector_ParticleBeam_cov = [
    [
        3.06115490e-08,
        5.57061383e-10,
        8.08462999e-13,
        2.21235109e-13,
        -6.40264932e-13,
        -7.69321934e-12,
        0.00000000e00,
    ],
    [
        5.57061383e-10,
        1.35380005e-11,
        9.87730674e-14,
        6.48450760e-15,
        -3.68956168e-14,
        -8.07087941e-14,
        0.00000000e00,
    ],
    [
        8.08462999e-13,
        9.87730674e-14,
        3.07158772e-08,
        5.61031529e-10,
        6.05542772e-13,
        3.40555441e-11,
        0.00000000e00,
    ],
    [
        2.21235109e-13,
        6.48450760e-15,
        5.61031529e-10,
        1.36463746e-11,
        6.44517278e-14,
        5.36540115e-12,
        0.00000000e00,
    ],
    [
        -6.40264932e-13,
        -3.68956168e-14,
        6.05542772e-13,
        6.44517278e-14,
        6.41849709e-11,
        3.00400242e-09,
        0.00000000e00,
    ],
    [
        -7.69321934e-12,
        -8.07087941e-14,
        3.40555441e-11,
        5.36540115e-12,
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
Cavity_ParticleBeam_n = 100000
Cavity_ParameterBeam_Energy = 107316902.44355084
Cavity_ParticleBeam_Energy = 107316902.44355084
Cavity_ParameterBeam_mu = torch.tensor(
    [
        8.2532e-07,
        5.9885e-08,
        -1.7300e-06,
        -1.1746e-07,
        5.7252e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
Cavity_ParticleBeam_mu = torch.tensor(
    [
        8.2532e-07,
        5.9885e-08,
        -1.7300e-06,
        -1.1746e-07,
        5.7252e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
Cavity_ParameterBeam_cov = torch.tensor(
    [
        [
            3.0612e-08,
            5.5706e-10,
            8.0846e-13,
            2.2124e-13,
            -6.4027e-13,
            -7.6932e-12,
            0.0000e00,
        ],
        [
            5.5706e-10,
            1.3538e-11,
            9.8773e-14,
            6.4855e-15,
            -3.6896e-14,
            -8.0708e-14,
            0.0000e00,
        ],
        [
            8.0846e-13,
            9.8773e-14,
            3.0716e-08,
            5.6103e-10,
            6.0556e-13,
            3.4055e-11,
            0.0000e00,
        ],
        [
            2.2124e-13,
            6.4855e-15,
            5.6103e-10,
            1.3646e-11,
            6.4454e-14,
            5.3652e-12,
            0.0000e00,
        ],
        [
            -6.4027e-13,
            -3.6896e-14,
            6.0556e-13,
            6.4454e-14,
            6.4188e-11,
            3.0064e-09,
            0.0000e00,
        ],
        [
            -7.6932e-12,
            -8.0708e-14,
            3.4055e-11,
            5.3652e-12,
            3.0064e-09,
            5.2005e-06,
            0.0000e00,
        ],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
Cavity_ParticleBeam_cov = [
    [
        3.06115490e-08,
        5.57061383e-10,
        8.08462999e-13,
        2.21243689e-13,
        -6.40268513e-13,
        -7.69321934e-12,
        0.00000000e00,
    ],
    [
        5.57061383e-10,
        1.35380005e-11,
        9.87730674e-14,
        6.48545528e-15,
        -3.68956562e-14,
        -8.07087941e-14,
        0.00000000e00,
    ],
    [
        8.08462999e-13,
        9.87730674e-14,
        3.07158772e-08,
        5.61031499e-10,
        6.05558103e-13,
        3.40555441e-11,
        0.00000000e00,
    ],
    [
        2.21243689e-13,
        6.48545528e-15,
        5.61031499e-10,
        1.36463749e-11,
        6.44539428e-14,
        5.36516751e-12,
        0.00000000e00,
    ],
    [
        -6.40268513e-13,
        -3.68956562e-14,
        6.05558103e-13,
        6.44539428e-14,
        6.41876963e-11,
        3.00636064e-09,
        0.00000000e00,
    ],
    [
        -7.69321934e-12,
        -8.07087941e-14,
        3.40555441e-11,
        5.36516751e-12,
        3.00636064e-09,
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
BPM_ParticleBeam_n = 100000
BPM_ParameterBeam_Energy = 107315902.44355084
BPM_ParticleBeam_Energy = 107315902.44355084
BPM_ParameterBeam_mu = torch.tensor(
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
BPM_ParticleBeam_mu = torch.tensor(
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
BPM_ParameterBeam_cov = torch.tensor(
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
BPM_ParticleBeam_cov = [
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
Screen_ParticleBeam_n = 100000
Screen_ParameterBeam_Energy = 107315902.44355084
Screen_ParticleBeam_Energy = 107315902.44355084
Screen_ParameterBeam_mu = torch.tensor(
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
Screen_ParticleBeam_mu = torch.tensor(
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
Screen_ParameterBeam_cov = torch.tensor(
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
Screen_ParticleBeam_cov = [
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
Undulator_ParticleBeam_n = 100000
Undulator_ParameterBeam_Energy = 107315902.44355084
Undulator_ParticleBeam_Energy = 107315902.44355084
Undulator_ParameterBeam_mu = torch.tensor(
    [
        8.2532e-07,
        5.9885e-08,
        -1.7300e-06,
        -1.1746e-07,
        5.7252e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
Undulator_ParticleBeam_mu = torch.tensor(
    [
        8.2532e-07,
        5.9885e-08,
        -1.7300e-06,
        -1.1746e-07,
        5.7252e-06,
        3.8292e-04,
        1.0000e00,
    ]
)
Undulator_ParameterBeam_cov = torch.tensor(
    [
        [
            3.0612e-08,
            5.5706e-10,
            8.0846e-13,
            2.2124e-13,
            -6.4027e-13,
            -7.6932e-12,
            0.0000e00,
        ],
        [
            5.5706e-10,
            1.3538e-11,
            9.8773e-14,
            6.4855e-15,
            -3.6896e-14,
            -8.0708e-14,
            0.0000e00,
        ],
        [
            8.0846e-13,
            9.8773e-14,
            3.0716e-08,
            5.6103e-10,
            6.0556e-13,
            3.4055e-11,
            0.0000e00,
        ],
        [
            2.2124e-13,
            6.4855e-15,
            5.6103e-10,
            1.3646e-11,
            6.4454e-14,
            5.3652e-12,
            0.0000e00,
        ],
        [
            -6.4027e-13,
            -3.6896e-14,
            6.0556e-13,
            6.4454e-14,
            6.4188e-11,
            3.0064e-09,
            0.0000e00,
        ],
        [
            -7.6932e-12,
            -8.0708e-14,
            3.4055e-11,
            5.3652e-12,
            3.0064e-09,
            5.2005e-06,
            0.0000e00,
        ],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
    ]
)
Undulator_ParticleBeam_cov = [
    [
        3.06115490e-08,
        5.57061383e-10,
        8.08462999e-13,
        2.21243689e-13,
        -6.40268513e-13,
        -7.69321934e-12,
        0.00000000e00,
    ],
    [
        5.57061383e-10,
        1.35380005e-11,
        9.87730674e-14,
        6.48545528e-15,
        -3.68956562e-14,
        -8.07087941e-14,
        0.00000000e00,
    ],
    [
        8.08462999e-13,
        9.87730674e-14,
        3.07158772e-08,
        5.61031499e-10,
        6.05558103e-13,
        3.40555441e-11,
        0.00000000e00,
    ],
    [
        2.21243689e-13,
        6.48545528e-15,
        5.61031499e-10,
        1.36463749e-11,
        6.44539428e-14,
        5.36516751e-12,
        0.00000000e00,
    ],
    [
        -6.40268513e-13,
        -3.68956562e-14,
        6.05558103e-13,
        6.44539428e-14,
        6.41876963e-11,
        3.00636064e-09,
        0.00000000e00,
    ],
    [
        -7.69321934e-12,
        -8.07087941e-14,
        3.40555441e-11,
        5.36516751e-12,
        3.00636064e-09,
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


# Tests
# Quadrupole
def test_Quadrupole_ParticleBeam_n():
    assert Quadrupole_ParticleBeam.n == Quadrupole_ParticleBeam_n


def test_Quadrupole_ParameterBeam_Energy():
    assert np.isclose(
        Quadrupole_ParameterBeam.energy,
        Quadrupole_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_Quadrupole_ParticleBeam_Energy():
    assert np.isclose(
        Quadrupole_ParticleBeam.energy,
        Quadrupole_ParticleBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_Quadrupole_ParameterBeam_mu():
    assert torch.allclose(
        Quadrupole_ParameterBeam._mu,
        Quadrupole_ParameterBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Quadrupole_ParticleBeam_mu():
    assert torch.allclose(
        Quadrupole_ParticleBeam.particles.mean(axis=0),
        Quadrupole_ParticleBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Quadrupole_ParameterBeam_Quadrupole_ParticleBeam_mu_dif():
    assert torch.allclose(
        Quadrupole_ParameterBeam._mu,
        Quadrupole_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Quadrupole_ParameterBeam_cov():
    assert torch.allclose(
        Quadrupole_ParameterBeam._cov,
        Quadrupole_ParameterBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_Quadrupole_ParticleBeam_cov():
    assert np.allclose(
        np.cov(Quadrupole_ParticleBeam.particles.t().numpy()),
        Quadrupole_ParticleBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_Quadrupole_ParameterBeam_Quadrupole_ParticleBeam_cov_dif():
    assert np.allclose(
        Quadrupole_ParameterBeam._cov,
        np.cov(Quadrupole_ParticleBeam.particles.t().numpy()),
        rtol=1e-1,
        atol=1e-16,
        equal_nan=False,
    )


# HorizontalCorrector
def test_HorizontalCorrector_ParticleBeam_n():
    assert HorizontalCorrector_ParticleBeam.n == HorizontalCorrector_ParticleBeam_n


def test_HorizontalCorrector_ParameterBeam_Energy():
    assert np.isclose(
        HorizontalCorrector_ParameterBeam.energy,
        HorizontalCorrector_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParticleBeam_Energy():
    assert np.isclose(
        HorizontalCorrector_ParticleBeam.energy,
        HorizontalCorrector_ParticleBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParameterBeam_mu():
    assert torch.allclose(
        HorizontalCorrector_ParameterBeam._mu,
        HorizontalCorrector_ParameterBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParticleBeam_mu():
    assert torch.allclose(
        HorizontalCorrector_ParticleBeam.particles.mean(axis=0),
        HorizontalCorrector_ParticleBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParameterBeam_HorizontalCorrector_ParticleBeam_mu_dif():
    assert torch.allclose(
        HorizontalCorrector_ParameterBeam._mu,
        HorizontalCorrector_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParameterBeam_cov():
    assert torch.allclose(
        HorizontalCorrector_ParameterBeam._cov,
        HorizontalCorrector_ParameterBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParticleBeam_cov():
    assert np.allclose(
        np.cov(HorizontalCorrector_ParticleBeam.particles.t().numpy()),
        HorizontalCorrector_ParticleBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_HorizontalCorrector_ParameterBeam_HorizontalCorrector_ParticleBeam_cov_dif():
    assert np.allclose(
        HorizontalCorrector_ParameterBeam._cov,
        np.cov(HorizontalCorrector_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-15,
        equal_nan=False,
    )


# VerticalCorrector
def test_VerticalCorrector_ParticleBeam_n():
    assert VerticalCorrector_ParticleBeam.n == VerticalCorrector_ParticleBeam_n


def test_VerticalCorrector_ParameterBeam_Energy():
    assert np.isclose(
        VerticalCorrector_ParameterBeam.energy,
        VerticalCorrector_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_VerticalCorrector_ParticleBeam_Energy():
    assert np.isclose(
        VerticalCorrector_ParticleBeam.energy,
        VerticalCorrector_ParticleBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_VerticalCorrector_ParameterBeam_mu():
    assert torch.allclose(
        VerticalCorrector_ParameterBeam._mu,
        VerticalCorrector_ParameterBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_VerticalCorrector_ParticleBeam_mu():
    assert torch.allclose(
        VerticalCorrector_ParticleBeam.particles.mean(axis=0),
        VerticalCorrector_ParticleBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_VerticalCorrector_ParameterBeam_VerticalCorrector_ParticleBeam_mu_dif():
    assert torch.allclose(
        VerticalCorrector_ParameterBeam._mu,
        VerticalCorrector_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_VerticalCorrector_ParameterBeam_cov():
    assert torch.allclose(
        VerticalCorrector_ParameterBeam._cov,
        VerticalCorrector_ParameterBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_VerticalCorrector_ParticleBeam_cov():
    assert np.allclose(
        np.cov(VerticalCorrector_ParticleBeam.particles.t().numpy()),
        VerticalCorrector_ParticleBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_VerticalCorrector_ParameterBeam_VerticalCorrector_ParticleBeam_cov_dif():
    assert np.allclose(
        VerticalCorrector_ParameterBeam._cov,
        np.cov(VerticalCorrector_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


# Cavity
def test_Cavity_ParticleBeam_n():
    assert Cavity_ParticleBeam.n == Cavity_ParticleBeam_n


def test_Cavity_ParameterBeam_Energy():
    assert np.isclose(
        Cavity_ParameterBeam.energy,
        Cavity_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_Cavity_ParticleBeam_Energy():
    assert np.isclose(
        Cavity_ParticleBeam.energy,
        Cavity_ParticleBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_Cavity_ParameterBeam_mu():
    assert torch.allclose(
        Cavity_ParameterBeam._mu,
        Cavity_ParameterBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Cavity_ParticleBeam_mu():
    assert torch.allclose(
        Cavity_ParticleBeam.particles.mean(axis=0),
        Cavity_ParticleBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Cavity_ParameterBeam_Cavity_ParticleBeam_mu_dif():
    assert torch.allclose(
        Cavity_ParameterBeam._mu,
        Cavity_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Cavity_ParameterBeam_cov():
    assert torch.allclose(
        Cavity_ParameterBeam._cov,
        Cavity_ParameterBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_Cavity_ParticleBeam_cov():
    assert np.allclose(
        np.cov(Cavity_ParticleBeam.particles.t().numpy()),
        Cavity_ParticleBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_Cavity_ParameterBeam_Cavity_ParticleBeam_cov_dif():
    assert np.allclose(
        Cavity_ParameterBeam._cov,
        np.cov(Cavity_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


# BPM
def test_BPM_ParticleBeam_n():
    assert BPM_ParticleBeam.n == BPM_ParticleBeam_n


def test_BPM_ParameterBeam_Energy():
    assert np.isclose(
        BPM_ParameterBeam.energy,
        BPM_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_BPM_ParticleBeam_Energy():
    assert np.isclose(
        BPM_ParticleBeam.energy,
        BPM_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_BPM_ParameterBeam_mu():
    assert torch.allclose(
        BPM_ParameterBeam._mu,
        BPM_ParameterBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_BPM_ParticleBeam_mu():
    assert torch.allclose(
        BPM_ParticleBeam.particles.mean(axis=0),
        BPM_ParticleBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_BPM_ParameterBeam_BPM_ParticleBeam_mu_dif():
    assert torch.allclose(
        BPM_ParameterBeam._mu,
        BPM_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_BPM_ParameterBeam_cov():
    assert torch.allclose(
        BPM_ParameterBeam._cov,
        BPM_ParameterBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_BPM_ParticleBeam_cov():
    assert np.allclose(
        np.cov(BPM_ParticleBeam.particles.t().numpy()),
        BPM_ParticleBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_BPM_ParameterBeam_BPM_ParticleBeam_cov_dif():
    assert np.allclose(
        BPM_ParameterBeam._cov,
        np.cov(BPM_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


# Screen
def test_Screen_ParticleBeam_n():
    assert Screen_ParticleBeam.n == Screen_ParticleBeam_n


def test_Screen_ParameterBeam_Energy():
    assert np.isclose(
        Screen_ParameterBeam.energy,
        Screen_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_Screen_ParticleBeam_Energy():
    assert np.isclose(
        Screen_ParticleBeam.energy,
        Screen_ParticleBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_Screen_ParameterBeam_mu():
    assert torch.allclose(
        Screen_ParameterBeam._mu,
        Screen_ParameterBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Screen_ParticleBeam_mu():
    assert torch.allclose(
        Screen_ParticleBeam.particles.mean(axis=0),
        Screen_ParticleBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Screen_ParameterBeam_Screen_ParticleBeam_mu_dif():
    assert torch.allclose(
        Screen_ParameterBeam._mu,
        Screen_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Screen_ParameterBeam_cov():
    assert torch.allclose(
        Screen_ParameterBeam._cov,
        Screen_ParameterBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_Screen_ParticleBeam_cov():
    assert np.allclose(
        np.cov(Screen_ParticleBeam.particles.t().numpy()),
        Screen_ParticleBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_Screen_ParameterBeam_Screen_ParticleBeam_cov_dif():
    assert np.allclose(
        Screen_ParameterBeam._cov,
        np.cov(Screen_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


# Undulator
def test_Undulator_ParticleBeam_n():
    assert Undulator_ParticleBeam.n == Undulator_ParticleBeam_n


def test_Undulator_ParameterBeam_Energy():
    assert np.isclose(
        Undulator_ParameterBeam.energy,
        Undulator_ParameterBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_Undulator_ParticleBeam_Energy():
    assert np.isclose(
        Undulator_ParticleBeam.energy,
        Undulator_ParticleBeam_Energy,
        rtol=1e-4,
        atol=1e-8,
        equal_nan=False,
    )


def test_Undulator_ParameterBeam_mu():
    assert torch.allclose(
        Undulator_ParameterBeam._mu,
        Undulator_ParameterBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Undulator_ParticleBeam_mu():
    assert torch.allclose(
        Undulator_ParticleBeam.particles.mean(axis=0),
        Undulator_ParticleBeam_mu,
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Undulator_ParameterBeam_Undulator_ParticleBeam_mu_dif():
    assert torch.allclose(
        Undulator_ParameterBeam._mu,
        Undulator_ParticleBeam.particles.mean(axis=0),
        rtol=1e-4,
        atol=1e-9,
        equal_nan=False,
    )


def test_Undulator_ParameterBeam_cov():
    assert torch.allclose(
        Undulator_ParameterBeam._cov,
        Undulator_ParameterBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_Undulator_ParticleBeam_cov():
    assert np.allclose(
        np.cov(Undulator_ParticleBeam.particles.t().numpy()),
        Undulator_ParticleBeam_cov,
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )


def test_Undulator_ParameterBeam_Undulator_ParticleBeam_cov_dif():
    assert np.allclose(
        Undulator_ParameterBeam._cov,
        np.cov(Undulator_ParticleBeam.particles.t().numpy()),
        rtol=1e-4,
        atol=1e-16,
        equal_nan=False,
    )
