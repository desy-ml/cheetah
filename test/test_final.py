import test.ARESlatticeStage3v1_9 as ares

import numpy as np
import torch

from cheetah import (
    BPM,
    Cavity,
    Drift,
    HorizontalCorrector,
    ParameterBeam,
    ParticleBeam,
    Quadrupole,
    Screen,
    Segment,
    VerticalCorrector,
)

"""
Test Beam, which can be found in GitHub in the folder
benchmark/cheetah/ACHIP_EA1_2021.1351.001
"""


ParameterBeam = ParameterBeam.from_astra("benchmark/cheetah/ACHIP_EA1_2021.1351.001")
ParticleBeam = ParticleBeam.from_astra("benchmark/cheetah/ACHIP_EA1_2021.1351.001")


segment = Segment.from_ocelot(ares.cell)

FinalTestResult_ParameterBeam = segment(ParameterBeam)
FinalTestResult_ParticleBeam = segment(ParticleBeam)


def test_import():
    assert str(segment) == str(
        Segment(
            [
                Drift(length=0.00, name="ARLISOLG1"),
                Drift(length=0.20, name="Drift_ARLISOLG1"),
                Drift(length=0.09, name="ARLIMSOG1"),
                Drift(length=0.09, name="ARLIMSOG1"),
                Drift(length=0.17, name="Drift_ARLIMSOG1p"),
                HorizontalCorrector(length=0.00, angle=0.0, name="ARLIMCXG1"),
                VerticalCorrector(length=0.00, angle=0.0, name="ARLIMCXG1"),
                Drift(length=0.19, name="Drift_ARLIMCVG1"),
                Drift(length=0.00, name="ARLIBSCL1"),
                Drift(length=0.14, name="Drift_ARLIBSCL1"),
                Drift(length=0.00, name="ARLIBAML1"),
                Drift(length=0.30, name="Drift_ARLIBAML1"),
                Drift(length=0.00, name="ARLIBSCX1"),
                Drift(length=0.19, name="Drift_ARLIBSCX1"),
                Drift(length=0.00, name="ARLISLHG1"),
                Drift(length=0.49, name="Drift_ARLISLHG1"),
                HorizontalCorrector(length=0.00, angle=0.0, name="ARLIMCXG2"),
                VerticalCorrector(length=0.00, angle=0.0, name="ARLIMCXG2"),
                Drift(length=0.06, name="Drift_ARLIMCVG2"),
                Drift(length=0.00, name="ARLIBCMG1"),
                Drift(length=0.33, name="Drift_ARLIBCMG1"),
                Screen(
                    resolution=(2448, 2040),
                    pixel_size=(3.5488e-06, 2.5003e-06),
                    binning=1,
                    misalignment=(0, 0),
                    name="ARLIBSCR1",
                ),
                Drift(length=0.15, name="Drift_ARLIBSCR1"),
                Drift(length=0.00, name="ARLIEOLG1"),
                Drift(length=0.00, name="ARLISOLS1"),
                Cavity(length=4.14, delta_energy=0, name="ARLIRSBL1"),
                Drift(length=0.02, name="Drift_ARLIRSBL1"),
                HorizontalCorrector(length=0.00, angle=0.0, name="ARLIMCXG3"),
                VerticalCorrector(length=0.00, angle=0.0, name="ARLIMCXG3"),
                Drift(length=0.30, name="Drift_ARLIMCVG3"),
                Screen(
                    resolution=(2448, 2040),
                    pixel_size=(3.5488e-06, 2.5003e-06),
                    binning=1,
                    misalignment=(0, 0),
                    name="ARLIBSCR2",
                ),
                Drift(length=0.19, name="Drift_ARLIBSCR2"),
                HorizontalCorrector(length=0.02, angle=0.0, name="ARLIMCHM1"),
                Drift(length=0.41, name="Drift_ARLIMCHM1"),
                BPM(name="ARLIBPMG1"),
                Drift(length=0.31, name="Drift_ARLIBPMG1"),
                VerticalCorrector(length=0.02, angle=0.0, name="ARLIMCVM1"),
                Drift(length=0.09, name="Drift_ARLIMCVM1"),
                Cavity(length=4.14, delta_energy=0, name="ARLIRSBL2"),
                Drift(length=0.02, name="Drift_ARLIRSBL2"),
                HorizontalCorrector(length=0.00, angle=0.0, name="ARLIMCXG4A"),
                VerticalCorrector(length=0.00, angle=0.0, name="ARLIMCXG4B"),
                Drift(length=0.30, name="Drift_ARLIMCVG4"),
                Screen(
                    resolution=(2448, 2040),
                    pixel_size=(3.5488e-06, 2.5003e-06),
                    binning=1,
                    misalignment=(0, 0),
                    name="ARLIBSCR3",
                ),
                Drift(length=0.19, name="Drift_ARLIBSCR3"),
                HorizontalCorrector(length=0.02, angle=0.0, name="ARLIMCHM2"),
                Drift(length=0.41, name="Drift_ARLIMCHM1"),
                BPM(name="ARLIBPMG2"),
                Drift(length=0.31, name="Drift_ARLIBPMG1"),
                VerticalCorrector(length=0.02, angle=0.0, name="ARLIMCVM2"),
                Drift(length=0.09, name="Drift_ARLIMCVM2"),
                Drift(length=0.00, name="ARLIEOLS1"),
                Drift(length=0.00, name="AREASOLA1"),
                Drift(length=0.18, name="Drift_AREASOLA1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="AREAMQZM1"),
                Drift(length=0.43, name="Drift_AREAMQZM1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="AREAMQZM2"),
                Drift(length=0.20, name="Drift_AREAMQZM2"),
                VerticalCorrector(length=0.02, angle=0.0, name="AREAMCVM1"),
                Drift(length=0.20, name="Drift_AREAMCVM1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="AREAMQZM3"),
                Drift(length=0.18, name="Drift_AREAMQZM3"),
                HorizontalCorrector(length=0.02, angle=0.0, name="AREAMCHM1"),
                Drift(length=0.45, name="Drift_AREAMCHM1"),
                Screen(
                    resolution=(2448, 2040),
                    pixel_size=(3.5488e-06, 2.5003e-06),
                    binning=1,
                    misalignment=(0, 0),
                    name="AREABSCR1",
                ),
                Drift(length=1.38, name="Drift_AREABSCR1"),
                Drift(length=0.00, name="AREAECHA1"),
                Drift(length=0.37, name="Drift_AREAECHA1"),
                HorizontalCorrector(length=0.02, angle=0.0, name="AREAMCHM2"),
                Drift(length=0.23, name="Drift_AREAMCHM2"),
                VerticalCorrector(length=0.02, angle=0.0, name="AREAMCVM2"),
                Drift(length=0.07, name="Drift_AREAMCVM2"),
                Drift(length=0.00, name="AREAEOLA1"),
                Drift(length=0.00, name="ARMRSOLT1"),
                Drift(length=0.17, name="Drift_ARMRSOLT1"),
                HorizontalCorrector(length=0.02, angle=0.0, name="ARMRMCHM1"),
                Drift(length=0.37, name="Drift_ARMRMCHM1"),
                VerticalCorrector(length=0.02, angle=0.0, name="ARMRMCVM1"),
                Drift(length=0.37, name="Drift_ARMRMCVM1"),
                BPM(name="ARMRBPMG1"),
                Drift(length=0.33, name="Drift_ARMRBPMG1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARMRMQZM1"),
                Drift(length=0.25, name="Drift_ARMRMQZM1"),
                VerticalCorrector(length=0.02, angle=0.0, name="ARMRMCVM2"),
                Drift(length=0.36, name="Drift_ARMRMCVM2"),
                HorizontalCorrector(length=0.02, angle=0.0, name="ARMRMCHM2"),
                Drift(length=0.27, name="Drift_ARMRMCHM2"),
                Screen(
                    resolution=(2448, 2040),
                    pixel_size=(3.5488e-06, 2.5003e-06),
                    binning=1,
                    misalignment=(0, 0),
                    name="ARMRBSCR1",
                ),
                Drift(length=0.18, name="Drift_ARMRBSCR1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARMRMQZM2"),
                Drift(length=0.25, name="Drift_ARMRMQZM1"),
                HorizontalCorrector(length=0.02, angle=0.0, name="ARMRMCHM3"),
                Drift(length=0.15, name="Drift_ARMRMCHM3"),
                Drift(length=0.00, name="ARMRBCMG1"),
                Drift(length=0.15, name="Drift_ARMRBCMG1"),
                VerticalCorrector(length=0.02, angle=0.0, name="ARMRMCVM3"),
                Drift(length=0.31, name="Drift_ARMRMCVM3"),
                BPM(name="ARMRBPMG2"),
                Drift(length=0.33, name="Drift_ARMRBPMG1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARMRMQZM3"),
                Drift(length=0.22, name="Drift_ARMRMQZM3"),
                Drift(length=0.00, name="ARMRBAML1"),
                Drift(length=0.14, name="Drift_ARMRBAML1"),
                VerticalCorrector(length=0.02, angle=0.0, name="ARMRMCVM4"),
                Drift(length=0.16, name="Drift_ARMRMCVM4"),
                Drift(length=0.00, name="ARMRTORF1"),
                Drift(length=0.13, name="Drift_ARMRTORF1"),
                HorizontalCorrector(length=0.02, angle=0.0, name="ARMRMCHM4"),
                Drift(length=0.27, name="Drift_ARMRMCHM2"),
                Screen(
                    resolution=(2448, 2040),
                    pixel_size=(3.5488e-06, 2.5003e-06),
                    binning=1,
                    misalignment=(0, 0),
                    name="ARMRBSCR2",
                ),
                Drift(length=0.18, name="Drift_ARMRBSCR1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARMRMQZM4"),
                Drift(length=0.15, name="Drift_ARMRMQZM4"),
                Drift(length=0.00, name="ARMREOLT1"),
                Drift(length=0.00, name="ARDGSOLO1"),
                Drift(length=1.55, name="Drift_ARDGSOLO1"),
                Drift(length=0.00, name="ARDGEOLO1"),
                Drift(length=0.00, name="ARMRSOLB1"),
                Drift(length=0.23, name="Drift_ARMRSOLB1"),
                BPM(name="ARMRBPMG3"),
                Drift(length=0.38, name="Drift_ARMRBPMG3"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARMRMQZM5"),
                Drift(length=0.25, name="Drift_ARMRMQZM5"),
                VerticalCorrector(length=0.02, angle=0.0, name="ARMRMCVM5"),
                Drift(length=0.28, name="Drift_ARMRMCVM5"),
                HorizontalCorrector(length=0.02, angle=0.0, name="ARMRMCHM5"),
                Drift(length=0.27, name="Drift_ARMRMCHM2"),
                Screen(
                    resolution=(2448, 2040),
                    pixel_size=(3.5488e-06, 2.5003e-06),
                    binning=1,
                    misalignment=(0, 0),
                    name="ARMRBSCR3",
                ),
                Drift(length=0.18, name="Drift_ARMRBSCR1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARMRMQZM6"),
                Drift(length=0.32, name="Drift_ARMRMQZM6"),
                Drift(length=0.00, name="ARMREOLB1"),
                Drift(length=0.00, name="ARBCSOLC"),
                Drift(length=0.27, name="Drift_ARBCSOLC"),
                Drift(length=0.22, name="ARBCMBHB1"),
                Drift(length=0.60, name="Drift_ARBCMBHB1"),
                Drift(length=0.22, name="ARBCMBHB2"),
                Drift(length=0.56, name="Drift_ARBCMBHB2"),
                BPM(name="ARBCBPML1"),
                Drift(length=0.60, name="Drift_ARBCBPML1"),
                Drift(length=0.00, name="ARBCSLHB1"),
                Drift(length=0.38, name="Drift_ARBCSLHB1"),
                Drift(length=0.00, name="ARBCSLHS1"),
                Drift(length=0.61, name="Drift_ARBCSLHS1"),
                Drift(length=0.00, name="ARBCBSCE1"),
                Drift(length=0.55, name="Drift_ARBCBSCE1"),
                Drift(length=0.22, name="ARBCMBHB3"),
                Drift(length=0.60, name="Drift_ARBCMBHB1"),
                Drift(length=0.22, name="ARBCMBHB4"),
                Drift(length=0.26, name="Drift_ARBCMBHB4"),
                Drift(length=0.00, name="ARBCEOLC"),
                Drift(length=0.00, name="ARDLSOLM1"),
                Drift(length=0.20, name="Drift_ARDLSOLM1"),
                VerticalCorrector(length=0.02, angle=0.0, name="ARDLMCVM1"),
                Drift(length=0.16, name="Drift_ARDLMCVM1"),
                Drift(length=0.00, name="ARDLTORF1"),
                Drift(length=0.13, name="Drift_ARDLTORF1"),
                HorizontalCorrector(length=0.02, angle=0.0, name="ARDLMCHM1"),
                Drift(length=0.25, name="Drift_ARMRMQZM1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARDLMQZM1"),
                Drift(length=0.36, name="Drift_ARDLMQZM1"),
                BPM(name="ARDLBPMG1"),
                Drift(length=0.33, name="Drift_ARDLBPMG1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARDLMQZM2"),
                Drift(length=0.34, name="Drift_ARDLMQZM2"),
                Screen(
                    resolution=(2448, 2040),
                    pixel_size=(3.5488e-06, 2.5003e-06),
                    binning=1,
                    misalignment=(0, 0),
                    name="ARDLBSCR1",
                ),
                Drift(length=0.53, name="Drift_ARDLBSCR1"),
                Drift(length=1.00, name="ARDLRXBD1"),
                Drift(length=0.09, name="Drift_ARDLRXBD1"),
                Drift(length=1.00, name="ARDLRXBD2"),
                Drift(length=0.65, name="Drift_ARDLRXBD2"),
                Drift(length=0.00, name="ARDLBSCE1"),
                Drift(length=0.54, name="Drift_ARDLBSCE1"),
                BPM(name="ARDLBPMG2"),
                Drift(length=0.47, name="Drift_ARDLBPMG2"),
                VerticalCorrector(length=0.02, angle=0.0, name="ARDLMCVM2"),
                Drift(length=0.25, name="Drift_ARMRMQZM1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARDLMQZM3"),
                Drift(length=0.61, name="Drift_ARDLMQZM3"),
                HorizontalCorrector(length=0.02, angle=0.0, name="ARDLMCHM2"),
                Drift(length=0.25, name="Drift_ARMRMQZM1"),
                Quadrupole(length=0.12, k1=0, misalignment=(0, 0), name="ARDLMQZM4"),
                Drift(length=0.15, name="Drift_ARDLMQZM4"),
                Drift(length=0.00, name="ARDLEOLM1"),
                Drift(length=0.00, name="ARSHSOLH1"),
                Drift(length=0.87, name="Drift_ARSHSOLH1"),
                Drift(length=0.44, name="ARSHMBHO1"),
                Drift(length=1.11, name="Drift_ARSHMBHO1"),
                Drift(length=0.00, name="ARSHBSCE2"),
                Drift(length=-0.87, name="Drift_ARSHBSCE2"),
                Drift(length=0.00, name="ARSHBSCE1"),
                Drift(length=0.04, name="Drift_ARSHBSCE1"),
                Drift(length=0.00, name="ARSHEOLH1"),
                Drift(length=0.89, name="Drift_ARSHEOLH1"),
                Drift(length=0.00, name="ARSHEOLH2"),
            ]
        )
    )


def test_FinalTestResult_ParticleBeam_n():
    assert FinalTestResult_ParticleBeam.n == 100000


def test_FinalTestResult_ParameterBeam_energy():
    actual = FinalTestResult_ParameterBeam.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_FinalTestResult_ParticleBeam_energy():
    actual = FinalTestResult_ParticleBeam.energy
    expected = 107315902.44355084
    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False)


def test_FinalTestResult_ParameterBeam_mu():
    actual = FinalTestResult_ParameterBeam._mu
    expected = torch.tensor(
        [
            3.3602e-06,
            5.9885e-08,
            -6.7022e-06,
            -1.1746e-07,
            6.0613e-06,
            3.8292e-04,
            1.0000e00,
        ]
    )
    assert np.allclose(actual, expected, rtol=1e-3, atol=1e-8, equal_nan=False)


def test_FinalTestResult_ParticleBeam_particles_mean():
    actual = FinalTestResult_ParticleBeam.particles.mean(axis=0)
    expected = torch.tensor(
        [
            3.3602e-06,
            5.9885e-08,
            -6.7022e-06,
            -1.1746e-07,
            6.0613e-06,
            3.8292e-04,
            1.0000e00,
        ]
    )
    assert np.allclose(actual, expected, rtol=1e-3, atol=1e-8, equal_nan=False)


def test_FinalTestResult_ParameterBeam_FinalTestResult_ParticleBeam_mu_dif():
    assert torch.allclose(
        FinalTestResult_ParameterBeam._mu,
        FinalTestResult_ParticleBeam.particles.mean(axis=0),
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_FinalTestResult_ParameterBeam_cov():
    actual = FinalTestResult_ParameterBeam._cov
    expected = torch.tensor(
        [
            [
                1.0203e-07,
                1.1301e-09,
                2.5975e-11,
                4.9577e-13,
                -2.2118e-12,
                -1.1110e-11,
                0.0000e00,
            ],
            [
                1.1301e-09,
                1.3538e-11,
                3.7330e-13,
                6.4855e-15,
                -3.6967e-14,
                -8.0708e-14,
                0.0000e00,
            ],
            [
                2.5975e-11,
                3.7330e-13,
                1.0266e-07,
                1.1387e-09,
                3.5631e-12,
                2.6116e-10,
                0.0000e00,
            ],
            [
                4.9577e-13,
                6.4855e-15,
                1.1387e-09,
                1.3646e-11,
                6.9164e-14,
                5.3652e-12,
                0.0000e00,
            ],
            [
                -2.2118e-12,
                -3.6967e-14,
                3.5631e-12,
                6.9164e-14,
                7.3473e-11,
                7.5715e-09,
                0.0000e00,
            ],
            [
                -1.1110e-11,
                -8.0708e-14,
                2.6116e-10,
                5.3652e-12,
                7.5715e-09,
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


def test_FinalTestResult_ParticleBeam_cov():
    actual = np.cov(FinalTestResult_ParticleBeam.particles.t().numpy())
    expected = np.array(
        [
            [
                1.02029056e-07,
                1.13011842e-09,
                2.59751889e-11,
                4.95770046e-13,
                -2.21179593e-12,
                -1.11094675e-11,
                0.00000000e00,
            ],
            [
                1.13011842e-09,
                1.35380005e-11,
                3.73299243e-13,
                6.48545528e-15,
                -3.69664988e-14,
                -8.07087941e-14,
                0.00000000e00,
            ],
            [
                2.59751889e-11,
                3.73299243e-13,
                1.02663675e-07,
                1.13867597e-09,
                3.56311933e-12,
                2.61160482e-10,
                0.00000000e00,
            ],
            [
                4.95770046e-13,
                6.48545528e-15,
                1.13867597e-09,
                1.36463749e-11,
                6.91636914e-14,
                5.36516751e-12,
                0.00000000e00,
            ],
            [
                -2.21179593e-12,
                -3.69664988e-14,
                3.56311933e-12,
                6.91636914e-14,
                7.34733682e-11,
                7.57152810e-09,
                0.00000000e00,
            ],
            [
                -1.11094675e-11,
                -8.07087941e-14,
                2.61160482e-10,
                5.36516751e-12,
                7.57152810e-09,
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


def test_FinalTestResult_ParameterBeam_FinalTestResult_ParticleBeam_cov_dif():
    assert np.allclose(
        FinalTestResult_ParameterBeam._cov,
        np.cov(FinalTestResult_ParticleBeam.particles.t().numpy()),
        rtol=1e-3,
        atol=1e-8,
        equal_nan=True,
    )
