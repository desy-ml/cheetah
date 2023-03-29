import cheetah
import numpy as np
import torch
import sys
sys.path.append('c:/users/ftheilen/appdata/local/packages/pythonsoftwarefoundation.python.3.10_qbz5n2kfra8p0/localcache/local-packages/python310/site-packages')

beam1 = cheetah.ParameterBeam.from_astra("H:/Source/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001")
beam2 = cheetah.ParticleBeam.from_astra("H:/Source/cheetah/benchmark/cheetah/ACHIP_EA1_2021.1351.001")

def test_import_ParameterBeam():
    assert str(beam1) == "ParameterBeam(mu_x=0.000001, mu_xp=0.000000, mu_y=-0.000002, mu_yp=-0.000000, sigma_x=0.000175, sigma_xp=0.000004, sigma_y=0.000175, sigma_yp=0.000004, sigma_s=0.000008, sigma_p=0.002280, energy=107315902.444)"

def test_import_ParticleBeam():
    assert str(beam2) == "ParticleBeam(n=100000, mu_x=0.000001, mu_xp=0.000000, mu_y=-0.000002, mu_yp=-0.000000, sigma_x=0.000175, sigma_xp=0.000004, sigma_y=0.000175, sigma_yp=0.000004, sigma_s=0.000008, sigma_p=0.002280, energy=107315902.444)"

def test_beam1_mu():
    assert str(beam1._mu) == str(torch.tensor([ 8.2413e-07,  5.9885e-08, -1.7276e-06, -1.1746e-07,  5.7250e-06, 3.8292e-04,  1.0000e+00], dtype=torch.float32))

def test_beam2_particles_mean():
    assert str(beam2.particles.mean(axis=0)) == str(torch.tensor([ 8.2413e-07,  5.9885e-08, -1.7276e-06, -1.1746e-07,  5.7250e-06, 3.8292e-04,  1.0000e+00], dtype=torch.float32))

def test_division_beam1_beam2():
    actual = beam1._mu / beam2.particles.mean(axis=0)
    expected = torch.tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8)

def test_division_beam2_beam1():
    actual = beam2.particles.mean(axis=0) / beam1._mu
    expected = torch.tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8)

def test_division_cov_beam1_beam2():
    assert str(beam1._cov / np.cov(beam2.particles.t().numpy())) == 'tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [   nan,    nan,    nan,    nan,    nan,    nan,    nan]],\n' \
                                                                    '       dtype=torch.float64)'

def test_division_cov_beam2_beam1():
    assert str(np.cov(beam2.particles.t().numpy()) / beam1._cov) == 'tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n' \
                                                                    '        [   nan,    nan,    nan,    nan,    nan,    nan,    nan]],\n' \
                                                                    '       dtype=torch.float64)'

segment = cheetah.Segment([
    cheetah.HorizontalCorrector(length=0.02, name="quad"),
    cheetah.Drift(length=2.0)
])
segment.quad.angle = 2e-3

result1 = segment(beam1)
result2 = segment(beam2)

def test_segment_beam1():
    assert str(result1) == "ParameterBeam(mu_x=0.004001, mu_xp=0.002000, mu_y=-0.000002, mu_yp=-0.000000, sigma_x=0.000181, sigma_xp=0.000004, sigma_y=0.000182, sigma_yp=0.000004, sigma_s=0.000008, sigma_p=0.002280, energy=107315902.444)"

def test_segment_beam2():
    assert str(result2) == "ParticleBeam(n=100000, mu_x=0.004001, mu_xp=0.002000, mu_y=-0.000002, mu_yp=-0.000000, sigma_x=0.000181, sigma_xp=0.000004, sigma_y=0.000182, sigma_yp=0.000004, sigma_s=0.000008, sigma_p=0.002280, energy=107315902.444)"

def test_result1_mu():
    assert str(result1._mu) == str(torch.tensor([ 4.0009e-03,  2.0001e-03, -1.9649e-06, -1.1746e-07,  5.7424e-06, 3.8292e-04,  1.0000e+00]))

def test_result2_particles_mean():
    assert str(result2.particles.mean(axis=0)) == str(torch.tensor([ 4.0009e-03,  2.0001e-03, -1.9649e-06, -1.1746e-07,  5.7424e-06, 3.8292e-04,  1.0000e+00]))

def test_division_result1_result2():
    actual = result1._mu / result2.particles.mean(axis=0)
    expected = torch.tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8)

def test_division_result2_result1():
    actual = result2.particles.mean(axis=0) / result1._mu
    expected = torch.tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-8)

def test_result1_cov():
    assert str(result1._cov) == str(torch.tensor([[ 3.2894e-08,  5.8414e-10,  1.4744e-12,  2.3421e-13, -7.1441e-13, -7.8546e-12,  0.0000e+00],
        [ 5.8414e-10,  1.3538e-11,  1.1174e-13,  6.4855e-15, -3.6899e-14, -8.0708e-14,  0.0000e+00],
        [ 1.4744e-12,  1.1174e-13,  3.3015e-08,  5.8832e-10,  7.3648e-13, 4.4786e-11,  0.0000e+00],
        [ 2.3421e-13,  6.4855e-15,  5.8832e-10,  1.3646e-11,  6.4695e-14, 5.3652e-12,  0.0000e+00],
        [-7.1441e-13, -3.6899e-14,  7.3648e-13,  6.4695e-14,  6.4468e-11, 3.2398e-09,  0.0000e+00],
        [-7.8546e-12, -8.0708e-14,  4.4786e-11,  5.3652e-12,  3.2398e-09, 5.2005e-06,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00]]))


def test_result2_cov():
    expected = np.array([[ 3.28939465e-08,  5.84137324e-10,  1.47452394e-12, 2.34217086e-13, -7.14414075e-13, -7.85582424e-12, 0.00000000e+00],
       [ 5.84137324e-10,  1.35379983e-11,  1.11718174e-13, 6.48437255e-15, -3.68973966e-14, -8.13532478e-14, 0.00000000e+00],
       [ 1.47452394e-12,  1.11718174e-13,  3.30145887e-08, 5.88324249e-10,  7.36476744e-13,  4.47858536e-11, 0.00000000e+00],
       [ 2.34217086e-13,  6.48437255e-15,  5.88324249e-10, 1.36463749e-11,  6.46948015e-14,  5.36516751e-12, 0.00000000e+00],
       [-7.14414075e-13, -3.68973966e-14,  7.36476744e-13, 6.46948015e-14,  6.44681053e-11,  3.23982437e-09, 0.00000000e+00],
       [-7.85582424e-12, -8.13532478e-14,  4.47858536e-11, 5.36516751e-12,  3.23982437e-09,  5.20046739e-06, 0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00]])
    actual = np.cov(result2.particles.t().numpy())
    assert np.allclose(actual, expected, rtol=1e-05, atol=1e-14, equal_nan=False)


def test_division_cov_result1_result2():
    assert str(result1._cov / np.cov(result2.particles.t().numpy())) == 'tensor([[1.0000, 1.0000, 0.9999, 1.0000, 1.0000, 0.9998,    nan],\n'\
        '        [1.0000, 1.0000, 1.0002, 1.0002, 1.0001, 0.9921,    nan],\n'\
        '        [0.9999, 1.0002, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n'\
        '        [1.0000, 1.0002, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n'\
        '        [1.0000, 1.0001, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n'\
        '        [0.9998, 0.9921, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n'\
        '        [   nan,    nan,    nan,    nan,    nan,    nan,    nan]],\n'\
        '       dtype=torch.float64)'

def test_division_cov_result2_result1():
    assert str(np.cov(result2.particles.t().numpy()) / result1._cov) == 'tensor([[1.0000, 1.0000, 1.0001, 1.0000, 1.0000, 1.0002,    nan],\n'\
        '        [1.0000, 1.0000, 0.9998, 0.9998, 0.9999, 1.0080,    nan],\n'\
        '        [1.0001, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n'\
        '        [1.0000, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n'\
        '        [1.0000, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n'\
        '        [1.0002, 1.0080, 1.0000, 1.0000, 1.0000, 1.0000,    nan],\n'\
        '        [   nan,    nan,    nan,    nan,    nan,    nan,    nan]],\n'\
        '       dtype=torch.float64)'