import numpy as np

import cheetah


def test_astra_to_parameter_beam():
    """Test that Astra beams are correctly loaded into parameter beams."""
    beam = cheetah.ParameterBeam.from_astra("benchmark/astra/ACHIP_EA1_2021.1351.001")

    # True values taken from version of Cheetah that is belived to be correct (v0.5.19)
    assert np.allclose(beam.mu_x, 8.24126345833065e-07)
    assert np.allclose(beam.mu_xp, 5.988477624896404e-08)
    assert np.allclose(beam.mu_y, -1.7276204289373709e-06)
    assert np.allclose(beam.mu_yp, -1.1746412553748087e-07)
    assert np.allclose(beam.sigma_x, 0.00017489789752289653)
    assert np.allclose(beam.sigma_xp, 3.679402198031312e-06)
    assert np.allclose(beam.sigma_y, 0.00017519544053357095)
    assert np.allclose(beam.sigma_yp, 3.6941000871593133e-06)
    assert np.allclose(beam.sigma_s, 8.011552381503861e-06)
    assert np.allclose(beam.sigma_p, 0.0022804534528404474)
    assert np.allclose(beam.energy, 107315902.44394557)


def test_astra_to_particle_beam():
    """Test that Astra beams are correctly loaded into particle beams."""
    beam = cheetah.ParticleBeam.from_astra("benchmark/astra/ACHIP_EA1_2021.1351.001")

    # True values taken from version of Cheetah that is belived to be correct (v0.5.19)
    assert beam.num_particles == 100_000
    assert np.allclose(beam.mu_x, 8.24126345833065e-07)
    assert np.allclose(beam.mu_xp, 5.988477624896404e-08)
    assert np.allclose(beam.mu_y, -1.7276204289373709e-06)
    assert np.allclose(beam.mu_yp, -1.1746412553748087e-07)
    assert np.allclose(beam.sigma_x, 0.00017489789752289653)
    assert np.allclose(beam.sigma_xp, 3.679402198031312e-06)
    assert np.allclose(beam.sigma_y, 0.00017519544053357095)
    assert np.allclose(beam.sigma_yp, 3.6941000871593133e-06)
    assert np.allclose(beam.sigma_s, 8.011552381503861e-06)
    assert np.allclose(beam.sigma_p, 0.0022804534528404474)
    assert np.allclose(beam.energy, 107315902.44394557)
