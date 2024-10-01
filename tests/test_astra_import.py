import numpy as np
import torch

import cheetah


def test_astra_to_parameter_beam():
    """Test that Astra beams are correctly loaded into parameter beams."""
    beam = cheetah.ParameterBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # True values taken from version of Cheetah that is belived to be correct (v0.5.19)
    assert np.allclose(beam.mu_x.cpu().numpy(), 8.24126345833065e-07)
    assert np.allclose(beam.mu_px.cpu().numpy(), 5.988477624896404e-08)
    assert np.allclose(beam.mu_y.cpu().numpy(), -1.7276204289373709e-06)
    assert np.allclose(beam.mu_py.cpu().numpy(), -1.1746412553748087e-07)
    assert np.allclose(beam.sigma_x.cpu().numpy(), 0.00017489789752289653)
    assert np.allclose(beam.sigma_px.cpu().numpy(), 3.679402198031312e-06)
    assert np.allclose(beam.sigma_y.cpu().numpy(), 0.00017519544053357095)
    assert np.allclose(beam.sigma_py.cpu().numpy(), 3.6941000871593133e-06)
    assert np.allclose(beam.sigma_tau.cpu().numpy(), 8.011552381503861e-06)
    assert np.allclose(beam.sigma_p.cpu().numpy(), 0.0022804534528404474)
    assert np.allclose(beam.energy.cpu().numpy(), 107315902.44394557)
    assert np.allclose(beam.total_charge.cpu().numpy(), 5.000000000010205e-13)


def test_astra_to_particle_beam():
    """Test that Astra beams are correctly loaded into particle beams."""
    beam = cheetah.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # True values taken from version of Cheetah that is belived to be correct (v0.5.19)
    assert beam.num_particles == 100_000
    assert np.allclose(beam.mu_x.cpu().numpy(), 8.24126345833065e-07)
    assert np.allclose(beam.mu_px.cpu().numpy(), 5.988477624896404e-08)
    assert np.allclose(beam.mu_y.cpu().numpy(), -1.7276204289373709e-06)
    assert np.allclose(beam.mu_py.cpu().numpy(), -1.1746412553748087e-07)
    assert np.allclose(beam.sigma_x.cpu().numpy(), 0.00017489789752289653)
    assert np.allclose(beam.sigma_px.cpu().numpy(), 3.679402198031312e-06)
    assert np.allclose(beam.sigma_y.cpu().numpy(), 0.00017519544053357095)
    assert np.allclose(beam.sigma_py.cpu().numpy(), 3.6941000871593133e-06)
    assert np.allclose(beam.sigma_tau.cpu().numpy(), 8.011552381503861e-06)
    assert np.allclose(beam.sigma_p.cpu().numpy(), 0.0022804534528404474)
    assert np.allclose(beam.energy.cpu().numpy(), 107315902.44394557)
    assert np.allclose(beam.total_charge.cpu().numpy(), 5.000000000010205e-13)


def test_astra_to_parameter_beam_dtypes():
    """Test that Astra beams are correctly loaded into particle beams."""
    beam = cheetah.ParameterBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    assert beam.mu_x.dtype == torch.float32
    assert beam.mu_px.dtype == torch.float32
    assert beam.mu_y.dtype == torch.float32
    assert beam.mu_py.dtype == torch.float32
    assert beam.sigma_x.dtype == torch.float32
    assert beam.sigma_px.dtype == torch.float32
    assert beam.sigma_y.dtype == torch.float32
    assert beam.sigma_py.dtype == torch.float32
    assert beam.sigma_tau.dtype == torch.float32
    assert beam.sigma_p.dtype == torch.float32
    assert beam.energy.dtype == torch.float32
    assert beam.total_charge.dtype == torch.float32
