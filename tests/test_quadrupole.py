import numpy as np
import torch

from cheetah import Drift, ParameterBeam, Quadrupole


def test_quadrupole_off():
    """
    Test that a quadrupole with k1=0 behaves still like a drift.
    """
    quadrupole = Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(0.0))
    drift = Drift(length=torch.tensor(1.0))
    incoming_beam = ParameterBeam.from_parameters(
        sigma_xp=torch.tensor(2e-7), sigma_yp=torch.tensor(2e-7)
    )
    outbeam_quad = quadrupole(incoming_beam)
    outbeam_drift = drift(incoming_beam)

    quadrupole.k1 = torch.tensor(1.0)
    outbeam_quad_on = quadrupole(incoming_beam)

    assert np.allclose(outbeam_quad.sigma_x, outbeam_drift.sigma_x)
    assert not np.allclose(outbeam_quad_on.sigma_x, outbeam_drift.sigma_x)
