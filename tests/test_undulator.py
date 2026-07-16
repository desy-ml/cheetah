import torch

import cheetah


def test_undulator_off():
    """Test that an undulator with Kx=Ky=0 behaves like a drift."""
    undulator = cheetah.Undulator(length=torch.tensor(1.0), period=torch.tensor(0.1))
    drift = cheetah.Drift(length=torch.tensor(1.0))
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
    )

    outgoing_beam_drift = drift.track(incoming_beam)
    outgoing_beam_undulator_off = undulator.track(incoming_beam)

    undulator.Kx = torch.tensor(1.0)
    undulator.Ky = torch.tensor(0.0)
    outgoing_beam_undulator_on_y = undulator.track(incoming_beam)

    undulator.Kx = torch.tensor(0.0)
    undulator.Ky = torch.tensor(1.0)
    outgoing_beam_undulator_on_x = undulator.track(incoming_beam)

    assert torch.allclose(
        outgoing_beam_undulator_off.sigma_x, outgoing_beam_drift.sigma_x
    )
    assert torch.allclose(
        outgoing_beam_undulator_off.sigma_y, outgoing_beam_drift.sigma_y
    )
    assert not torch.allclose(
        outgoing_beam_undulator_on_y.sigma_y, outgoing_beam_drift.sigma_y
    )
    assert not torch.allclose(
        outgoing_beam_undulator_on_x.sigma_x, outgoing_beam_drift.sigma_x
    )
