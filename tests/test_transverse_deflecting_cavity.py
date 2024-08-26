import pytest
import torch

import cheetah


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transverse_deflecting_cavity_bmadx_tracking(dtype):
    """
    Test that the results of tracking through a TDC with the `"bmadx"` tracking method
    match the results from Bmad-X.
    """
    incoming_beam = torch.load(
        "tests/resources/bmadx/incoming_beam.pt", weights_only=False
    ).to(dtype)
    tdc = cheetah.TransverseDeflectingCavity(
        length=torch.tensor([1.0]),
        voltage=torch.tensor([1e7]),
        phase=torch.tensor([0.2], dtype=dtype),
        frequency=torch.tensor([1e9]),
        tracking_method="bmadx",
        dtype=dtype,
    )

    # Run tracking
    outgoing_beam = tdc.track(incoming_beam)

    # Load reference result computed with Bmad-X
    outgoing_bmadx = torch.load(
        "tests/resources/bmadx/outgoing_bmadx_crab_cavity.pt", weights_only=False
    )

    assert torch.allclose(
        outgoing_beam.particles,
        outgoing_bmadx.to(dtype),
        atol=1e-14 if dtype == torch.float64 else 0.00001,
        rtol=1e-14 if dtype == torch.float64 else 1e-8,
    )
