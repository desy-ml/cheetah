import pytest
import torch

import cheetah


def test_transverse_deflecting_cavity_bmadx_tracking():
    """
    Test that the results of tracking through a TDC with the `"bmadx"` tracking
    method match the results from Bmad-X.
    """
    incoming_beam = torch.load("tests/resources/bmadx/incoming_beam.pt")
    tdc = cheetah.TransverseDeflectingCavity(
        length=torch.tensor([1.0], dtype=torch.double),
        voltage=torch.tensor([1e7], dtype=torch.double),
        phase=torch.tensor([0.2], dtype=torch.double),
        frequency=torch.tensor([1e9], dtype=torch.double),
        tracking_method="bmadx",
        dtype=torch.double,
    )

    # Run tracking
    outgoing_beam = tdc.track(incoming_beam)

    # Load reference result computed with Bmad-X
    outgoing_beam_bmadx = torch.load(
        "tests/resources/bmadx/outgoing_beam_bmadx_crab_cavity.pt"
    )

    assert torch.allclose(
        outgoing_beam.particles, outgoing_beam_bmadx.particles, atol=1e-14, rtol=1e-14
    )
