import pytest
import torch

import cheetah


def test_crab_cavity_bmadx_tracking():
    """
    Test that the results of tracking through a crab cavity with the `"bmadx"` tracking
    method match the results from Bmad-X.
    """
    incoming_beam = torch.load("tests/resources/bmadx/incoming_beam.pt")
    tdc = cheetah.CrabCavity(
        length=torch.tensor([1.0]),
        voltage=torch.tensor([1e7]),
        phase=torch.tensor([0.2]),
        frequency=torch.tensor([1e9]),
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
        outgoing_beam.particles, outgoing_beam_bmadx.particles, atol=1e-7, rtol=1e-7
    )
