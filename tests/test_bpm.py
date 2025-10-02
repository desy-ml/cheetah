import torch

import cheetah


def test_reading_dtype_conversion():
    """Test that a dtype conversion is correctly reflected in the BPM reading."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0), dtype=torch.float32),
            cheetah.BPM(name="bpm", is_active=True, dtype=torch.float32),
        ],
    )
    beam = cheetah.ParameterBeam.from_parameters(dtype=torch.float32)
    assert segment.bpm.reading.dtype == torch.float32

    segment.track(beam)
    assert segment.bpm.reading.dtype == torch.float32

    segment = segment.double()
    assert segment.bpm.reading.dtype == torch.float64


def test_bpm_misalignment():
    """Test that the BPM misalignment is correctly applied to the reading."""
    bpm = cheetah.BPM(name="bpm", is_active=True, misalignment=torch.tensor([0.1, 0.2]))
    incoming = cheetah.ParameterBeam.from_parameters(
        mu_x=torch.tensor(0.0), mu_y=torch.tensor(0.0)
    )

    _ = bpm.track(incoming)

    assert torch.allclose(bpm.reading, -torch.tensor([0.1, 0.2]))
