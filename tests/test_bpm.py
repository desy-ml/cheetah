import pytest
import torch

import cheetah


@pytest.mark.parametrize("is_bpm_active", [True, False])
@pytest.mark.parametrize("beam_class", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_no_tracking_error(is_bpm_active, beam_class):
    """Test that tracking a beam through an inactive BPM does not raise an error."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.BPM(name="my_bpm"),
            cheetah.Drift(length=torch.tensor(1.0)),
        ],
    )
    beam = beam_class.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    segment.my_bpm.is_active = is_bpm_active

    _ = segment.track(beam)


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
