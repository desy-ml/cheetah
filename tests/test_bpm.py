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
    beam = beam_class.from_astra("benchmark/cheetah/ACHIP_EA1_2021.1351.001")

    segment.my_bpm.is_active = is_bpm_active

    _ = segment.track(beam)
