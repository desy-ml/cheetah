import torch

import cheetah

beam_in = cheetah.ParticleBeam.from_parameters(num_particles=100)


# Only Marker
def test_tracking_marker_only():
    segment = cheetah.Segment([cheetah.Marker(name="start")])

    beam_out = segment.track(beam_in)

    assert torch.allclose(beam_out.particles, beam_in.particles)


# Only length-less elements between non-skippable elements
def test_tracking_lengthless_elements():
    segment = cheetah.Segment(
        [
            cheetah.Cavity(
                length=torch.tensor(0.1), voltage=torch.tensor(1e6), name="C2"
            ),
            cheetah.Marker(name="start"),
            cheetah.Cavity(
                length=torch.tensor(0.1), voltage=torch.tensor(1e6), name="C1"
            ),
        ]
    )

    _ = segment.track(beam_in)
