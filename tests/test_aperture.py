import torch

import cheetah


def test_aperture_shape():
    """Test that the two aperture shapes produce differently shaped beams."""

    incoming = cheetah.ParticleBeam.make_linspaced(num_particles=5)
    incoming.x = torch.tensor([0.0, 2e-4, 2e-4, -2e-4, -2e-4])
    incoming.y = torch.tensor([0.0, 3e-4, -3e-4, -3e-4, 3e-4])

    # Choose aperture size slightly larger than the beam width
    aperture = cheetah.Aperture(
        x_max=torch.tensor(2.2e-4), y_max=torch.tensor(3.2e-4), shape="rectangular"
    )
    outgoing_rectangular = aperture.track(incoming)

    aperture.shape = "elliptical"
    outgoing_elliptical = aperture.track(incoming)

    assert torch.allclose(outgoing_rectangular.survival_probabilities, torch.ones(5))
    assert torch.allclose(
        outgoing_elliptical.survival_probabilities,
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
    )
