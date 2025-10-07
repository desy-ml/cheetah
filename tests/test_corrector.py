import pytest
import torch

import cheetah


def test_corrector_off():
    """
    Test that a corrector with horizontal & vertical angle=0 behaves still like a drift.
    """
    corrector = cheetah.Corrector(
        length=torch.tensor(1.0),
        horizontal_angle=torch.tensor(0.0),
        vertical_angle=torch.tensor(0.0),
    )
    drift = cheetah.Drift(length=torch.tensor(1.0))
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        mu_px=torch.tensor(2e-7), mu_py=torch.tensor(2e-7)
    )
    outbeam_corrector_off = corrector.track(incoming_beam)
    outbeam_drift = drift.track(incoming_beam)

    corrector.horizontal_angle = torch.tensor(
        1.0, device=corrector.horizontal_angle.device
    )
    corrector.vertical_angle = torch.tensor(0.0, device=corrector.vertical_angle.device)
    outbeam_corrector_h_on = corrector.track(incoming_beam)

    corrector.horizontal_angle = torch.tensor(
        0.0, device=corrector.horizontal_angle.device
    )
    corrector.vertical_angle = torch.tensor(1.0, device=corrector.vertical_angle.device)
    outbeam_corrector_v_on = corrector.track(incoming_beam)

    assert corrector.name is not None
    assert torch.allclose(outbeam_corrector_off.mu_px, outbeam_drift.mu_px)
    assert torch.allclose(outbeam_corrector_off.mu_py, outbeam_drift.mu_py)

    assert not torch.allclose(outbeam_corrector_h_on.mu_px, outbeam_drift.mu_px)
    assert torch.allclose(outbeam_corrector_h_on.mu_py, outbeam_drift.mu_py)

    assert torch.allclose(outbeam_corrector_v_on.mu_px, outbeam_drift.mu_px)
    assert not torch.allclose(outbeam_corrector_v_on.mu_py, outbeam_drift.mu_py)


def test_corrector_vectorized_execution():
    """
    Test that a corrector with vector dimensions behaves as expected.
    """
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100, energy=torch.tensor(1e9), mu_x=torch.tensor(1e-5)
    )

    # Test vectorisation to generate 3 beam lines
    segment = cheetah.Segment(
        [
            cheetah.Corrector(
                length=torch.tensor([0.5, 0.5, 0.5]),
                horizontal_angle=torch.tensor([0.1, 0.0, 0.1]),
                vertical_angle=torch.tensor([0.1, 0.0, 0.1]),
            ),
            cheetah.Drift(length=torch.tensor(0.5)),
        ]
    )
    outgoing = segment.track(incoming)

    assert outgoing.particles.shape == torch.Size([3, 100, 7])
    assert outgoing.mu_x.shape == torch.Size([3])

    # Check that corrector with same h & v angle produce same output
    assert torch.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check different angles do make a difference
    assert not torch.allclose(outgoing.particles[0], outgoing.particles[1])

    # Test vectorisation to generate 18 beamlines
    segment = cheetah.Segment(
        [
            cheetah.Corrector(
                length=torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1),
                horizontal_angle=torch.tensor([0.1, 0.2, 0.1]).reshape(1, 3),
            ),
            cheetah.Drift(length=torch.tensor([0.5, 1.0]).reshape(2, 1, 1)),
        ]
    )
    outgoing = segment.track(incoming)
    assert outgoing.particles.shape == torch.Size([2, 3, 3, 100, 7])

    # Test improper vectorisation -- this does not obey torch broadcasting rules
    segment = cheetah.Segment(
        [
            cheetah.Corrector(
                length=torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1),
                horizontal_angle=torch.tensor([0.1, 0.2, 0.1]).reshape(1, 3),
            ),
            cheetah.Drift(length=torch.tensor([0.5, 1.0]).reshape(2, 1)),
        ]
    )
    with pytest.raises(RuntimeError):
        segment.track(incoming)
