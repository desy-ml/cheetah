import torch

from cheetah import Drift, ParameterBeam, ParticleBeam, Segment, VerticalCorrector


def test_vertical_corrector_off_on():
    """
    Test that a corrector with angle=0 behaves
    still like a drift and that the angle translates properly.
    """
    corrector = VerticalCorrector(length=torch.tensor([0.3]), angle=torch.tensor([0.0]))
    drift = Drift(length=torch.tensor([1.0]))
    incoming_beam = ParameterBeam.from_twiss(
        energy=torch.tensor([1.8e7]), beta_x=torch.tensor([5]), beta_y=torch.tensor([5])
    )

    outbeam_corrector_off = corrector(incoming_beam)
    outbeam_drift = drift(incoming_beam)

    corrector.angle = torch.tensor([7.0], device=corrector.angle.device)
    outbeam_corrector_on = corrector(incoming_beam)

    assert torch.allclose(outbeam_corrector_off.mu_yp, outbeam_drift.mu_yp)
    assert torch.allclose(outbeam_corrector_on.mu_xp, outbeam_drift.mu_xp)
    assert torch.allclose(outbeam_corrector_on.mu_yp, corrector.angle)
    assert not torch.allclose(outbeam_corrector_on.mu_yp, outbeam_drift.mu_yp)


def test_vertical_angle_property():
    try:
        VerticalCorrector(
            length=torch.tensor([0.3]), vertical_angle=torch.tensor([0.0])
        )
    except TypeError:
        pass


def test_horizontal_angle_property():
    try:
        VerticalCorrector(
            length=torch.tensor([0.3]), vertical_angle=torch.tensor([0.0])
        )
    except TypeError:
        pass


def test_corrector_batched_execution():
    """
    Test that a corrector with batch dimensions behaves as expected.
    """
    batch_shape = torch.Size([3])
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1000000), energy=torch.tensor([1.8e7])
    ).broadcast(batch_shape)
    segment = Segment(
        [
            VerticalCorrector(
                length=torch.tensor([0.04, 0.04, 0.04]),
                angle=torch.tensor([0.001, 0.003, 0.001]),
            ),
            Drift(length=torch.tensor([0.5])).broadcast(batch_shape),
        ]
    )

    outgoing = segment(incoming)

    # Check that dipole with same bend angle produce same output
    assert torch.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check different angles do make a difference
    assert not torch.allclose(outgoing.particles[0], outgoing.particles[1])
