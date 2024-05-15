import torch

from cheetah import Drift, ParameterBeam, Quadrupole


def test_quadrupole_off():
    """
    Test that a quadrupole with k1=0 behaves still like a drift.
    """
    quadrupole = Quadrupole(length=torch.tensor([1.0]), k1=torch.tensor([0.0]))
    drift = Drift(length=torch.tensor([1.0]))
    incoming_beam = ParameterBeam.from_parameters(
        sigma_xp=torch.tensor([2e-7]), sigma_yp=torch.tensor([2e-7])
    )
    outbeam_quad = quadrupole(incoming_beam)
    outbeam_drift = drift(incoming_beam)

    quadrupole.k1 = torch.tensor([1.0], device=quadrupole.k1.device)
    outbeam_quad_on = quadrupole(incoming_beam)

    assert torch.allclose(outbeam_quad.sigma_x, outbeam_drift.sigma_x)
    assert not torch.allclose(outbeam_quad_on.sigma_x, outbeam_drift.sigma_x)


def test_quadrupole_with_misalignments_batched():
    """
    Test that a quadrupole with misalignments behaves as expected.
    """

    quad_with_misalignment = Quadrupole(
        length=torch.tensor([1.0]),
        k1=torch.tensor([1.0]),
        misalignment=torch.tensor([[0.1, 0.1]]),
    )

    quad_without_misalignment = Quadrupole(
        length=torch.tensor([1.0]), k1=torch.tensor([1.0])
    )
    incoming_beam = ParameterBeam.from_parameters(
        sigma_xp=torch.tensor([2e-7]), sigma_yp=torch.tensor([2e-7])
    )
    outbeam_quad_with_misalignment = quad_with_misalignment(incoming_beam)
    outbeam_quad_without_misalignment = quad_without_misalignment(incoming_beam)

    assert not torch.allclose(
        outbeam_quad_with_misalignment.mu_x,
        outbeam_quad_without_misalignment.mu_x,
    )


def test_quadrupole_with_misalignments_multiple_batch_dimension():
    """
    Test that a quadrupole with misalignments with multiple batch dimension.
    """
    batch_shape = torch.Size([4, 3])
    quad_with_misalignment = Quadrupole(
        length=torch.tensor([1.0]),
        k1=torch.tensor([1.0]),
        misalignment=torch.tensor([[0.1, 0.1]]),
    ).broadcast(batch_shape)

    quad_without_misalignment = Quadrupole(
        length=torch.tensor([1.0]), k1=torch.tensor([1.0])
    ).broadcast(batch_shape)
    incoming_beam = ParameterBeam.from_parameters(
        sigma_xp=torch.tensor([2e-7]), sigma_yp=torch.tensor([2e-7])
    ).broadcast(batch_shape)
    outbeam_quad_with_misalignment = quad_with_misalignment(incoming_beam)
    outbeam_quad_without_misalignment = quad_without_misalignment(incoming_beam)

    # Check that the misalignment has an effect
    assert not torch.allclose(
        outbeam_quad_with_misalignment.mu_x,
        outbeam_quad_without_misalignment.mu_x,
    )

    # Check that the output shape is correct
    assert outbeam_quad_with_misalignment.mu_x.shape == batch_shape
