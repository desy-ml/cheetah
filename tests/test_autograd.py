import torch

from cheetah.utils.autograd import Log1plusXbyX


def test_log1plusxbyx():
    """
    Test that verifies that the custom autograd function Log1plusXbyX is correctly
    implementing log(1+x)/x and its derivative, including removing the singularity at 0.
    """
    test_points = torch.tensor(
        [-0.5, 0.0, 1.0], dtype=torch.float64, requires_grad=True
    )

    forward = Log1plusXbyX.apply(test_points)
    assert not torch.any(forward.isnan())
    assert torch.allclose(
        forward,
        torch.where(
            test_points != 0.0,
            test_points.log1p() / test_points,
            test_points.new_ones(()),
        ),
    )

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(func=Log1plusXbyX.apply, inputs=test_points)
