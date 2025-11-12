import torch

from cheetah.utils.autograd import log1pdiv, si1mdiv


def test_log1plusxbyx():
    """
    Verify that the custom autograd function log1pdiv is correctly implementing
    log(1+x)/x and its derivative, including removing the singularity at 0.
    """
    test_points = torch.tensor(
        [-0.5, 0.0, 1.0], dtype=torch.float64, requires_grad=True
    )

    forward = log1pdiv(test_points)
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
    assert torch.autograd.gradcheck(
        func=log1pdiv,
        inputs=test_points,
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )


def test_siminus1xbyx():
    """
    Verify that the custom autograd function si1mdiv is correctly implementing
    (1 - si(sqrt(x)))/x and its derivative, including removing the singularity at 0.
    """
    test_points = torch.tensor(
        [-0.5, 0.0, 1.0], dtype=torch.float64, requires_grad=True
    )
    si_points = (
        (torch.complex(test_points, test_points.new_zeros(())).sqrt() / torch.pi)
        .sinc()
        .real
    )

    forward = si1mdiv(test_points)
    assert not torch.any(forward.isnan())
    assert torch.allclose(
        forward,
        torch.where(
            test_points != 0.0,
            (1 - si_points) / test_points,
            test_points.new_ones(()) / 6,
        ),
    )

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=si1mdiv,
        inputs=test_points,
        rtol=0.01,
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )
