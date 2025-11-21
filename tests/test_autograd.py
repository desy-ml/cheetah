import torch

from cheetah.utils.autograd import (
    cossqrtmcosdivdiff,
    log1pdiv,
    si1mdiv,
    sicos1mdiv,
    sicoskuddelmuddel15mdiv,
    sipsicos3mdiv,
    sqrta2minusbdiva,
)


def test_log1plusxbyx():
    """
    Verify that the custom autograd function `log1pdiv` correctly implements
    `log(1 + x) / x` and its derivative, including removing the singularity at 0.
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
    Verify that the custom autograd function `si1mdiv` correctly implements
    `(1 - si(sqrt(x))) / x` and its derivative, including removing the singularity at 0.
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


def test_sicos1mdiv():
    """
    Verify that the custom autograd function `sicos1mdiv` correctly implements
    `(1 - si(sqrt(x)) * cos(sqrt(x))) / x` and its derivative, including removing the
    singularity at 0.
    """
    test_points = torch.tensor(
        [-0.5, 0.0, 1.0], dtype=torch.float64, requires_grad=True
    )
    sqrt_points = torch.complex(test_points, test_points.new_zeros(())).sqrt()
    si_points = (sqrt_points / torch.pi).sinc().real
    cos_points = sqrt_points.cos().real

    forward = sicos1mdiv(test_points)
    assert not torch.any(forward.isnan())
    assert torch.allclose(
        forward,
        torch.where(
            test_points != 0.0,
            (1 - si_points * cos_points) / test_points,
            test_points.new_ones(()) / 6.0,
        ),
    )

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=sicos1mdiv,
        inputs=test_points,
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )


def test_sipsicos3mdiv():
    """
    Verify that the custom autograd function `sipsicos3mdiv` correctly implements
    `(3 - 4 * si(sqrt(x)) + si(sqrt(x)) * cos(sqrt(x))) / (2 * x)` and its derivative,
    including removing the singularity at 0.
    """
    test_points = torch.tensor(
        [-0.5, 0.0, 1.0], dtype=torch.float64, requires_grad=True
    )
    sqrt_points = torch.complex(test_points, test_points.new_zeros(())).sqrt()
    si_points = (sqrt_points / torch.pi).sinc().real
    cos_points = sqrt_points.cos().real

    forward = sipsicos3mdiv(test_points)
    assert not torch.any(forward.isnan())
    assert torch.allclose(
        forward,
        torch.where(
            test_points != 0.0,
            (3.0 - 4.0 * si_points + si_points * cos_points) / (2.0 * test_points),
            test_points.new_zeros(()),
        ),
    )

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=sipsicos3mdiv,
        inputs=test_points,
        rtol=0.01,
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )


def test_sicoskuddelmuddel15mdiv():
    """
    Verify that the custom autograd function `sicoskuddelmuddel15mdiv` correctly
    implements `(15 - 22.5 * si(sqrt(x)) + 9 * si(sqrt(x)) * cos(sqrt(x)) - 1.5
    * si(sqrt(x)) * cos^2(sqrt(x))) + x * si^3(sqrt(x)) / (x^3)` and its derivative,
    including removing the singularity at `x == 0`.
    """
    test_points = torch.tensor(
        [-0.5, 0.0, 1.0], dtype=torch.float64, requires_grad=True
    )
    sqrt_points = torch.complex(test_points, test_points.new_zeros(())).sqrt()
    si_points = (sqrt_points / torch.pi).sinc().real
    cos_points = sqrt_points.cos().real

    forward = sicoskuddelmuddel15mdiv(test_points)
    assert not forward.isnan().any()
    assert torch.allclose(
        forward,
        torch.where(
            test_points != 0,
            (
                15.0
                - 22.5 * si_points
                + 9.0 * si_points * cos_points
                - 1.5 * si_points * cos_points.square()
                + test_points * si_points.pow(3)
            )
            / (test_points.pow(3)),
            1.0 / 56.0,
        ),
    )

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=sicoskuddelmuddel15mdiv,
        inputs=test_points,
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )


def test_cossqrtmcosdivdiff():
    """
    Verify that the custom autograd function cossqrtmcosdivdiff is correctly
    implementing `(cos(sqrt(b)) - cos(sqrt(a))) / (a - b)` and its derivative, including
    removing the singularity at `a == b`.
    """
    test_points_a = torch.tensor(
        [-0.5, 0.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True
    )
    test_points_b = torch.tensor(
        [0.0, 0.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True
    )
    sqrt_a_points = torch.complex(test_points_a, test_points_a.new_zeros(())).sqrt()
    sqrt_b_points = torch.complex(test_points_b, test_points_b.new_zeros(())).sqrt()
    sa_points = (sqrt_a_points / torch.pi).sinc().real
    ca_points = sqrt_a_points.cos().real
    cb_points = sqrt_b_points.cos().real
    demoninator_points = test_points_a - test_points_b

    forward = cossqrtmcosdivdiff(test_points_a, test_points_b)
    assert not forward.isnan().any()
    assert torch.allclose(
        forward,
        torch.where(
            demoninator_points != 0,
            (cb_points - ca_points) / demoninator_points,
            0.5 * sa_points,
        ),
    )
    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=cossqrtmcosdivdiff,
        inputs=(test_points_a, test_points_b),
        rtol=0.01,
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )


def test_sqrta2minusbdiva():
    """
    Verify that the custom autograd function sqrta2minusbdiva is correctly implementing
    `(sqrt(c^2 + g_tilde) - c) / g_tilde` and its derivative, including removing the
    singularity at `g_tilde == 0`.
    """
    test_points = torch.tensor(
        [-0.1, 0.0, 1.0], dtype=torch.float64, requires_grad=True
    )
    c = torch.tensor(0.5, dtype=torch.float64)

    forward = sqrta2minusbdiva(c, test_points)
    assert not forward.isnan().any()
    assert torch.allclose(
        forward,
        torch.where(
            test_points != 0,
            ((c.square() + test_points).sqrt() - c) / test_points,
            (2 * c).reciprocal(),
        ),
    )

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=sqrta2minusbdiva,
        inputs=(c, test_points),
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )
