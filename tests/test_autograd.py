import torch

from cheetah.utils.autograd import (
    cossqrtmcosdivdiff,
    log1pdiv,
    si1mdiv,
    si2msi2divdiff,
    sicos1mdiv,
    sicoskuddelmuddel15mdiv,
    simsidivdiff,
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


def test_simsidivdiff():
    """
    Verify that the custom autograd function simsidivdiff is correctly implementing
    `(si(sqrt(a)) - si(sqrt(b))) / (b - a)` and its derivative, including removing the
    singularity at `a == b`.
    """
    test_points_a = torch.tensor(
        [0.0, 0.0, -0.5, 0.1, 0.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True
    )
    test_points_b = torch.tensor(
        [-0.5, 0.1, 0.0, 0.0, 0.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True
    )

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=simsidivdiff,
        inputs=(test_points_a, test_points_b),
        rtol=0.01,
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )


def test_si2msi2divdiff():
    """
    Verify that the custom autograd function si2msi2divdiff is correctly implementing
    `(si^2(sqrt(b)) - si^2(sqrt(a))) / (a - b)` and its derivative, including removing
    the singularity at `a == b`.
    """
    test_points_a = torch.tensor(
        [0.0, 0.0, -0.5, 0.1, 0.0, 1.0, 1.0, 2.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    test_points_b = torch.tensor(
        [-0.5, 0.1, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=si2msi2divdiff,
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
