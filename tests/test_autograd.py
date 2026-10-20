import torch

from cheetah.utils.autograd import (
    cossqrtmcosdivdiff,
    log1pdiv,
    normalized_j3,
    normalized_jf,
    si1mdiv,
    si2msi2divdiff,
    sicos1mdiv,
    simsidivdiff,
    sipsicos3mdiv,
    sqrta2minusbdiva,
)


def test_log1plusxbyx():
    """
    Verify that the custom autograd function `log1pdiv` correctly implements the
    derivative of `log(1 + x) / x`, including removing the singularity at 0.
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
    Verify that `si1mdiv` correctly implements the
    derivative of `(1 - si(sqrt(x))) / x`, including removing the singularity at 0.
    """
    test_points = torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True
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
    assert torch.autograd.gradgradcheck(si1mdiv, test_points, check_fwd_over_rev=True)


def test_sicos1mdiv():
    """
    Verify that the custom autograd function `sicos1mdiv` correctly implements the
    derivative of `(1 - si(sqrt(x)) * cos(sqrt(x))) / x`, including removing the
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
    Verify that the custom autograd function `sipsicos3mdiv` correctly implements the
    derivative of `(3 - 4 * si(sqrt(x)) + si(sqrt(x)) * cos(sqrt(x))) / (2 * x**2)`,
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


def test_normalized_j3():
    """Check J3 reverse/forward AD, including zero and both series boundaries."""
    test_points = torch.tensor(
        [-4.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 4.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=normalized_j3,
        inputs=test_points,
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )
    assert torch.autograd.gradgradcheck(
        normalized_j3, test_points, check_fwd_over_rev=True
    )


def test_normalized_jf():
    """Check Jf AD at zero, signed strengths, and exact/near nonzero resonance."""
    a = torch.tensor(
        [0.0, 0.0, -0.5, 1.0, 1.0, 2.0, 2.0 - 1e-8, 2.0 + 1e-8, -2.0, 2.0, 0.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    b = torch.tensor(
        [0.0, -0.5, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, -2.0, 1.8, 2.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        normalized_jf,
        (a, b),
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )
    assert torch.autograd.gradgradcheck(normalized_jf, (a, b), check_fwd_over_rev=True)
    assert torch.autograd.gradcheck(normalized_jf, (a[:3, None], b[None, :3]))


def test_cossqrtmcosdivdiff():
    """
    Verify that the custom autograd function `cossqrtmcosdivdiff` correctly implements
    the derivative of `(cos(sqrt(b)) - cos(sqrt(a))) / (a - b)`, including removing the
    singularity at `a == b`.
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
    Verify that the custom autograd function `simsidivdiff` correctly implements the
    derivative of `(si(sqrt(a)) - si(sqrt(b))) / (b - a)`, including removing the
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
    Verify that the custom autograd function `si2msi2divdiff` correctly implements the
    derivative of `(si^2(sqrt(b)) - si^2(sqrt(a))) / (a - b)`, including removing the
    singularity at `a == b`.

    NOTE: Forward AD currently doesn't work for a==0 or b==0. That's why those checks
        are disabled.
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
        # check_forward_ad=True,
        check_batched_grad=True,
        # check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )


def test_sqrta2minusbdiva():
    """
    Verify that the custom autograd function `sqrta2minusbdiva` correctly implements the
    derivative of `(sqrt(a^2 + b) - a) / b`, including removing the singularity at
    `a == 0`.
    """
    test_points = torch.tensor(
        [-0.1, 0.0, 1.0], dtype=torch.float64, requires_grad=True
    )
    a = torch.tensor(0.5, dtype=torch.float64)

    # Check gradient calculation using finite difference methods
    assert torch.autograd.gradcheck(
        func=sqrta2minusbdiva,
        inputs=(a, test_points),
        check_backward_ad=True,
        check_forward_ad=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_grad_dtypes=True,
    )
