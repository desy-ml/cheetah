import pytest
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
    Verify that the custom autograd function `si1mdiv` correctly implements the
    derivative of `(1 - si(sqrt(x))) / x`, including removing the singularity at 0.
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
    derivative of `(3 - 4 * si(sqrt(x)) + si(sqrt(x)) * cos(sqrt(x))) / (2 * x)`,
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


@pytest.mark.xfail(
    reason=(
        "No proper limit exists for sicoskuddelmuddel15mdiv. A different model will "
        "have to be found in the future that avoids these issues altogether."
    )
)
def test_sicoskuddelmuddel15mdiv():
    """
    Verify that the custom autograd function `sicoskuddelmuddel15mdiv` correctly
    implements the derivative of `(15 - 22.5 * si(sqrt(x)) + 9 * si(sqrt(x))
    * cos(sqrt(x)) - 1.5 * si(sqrt(x)) * cos^2(sqrt(x))) + x * si^3(sqrt(x)) / (x^3)`,
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
    derivative of `(si^2(sqrt(b)) - si^2(sqrt(a))) / (a - b)` and its derivative,
    including removing the singularity at `a == b`.

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
