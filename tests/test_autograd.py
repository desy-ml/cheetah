import torch

from cheetah.utils.autograd import Log1plusXbyX


def test_log1plusxbyx():
    """
    Test that verifies that the custom autograd function Log1plusXbyX is correctly
    implementing log(1+x)/x and its derivative, including removing the singularity at 0.
    """
    test_points = torch.tensor([-0.5, 0.0, 1.0])

    fwd_cheetah, bwd_cheetah = torch.autograd.functional.jvp(
        func=Log1plusXbyX.apply, inputs=test_points, v=torch.ones_like(test_points)
    )
    fwd_torch, bwd_torch = torch.autograd.functional.jvp(
        func=lambda z: z.log1p() / z, inputs=test_points, v=torch.ones_like(test_points)
    )

    assert not torch.any(fwd_cheetah.isnan())
    assert not torch.any(bwd_cheetah.isnan())

    assert torch.allclose(
        fwd_cheetah,
        torch.where(
            test_points != 0.0,
            fwd_torch,
            test_points.new_ones(()),
        ),
    )
    assert torch.allclose(
        bwd_cheetah,
        torch.where(
            test_points != 0.0,
            bwd_torch,
            -0.5 * test_points.new_ones(()),
        ),
    )
