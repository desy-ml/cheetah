import torch


class Log1plusXbyX(torch.autograd.Function):
    """
    Implements a custom autograd function for the compound expression log(1+x)/x.
    The singularity at 0 is replaced by its limit.
    """

    @staticmethod
    def forward(ctx, x):
        result = torch.where(x != 0, x.log1p() / x, x.new_ones(()))
        ctx.save_for_backward(x, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, result = ctx.saved_tensors
        return grad_output * torch.where(
            x != 0, ((1 + x).reciprocal() - result) / x, -0.5 * x.new_ones(())
        )
