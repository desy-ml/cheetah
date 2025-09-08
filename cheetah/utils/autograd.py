import torch


class Log1plusXbyX(torch.autograd.Function):
    """
    Implements a custom autograd function for the compound expression log(1+x)/x.
    The singularity at 0 is replaced by its limit.
    """

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs  # inputs is always passed as a tuple

        # Save for backward mode AD
        ctx.save_for_backward(x, output)

        # Save for forward mode AD
        ctx.x = x
        ctx.result = output

    @staticmethod
    def forward(x):
        return torch.where(x != 0, x.log1p() / x, x.new_ones(()))

    @staticmethod
    def backward(ctx, grad_output):
        x, result = ctx.saved_tensors
        return grad_output * torch.where(
            x != 0, ((1 + x).reciprocal() - result) / x, -0.5 * x.new_ones(())
        )

    @staticmethod
    def jvp(ctx, grad_input):
        result = (
            torch.where(
                ctx.x != 0,
                ((1 + ctx.x).reciprocal() - ctx.result) / ctx.x,
                -0.5 * ctx.x.new_ones(()),
            )
            * grad_input
        )

        # Clear context that is not required for backward pass
        del ctx.x
        del ctx.result

        return result
