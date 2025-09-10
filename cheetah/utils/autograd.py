import torch


class Log1plusXbyX(torch.autograd.Function):
    """
    Implements a custom autograd function for the compound expression log(1+x)/x.
    The singularity at 0 is replaced by its limit.
    """

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs  # inputs is always passed as a tuple

        ctx.save_for_backward(x, output)
        ctx.save_for_forward(x, output)

    @staticmethod
    def forward(x):
        return torch.where(x != 0, x.log1p() / x, x.new_ones(()))

    @staticmethod
    def backward(ctx, grad_output):
        x, result = ctx.saved_tensors
        return grad_output * Log1plusXbyX._gradient(x, result)

    @staticmethod
    def jvp(ctx, grad_input):
        x, result = ctx.saved_tensors
        return Log1plusXbyX._gradient(x, result) * grad_input

    @staticmethod
    def _gradient(x, fx):
        return torch.where(
            x != 0, ((1 + x).reciprocal() - fx) / x, -0.5 * x.new_ones(())
        )
