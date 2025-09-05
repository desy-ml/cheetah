import torch


class Log1plusXbyX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = torch.where(x != 0, x.log1p() / x, torch.ones_like(x))
        ctx.save_for_backward(x, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, result = ctx.saved_tensors
        return grad_output * torch.where(
            x != 0, ((1 + x).reciprocal() - result) / x, -0.5 * torch.ones_like(x)
        )
