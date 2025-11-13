import torch


def log1pdiv(x: torch.Tensor) -> torch.Tensor:
    """Calculate log(1+x)/x with proper removal of its singularity at 0."""
    return Log1PDiv.apply(x)


def si1mdiv(x: torch.Tensor) -> torch.Tensor:
    """Calculate (1 - si(sqrt(x)))/x with proper removal of its singularity at 0."""
    return Si1MDiv.apply(x)


def sqrta2minusbdiva(c: torch.Tensor, g_tilde: torch.Tensor) -> torch.Tensor:
    """
    Calculate (sqrt(c^2 + g_tilde) - c) / g_tilde with proper removal of its singularity
    at g_tilde=0.
    """
    return SqrtA2MinusBDivA.apply(c, g_tilde)


class Log1PDiv(torch.autograd.Function):
    """
    Custom autograd function for the compound expression log(1+x)/x. The singularity at
    0 is replaced by its limit.
    """

    # Automatically generate a custom vmap implementation
    generate_vmap_rule = True

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
        x, fx = ctx.saved_tensors
        return grad_output * torch.where(
            x != 0, ((1 + x).reciprocal() - fx) / x, -0.5 * x.new_ones(())
        )

    @staticmethod
    def jvp(ctx, grad_input):
        x, fx = ctx.saved_tensors
        return (
            torch.where(x != 0, ((1 + x).reciprocal() - fx) / x, -0.5 * x.new_ones(()))
            * grad_input
        )


class Si1MDiv(torch.autograd.Function):
    """
    Custom autograd function for the compound expression (1-si(sqrt(x)))/x. The
    singularity at 0 is replaced by its limit.
    """

    # Automatically generate a custom vmap implementation
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs  # inputs is always passed as a tuple

        ctx.save_for_backward(x, output)
        ctx.save_for_forward(x, output)

    @staticmethod
    def forward(x):
        # Since x may be negative, we use complex arithmetic for the sqrt
        sx = (torch.complex(x, x.new_zeros(())).sqrt() / torch.pi).sinc().real
        return torch.where(x != 0, (1 - sx) / x, x.new_ones(()) / 6.0)

    @staticmethod
    def backward(ctx, grad_output):
        x, fx = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()
        sx = (torch.sinc(sqrt_x / torch.pi) - sqrt_x.cos()).real / (2 * x)

        return grad_output * torch.where(x != 0, (sx - fx) / x, -x.new_ones(()) / 120)

    @staticmethod
    def jvp(ctx, grad_input):
        x, fx = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()
        sx = (torch.sinc(sqrt_x / torch.pi) - sqrt_x.cos()).real / (2 * x)

        return torch.where(x != 0, (sx - fx) / x, -x.new_ones(()) / 120) * grad_input


class SqrtA2MinusBDivA(torch.autograd.Function):
    """
    Custom autograd function for the compound expression
    ((sqrt(c^2 + g_tilde) - c) / g_tilde). The singularity at g_tilde=0 is replaced by
    its limit.
    """

    # Automatically generate a custom vmap implementation
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        (c, g_tilde) = inputs

        ctx.save_for_backward(c, g_tilde, output)
        ctx.save_for_forward(c, g_tilde, output)

    @staticmethod
    def forward(c, g_tilde):
        return torch.where(
            g_tilde != 0,
            ((c.square() + g_tilde).sqrt() - c) / g_tilde,
            (2 * c).reciprocal(),
        )

    @staticmethod
    def backward(ctx, grad_output):
        c, g_tilde, fx = ctx.saved_tensors

        grad_c = torch.where(
            g_tilde != 0,
            (c / (c.square() + g_tilde).sqrt() - 1) / g_tilde,
            -(2.0 * c.square()).reciprocal(),
        )

        grad_g_tilde = torch.where(
            g_tilde != 0,
            ((-2.0 * c.square() - g_tilde) / (c.square() + g_tilde).sqrt() + 2.0 * c)
            / (2.0 * g_tilde.square()),
            -(8.0 * c.pow(3)).reciprocal(),
        )

        return grad_output * grad_c, grad_output * grad_g_tilde
