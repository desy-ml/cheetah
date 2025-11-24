import torch


def log1pdiv(x: torch.Tensor) -> torch.Tensor:
    """Calculate `log(1 + x) / x` with proper removal of its singularity at 0."""
    return Log1PDiv.apply(x)


def si1mdiv(x: torch.Tensor) -> torch.Tensor:
    """Calculate `(1 - si(sqrt(x))) / x` with proper removal of its singularity at 0."""
    return Si1MDiv.apply(x)


def sicos1mdiv(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(1 - si(sqrt(x)) * cos(sqrt(x))) / x` with proper removal of its
    singularity at 0.
    """
    return SiCos1MDiv.apply(x)


def sipsicos3mdiv(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(3 - 4 * si(sqrt(x)) + si(sqrt(x)) * cos(sqrt(x))) / (2 * x)` with proper
    removal of its singularity at 0.
    """
    return SiPSiCos3MDiv.apply(x)


def sicoskuddelmuddel15mdiv(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(15 - 22.5 * si(sqrt(x)) + 9 * si(sqrt(x)) * cos(sqrt(x)) - 1.5
    * si(sqrt(x)) * cos^2(sqrt(x))) + x * si^3(sqrt(x)) / (x^3)` with proper removal of
    its singularity at 0.
    """
    return SiCosKuddelMuddel15MDiv.apply(x)


def cossqrtmcosdivdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(cos(sqrt(b)) - cos(sqrt(a))) / (a - b)` with proper removal of its
    singularity at `a == b`.
    """
    return CosSqrtMCosDivDiff.apply(a, b)


def simsidivdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(si(sqrt(a)) - si(sqrt(b))) / (b - a)` with proper removal of its
    singularity at `a == b`.
    """
    return SiMSiDivDiff.apply(a, b)


def sqrta2minusbdiva(c: torch.Tensor, g_tilde: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(sqrt(c^2 + g_tilde) - c) / g_tilde` with proper removal of its
    singularity at `g_tilde == 0`.
    """
    return SqrtA2MinusBDivA.apply(c, g_tilde)


class Log1PDiv(torch.autograd.Function):
    """
    Custom autograd function for the compound expression `log(1 + x) / x`. The
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
    Custom autograd function for the compound expression `(1 - si(sqrt(x))) / x`. The
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
        sx = ((sqrt_x / torch.pi).sinc() - sqrt_x.cos()).real / (2 * x)

        return grad_output * torch.where(x != 0, (sx - fx) / x, -x.new_ones(()) / 120)

    @staticmethod
    def jvp(ctx, grad_input):
        x, fx = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()
        sx = ((sqrt_x / torch.pi).sinc() - sqrt_x.cos()).real / (2 * x)

        return torch.where(x != 0, (sx - fx) / x, -x.new_ones(()) / 120) * grad_input


class SiCos1MDiv(torch.autograd.Function):
    """
    Custom autograd function for the compound expression
    `(1 - si(sqrt(x)) * cos(sqrt(x))) / x`. The singularity at 0 is replaced by its
    limit.
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
        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        return torch.where(x != 0, (1 - sx * cx) / x, x.new_ones(()) / 6.0)

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = torch.where(
            x != 0,
            (sx * (x * sx + 2.0 * cx) - 2.0 - cx.square() + sx * cx)
            / (2.0 * x.square()),
            -2.0 / 15.0,
        )

        return grad_output * grad

    @staticmethod
    def jvp(ctx, grad_input):
        x, _ = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = torch.where(
            x != 0,
            (sx * (x * sx + 2.0 * cx) - 2.0 - cx.square() + sx * cx)
            / (2.0 * x.square()),
            -2.0 / 15.0,
        )

        return grad * grad_input


class SiPSiCos3MDiv(torch.autograd.Function):
    """
    Custom autograd function for the compound expression
    `(3 - 4 * si(sqrt(x)) + si(sqrt(x)) * cos(sqrt(x))) / (2 * x). The singularity at 0
    is replaced by its limit.
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
        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        return torch.where(
            x != 0, (3.0 - 4.0 * sx + sx * cx) / (2.0 * x), x.new_zeros(())
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = torch.where(
            x != 0,
            (
                -sx * (x * sx + 2.0 * cx - 8.0)
                - 6.0
                + 4.0 * sx
                + cx.square()
                - (4.0 + sx) * cx
            )
            / (4.0 * x.square()),
            0.05,
        )

        return grad_output * grad

    @staticmethod
    def jvp(ctx, grad_input):
        x, _ = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = torch.where(
            x != 0,
            (
                -sx * (x * sx + 2.0 * cx - 8.0)
                - 6.0
                + 4.0 * sx
                + cx.square()
                - (4.0 + sx) * cx
            )
            / (4.0 * x.square()),
            0.05,
        )

        return grad * grad_input


class SiCosKuddelMuddel15MDiv(torch.autograd.Function):
    """
    Custom autograd function for the compound expression
    `(15 - 22.5 * si(sqrt(x)) + 9 * si(sqrt(x)) * cos(sqrt(x)) - 1.5 * si(sqrt(x))
    * cos^2(sqrt(x))) + x * si^3(sqrt(x)) / (x^3)`. The singularity at 0 is replaced by
    its limit.
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
        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        return torch.where(
            x != 0,
            (15.0 - 22.5 * sx + 9.0 * sx * cx - 1.5 * sx * cx.square() + x * sx.pow(3))
            / (x.pow(3)),
            1.0 / 56.0,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = (
            -2.0 * x * sx.pow(3)
            + sx.square() * (1.5 * x * cx - 1.5 * x * sx)
            + sx
            * (-4.5 * x * sx + 4.5 * cx.square() + (1.5 * x * sx - 27.0) * cx + 67.5)
            - 45.0
            + 11.25 * sx
            - 0.75 * cx.pow(3)
            + (4.5 + 0.75 * sx) * cx.square()
            + (-11.25 - 4.5 * sx) * cx
        ) / (x.pow(4))

        return grad_output * grad

    @staticmethod
    def jvp(ctx, grad_input):
        x, _ = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = (
            -2.0 * x * sx.pow(3)
            + sx.square() * (1.5 * x * cx - 1.5 * x * sx)
            + sx
            * (-4.5 * x * sx + 4.5 * cx.square() + (1.5 * x * sx - 27.0) * cx + 67.5)
            - 45.0
            + 11.25 * sx
            - -0.75 * cx.pow(3)
            + (4.5 + 0.75 * sx) * cx.square()
            + (-11.25 - 4.5 * sx) * cx
        ) / (x.pow(4))

        return grad * grad_input


class CosSqrtMCosDivDiff(torch.autograd.Function):
    """
    Custom autograd function for the compound expression
    `(cos(sqrt(b)) - cos(sqrt(a)) / (a - b)`. The singularity at 0 is replaced by
    its limit.
    """

    # Automatically generate a custom vmap implementation
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        (a, b) = inputs  # inputs is always passed as a tuple

        ctx.save_for_backward(a, b, output)
        ctx.save_for_forward(a, b, output)

    @staticmethod
    def forward(a, b):
        sqrt_a = torch.complex(a, a.new_zeros(())).sqrt()
        sqrt_b = torch.complex(b, b.new_zeros(())).sqrt()

        sa = (sqrt_a / torch.pi).sinc().real
        ca = sqrt_a.cos().real
        cb = sqrt_b.cos().real

        demoninator = a - b

        return torch.where(demoninator != 0, (cb - ca) / demoninator, 0.5 * sa)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, _ = ctx.saved_tensors

        sqrt_a = torch.complex(a, a.new_zeros(())).sqrt()
        sqrt_b = torch.complex(b, b.new_zeros(())).sqrt()
        sa = (sqrt_a / torch.pi).sinc().real
        sb = (sqrt_b / torch.pi).sinc().real
        ca = sqrt_a.cos().real
        cb = sqrt_b.cos().real
        ab = a - b
        cbca = cb - ca
        demoninator = ab.square()

        limit = torch.where(a != 0, (ca - sa) / (8.0 * a), -1.0 / 24.0)

        grad_a = torch.where(a != b, (0.5 * sa * ab - cbca) / demoninator, limit)
        grad_b = torch.where(a != b, -(0.5 * sb * ab - cbca) / demoninator, limit)

        return grad_output * grad_a, grad_output * grad_b

    @staticmethod
    def jvp(ctx, grad_a, grad_b):
        a, b, _ = ctx.saved_tensors

        sqrt_a = torch.complex(a, a.new_zeros(())).sqrt()
        sqrt_b = torch.complex(b, b.new_zeros(())).sqrt()
        sa = (sqrt_a / torch.pi).sinc().real
        sb = (sqrt_b / torch.pi).sinc().real
        ca = sqrt_a.cos().real
        cb = sqrt_b.cos().real
        ab = a - b

        return torch.where(
            a != b,
            ((0.5 * ab * (grad_a * sa - grad_b * sb)) + (grad_a - grad_b) * (ca - cb))
            / ab.square(),
            torch.where(
                b != 0,
                ((grad_b + grad_a) * cb - (grad_b + grad_a) * sb) / (8.0 * b),
                -(grad_a + grad_b) / 24.0,
            ),
        )


class SiMSiDivDiff(torch.autograd.Function):
    """
    Custom autograd function for the compound expression
    `(si(sqrt(a)) - si(sqrt(b))) / (b - a)`. The singularity at`a == b` is replaced by
    its limit.
    """

    # Automatically generate a custom vmap implementation
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        (a, b) = inputs

        ctx.save_for_backward(a, b, output)
        ctx.save_for_forward(a, b, output)

    @staticmethod
    def forward(a, b):
        sqrt_a = torch.complex(a, a.new_zeros(())).sqrt()
        sqrt_b = torch.complex(b, b.new_zeros(())).sqrt()

        sa = (sqrt_a / torch.pi).sinc().real
        sb = (sqrt_b / torch.pi).sinc().real
        cb = sqrt_b.cos().real

        return torch.where(
            a != b, (sa - sb) / (b - a), torch.where(b != 0, 0.5 * (sb - cb) / b, 1 / 6)
        )

    @staticmethod
    def backward(ctx, grad_output):
        a, b, _ = ctx.saved_tensors

        sqrt_a = torch.complex(a, a.new_zeros(())).sqrt()
        sqrt_b = torch.complex(b, b.new_zeros(())).sqrt()

        sa = (sqrt_a / torch.pi).sinc().real
        sb = (sqrt_b / torch.pi).sinc().real
        ca = sqrt_a.cos().real
        cb = sqrt_b.cos().real

        ba = b - a

        a0_b0_limit = -1.0 / 120.0
        aeqb_limit = torch.where(
            b != 0, (3.0 * cb + (b - 3.0) * sb) / (8.0 * b.square()), a0_b0_limit
        )
        aneqb_a0_limit = (1.0 - b / 6.0 - sb) / b.square()
        aneqb_b0_limit = (1.0 - a / 6.0 - sa) / a.square()

        grad_a = torch.where(
            (a != b).logical_and(a != 0),
            (ca - sa) / (2.0 * a * ba) + (sa - sb) / ba.square(),
            torch.where(a != b, aneqb_a0_limit, aeqb_limit),
        )
        grad_b = torch.where(
            (a != b).logical_and(b != 0),
            -(cb - sb) / (2.0 * b * ba) + (sb - sa) / ba.square(),
            torch.where(a != b, aneqb_b0_limit, aeqb_limit),
        )

        return grad_output * grad_a, grad_output * grad_b

    @staticmethod
    def jvp(ctx, grad_a, grad_b):
        a, b, _ = ctx.saved_tensors

        sqrt_a = torch.complex(a, a.new_zeros(())).sqrt()
        sqrt_b = torch.complex(b, b.new_zeros(())).sqrt()

        sa = (sqrt_a / torch.pi).sinc().real
        sb = (sqrt_b / torch.pi).sinc().real
        nsa = (sqrt_a / 2.0).sin().real
        nsb = (sqrt_b / 2.0).sin().real
        ca = sqrt_a.cos().real
        cb = sqrt_b.cos().real

        ba = b - a

        aneqb_a0_limit = (  # At this point we know b != 0
            1.5 * grad_b - grad_a
        ) * sb / b.square() + (
            6.0 * grad_b * nsb.square() + (6.0 - b) * grad_a - 9.0 * grad_b
        ) / (
            6.0 * b.square()
        )
        aneqb_b0_limit = (  # At this point we know a != 0
            1.5 * grad_a - grad_b
        ) * sa / a.square() + (
            6.0 * grad_a * nsa.square() + (6.0 - a) * grad_b - 9.0 * grad_a
        ) / (
            6.0 * a.square()
        )
        a0_xor_b0_limit = torch.where(  # At this point we know only one of a or b is 0
            a == 0, aneqb_a0_limit, aneqb_b0_limit
        )
        a0_and_b0_limit = -(grad_a + grad_b) / 120.0
        a0_or_b0_limit = torch.where(
            (a == 0).logical_and(b == 0), a0_and_b0_limit, a0_xor_b0_limit
        )
        aeqb_limit = (  # At this point we know a != 0 and b != 0
            (grad_b + grad_a) * ((b - 3.0) * sb + 3.0 * cb) / (8.0 * b.square())
        )
        aeqb_or_a0_or_b0_limit = torch.where(
            (b != 0).logical_and(a != 0), aeqb_limit, a0_or_b0_limit
        )

        return torch.where(
            (a != b).logical_and(a != 0).logical_and(b != 0),
            (
                (sa - sb) * (grad_a - grad_b)
                + 0.5 * ba * ((ca - sa) * grad_a / a + (sb - cb) * grad_b / b)
            )
            / ba.square(),
            aeqb_or_a0_or_b0_limit,
        )


class SqrtA2MinusBDivA(torch.autograd.Function):
    """
    Custom autograd function for the compound expression
    `((sqrt(c^2 + g_tilde) - c) / g_tilde)`. The singularity at `g_tilde == 0` is
    replaced by its limit.
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
        c, g_tilde, _ = ctx.saved_tensors

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

    @staticmethod
    def jvp(ctx, grad_c, grad_g_tilde):
        c, g_tilde, _ = ctx.saved_tensors

        return torch.where(
            g_tilde != 0,
            (
                g_tilde
                * (
                    (2.0 * c * grad_c + grad_g_tilde)
                    / (2.0 * (c.square() + g_tilde).sqrt())
                    - grad_c
                )
                - ((c.square() + g_tilde).sqrt() - c) * grad_g_tilde
            )
            / (g_tilde.square()),
            (-grad_g_tilde - 4.0 * c * grad_c) / (8.0 * c.pow(3)),
        )
