import torch


def log1pdiv(x: torch.Tensor) -> torch.Tensor:
    """Calculate `log(1 + x) / x` with proper removal of its singularity at 0."""
    return Log1PDiv.apply(x)


def si1mdiv(x: torch.Tensor) -> torch.Tensor:
    """Calculate `(1 - si(sqrt(x))) / x` with a Taylor series for `abs(x) <= 1`."""
    small = x.abs() <= 1.0
    regular_x = x.where(~small, 1.0)
    sqrt_x = torch.complex(regular_x, x.new_zeros(())).sqrt()
    sx = (sqrt_x.sin() / sqrt_x).real
    series_x = x.where(small, 0.0)
    coefficients = (
        1.0 / 6.0,
        -1.0 / 120.0,
        1.0 / 5040.0,
        -1.0 / 362880.0,
        1.0 / 39916800.0,
        -1.0 / 6227020800.0,
    )
    series = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        series = series * series_x + coefficient
    return ((1.0 - sx) / regular_x).where(~small, series)


def sicos1mdiv(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(1 - si(sqrt(x)) * cos(sqrt(x))) / x` with proper removal of its
    singularity at 0.
    """
    return SiCos1MDiv.apply(x)


def sipsicos3mdiv(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(3 - 4 * si(sqrt(x)) + si(sqrt(x)) * cos(sqrt(x))) / (2 * x**2)` with
    proper removal of its singularity at 0.
    """
    return SiPSiCos3MDiv.apply(x)


def normalized_j3(x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the normalized cubic horizontal dispersion response integral.

    Calculate `(15 - 22*S + 9*S*C - 2*S*C**2) / (6*x**3)`, where
    `S = si(sqrt(x))` and `C = cos(sqrt(x))`, with limit `1/56` at zero.
    The dimensional integral is `J3 = L**7 * normalized_j3(kx2 * L**2)`.
    """
    return NormalizedJ3.apply(x)


def normalized_jf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evaluate `(F(b) - F(a)) / (a - b)`, with `F = si1mdiv`.

    The body map uses `Jf = L**5 * normalized_jf(kx2*L**2, 4*ky2*L**2)`.

    Two removable singularities need different treatments: the origin has limit
    1/120, whereas nonzero resonance (a == b) has a strength-dependent limit.
    With M = max(abs(a), abs(b)), select one of three branches:

    - M <= 1: (small) use the divided Taylor series
      1/120 - (a + b)/5040 + (a**2 + a*b + b**2)/362880 - ...,
      through total degree five. This avoids cancellation near the origin,
      including when a == b.
    - M > 1 and abs(a - b) <= 0.1*M: (near)use MAD-X's sum/difference-frequency
      identity. Define S(x) = sin(sqrt(x))/sqrt(x), p = (sqrt(a) + sqrt(b))/2,
      and m = (sqrt(a) - sqrt(b))/2. Then
      Q = (sinc(p)*cos(m) - cos(p)*sinc(m))/(2*sqrt(a)*sqrt(b))
      equals (S(b) - S(a))/(a - b), with sinc(t) = sin(t)/t. Evaluate Jf/L**5
      as (F(b) - Q)/a, avoiding division by a - b. At a == b == q != 0,
      this gives (2 + cos(sqrt(q)) - 3*S(q))/(2*q**2), rather than 1/120.
      Complex square roots extend the identity to negative arguments.
    - Otherwise, use the direct quotient (F(b) - F(a))/(a - b).

    The difference-frequency sinc is evaluated as 1 - m**2*F(m**2) so its
    second derivative remains finite at m == 0. Inactive branches receive safe
    placeholder inputs because torch.where evaluates both alternatives.
    """
    a, b = torch.broadcast_tensors(a, b)
    small = torch.maximum(a.abs(), b.abs()) <= 1.0
    near = ~small & ((a - b).abs() <= 0.1 * torch.maximum(a.abs(), b.abs()))
    far = ~(small | near)

    # Series expansion for small arguments
    series_a, series_b = a.where(small, 0.0), b.where(small, 0.0)
    h = power_b = torch.ones_like(a)
    series = h / 120.0
    for coefficient in (
        -1.0 / 5040.0,
        1.0 / 362880.0,
        -1.0 / 39916800.0,
        1.0 / 6227020800.0,
        -1.0 / 1307674368000.0,
    ):
        power_b = power_b * series_b
        h = series_a * h + power_b
        series = series + coefficient * h

    # Direct output
    fa, fb = si1mdiv(a), si1mdiv(b)
    quotient = (fb - fa) / (a - b).where(far, 1.0)

    # Near-resonance, a close to b, but not small, use the sum
    # difference-frequency identity
    near_a, near_b = a.where(near, 1.0), b.where(near, 1.0)
    sqrt_a = torch.complex(near_a, near_a.new_zeros(())).sqrt()
    sqrt_b = torch.complex(near_b, near_b.new_zeros(())).sqrt()
    plus, minus = (sqrt_a + sqrt_b) / 2.0, (sqrt_a - sqrt_b) / 2.0
    sp = (plus.sin() / plus).real
    # Express sinc through F so its second derivative is finite at minus == 0.
    m = minus.real.square() - minus.imag.square()
    sm = 1.0 - m * si1mdiv(m)
    # Real-component forms avoid unresolved conjugates in batched forward AD.
    cp = plus.real.cos() * plus.imag.cosh()
    cm = minus.real.cos() * minus.imag.cosh()
    denominator = 2.0 * (near_a * near_b).sqrt() * near_a.sign()
    js = (sp * cm - cp * sm) / denominator
    resonant = (fb - js) / near_a
    return quotient.where(far, resonant).where(~small, series)


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


def si2msi2divdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(si^2(sqrt(b)) - si^2(sqrt(a))) / (a - b)` with proper removal of its
    singularity at `a == b`.

    NOTE: Forward mode AD currently doesn't work properly for a==0 or b==0, and might
        return weird gradients in those cases.
    """
    return Si2MSi2DivDiff.apply(a, b)


def sqrta2minusbdiva(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate `(sqrt(a^2 + b) - a) / b` with proper removal of its singularity at
    `b == 0`.
    """
    return SqrtA2MinusBDivA.apply(a, b)


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
        return (x.log1p() / x).where(x != 0, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        x, fx = ctx.saved_tensors
        return grad_output * (((1.0 + x).reciprocal() - fx) / x).where(x != 0, -0.5)

    @staticmethod
    def jvp(ctx, grad_input):
        x, fx = ctx.saved_tensors
        return (((1.0 + x).reciprocal() - fx) / x).where(x != 0, -0.5) * grad_input


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

        return ((1.0 - sx * cx) / x).where(x != 0, 1.0 / 6.0)

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = (
            (sx * (x * sx + 2.0 * cx) - 2.0 - cx.square() + sx * cx)
            / (2.0 * x.square())
        ).where(x != 0, -2.0 / 15.0)

        return grad_output * grad

    @staticmethod
    def jvp(ctx, grad_input):
        x, _ = ctx.saved_tensors

        sqrt_x = torch.complex(x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = (
            (sx * (x * sx + 2.0 * cx) - 2.0 - cx.square() + sx * cx)
            / (2.0 * x.square())
        ).where(x != 0, -2.0 / 15.0)

        return grad * grad_input


class SiPSiCos3MDiv(torch.autograd.Function):
    r"""
    Custom autograd function for :math:`H(a)`, where ``a`` is the input ``x``:

    .. math::
        H(a) = \frac{3 - 4S(a) + S(a)C(a)}{2a^2}, \qquad
        S(a) = \frac{\sin\sqrt{a}}{\sqrt{a}}, \quad C(a) = \cos\sqrt{a}.

    Near zero, evaluate the Taylor series to avoid cancellation:

    .. math::
        H(a) = \frac{1}{20} - \frac{a}{168} + \frac{a^2}{2880}
        - \frac{17a^3}{1330560} + \frac{31a^4}{94348800}
        - \frac{a^5}{159667200} + O(a^6).

    Thus :math:`H(0) = 1/20` and :math:`H'(0) = -1/168`.
    The second-order map uses :math:`J_2 = L^5 H(k_x^2 L^2)`.
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
        # Float32 needs a wider series interval to avoid cancellation.
        small = x.abs() <= (0.1 if x.dtype == torch.float64 else 1.0)
        regular_x = x.where(~small, 1.0)
        # Since x may be negative, we use complex arithmetic for the sqrt
        sqrt_x = torch.complex(regular_x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        # ponytail: degree 5 covers these intervals; revalidate before widening them.
        series_x = x.where(small, 0.0)
        series = 1.0 / 20.0 + series_x * (
            -1.0 / 168.0
            + series_x
            * (
                1.0 / 2880.0
                + series_x
                * (
                    -17.0 / 1330560.0
                    + series_x * (31.0 / 94348800.0 - series_x / 159667200.0)
                )
            )
        )
        return ((3.0 - 4.0 * sx + sx * cx) / (2.0 * regular_x.square())).where(
            ~small, series
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors

        small = x.abs() <= (0.1 if x.dtype == torch.float64 else 1.0)
        regular_x = x.where(~small, 1.0)
        sqrt_x = torch.complex(regular_x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = (
            cx.square()
            - 5.0 * sx * cx
            - 4.0 * cx
            + 20.0 * sx
            - 12.0
            - regular_x * sx.square()
        ) / (4.0 * regular_x.pow(3))
        series_x = x.where(small, 0.0)
        series = -1.0 / 168.0 + series_x * (
            1.0 / 1440.0
            + series_x
            * (
                -17.0 / 443520.0
                + series_x * (31.0 / 23587200.0 - series_x / 31933440.0)
            )
        )
        grad = grad.where(~small, series)

        return grad_output * grad

    @staticmethod
    def jvp(ctx, grad_input):
        x, _ = ctx.saved_tensors

        small = x.abs() <= (0.1 if x.dtype == torch.float64 else 1.0)
        regular_x = x.where(~small, 1.0)
        sqrt_x = torch.complex(regular_x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        grad = (
            cx.square()
            - 5.0 * sx * cx
            - 4.0 * cx
            + 20.0 * sx
            - 12.0
            - regular_x * sx.square()
        ) / (4.0 * regular_x.pow(3))
        series_x = x.where(small, 0.0)
        series = -1.0 / 168.0 + series_x * (
            1.0 / 1440.0
            + series_x
            * (
                -17.0 / 443520.0
                + series_x * (31.0 / 23587200.0 - series_x / 31933440.0)
            )
        )
        grad = grad.where(~small, series)

        return grad * grad_input


class NormalizedJ3(torch.autograd.Function):
    r"""Dimensionless J3 integral :math:`H(a)`, with :math:`J_3 = L^7 H(KL^2)`.

    .. math::
        H(a) = \frac{15 - 22S(a) + 9S(a)C(a) - 2S(a)C(a)^2}{6a^3},
        \qquad S(a) = \frac{\sin\sqrt{a}}{\sqrt{a}},\quad C(a) = \cos\sqrt{a}.

    The near-zero series starts at :math:`1/56 - a/288 + 7a^2/21120`.
    Use its degree-seven polynomial for :math:`|a| \leq 1` in both precisions
    to avoid cancellation near zero.
    Differentiating :math:`J_3 = \int_0^L d(s)^3 ds` with respect to length gives
    :math:`H'(a) = (D(a)^3 - 7H(a))/(2a)`, where
    :math:`D(a) = (1-C(a))/a = \tfrac12\mathop{\text{sinc}}^2(\sqrt{a}/2)`.
    """

    generate_vmap_rule = True

    _coefficients = (
        1.0 / 56.0,
        -1.0 / 288.0,
        7.0 / 21120.0,
        -1.0 / 49140.0,
        13.0 / 14515200.0,
        -31.0 / 1034035200.0,
        63047.0 / 79506601574400.0,
        -1069.0 / 62768369664000.0,
    )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs

        ctx.save_for_backward(x, output)
        ctx.save_for_forward(x, output)

    @staticmethod
    def forward(x):
        small = x.abs() <= 1.0
        regular_x = x.where(~small, 1.0)
        sqrt_x = torch.complex(regular_x, x.new_zeros(())).sqrt()

        cx = sqrt_x.cos().real
        sx = (sqrt_x / torch.pi).sinc().real

        series_x = x.where(small, 0.0)
        coefficients = NormalizedJ3._coefficients
        series = coefficients[-1]
        for coefficient in reversed(coefficients[:-1]):
            series = series * series_x + coefficient
        return (
            (15.0 - 22.0 * sx + 9.0 * sx * cx - 2.0 * sx * cx.square())
            / (6.0 * regular_x.pow(3))
        ).where(~small, series)

    @staticmethod
    def _derivative(x, fx):
        small = x.abs() <= 1.0
        regular_x = x.where(~small, 1.0)
        sqrt_x = torch.complex(regular_x, x.new_zeros(())).sqrt()
        d = 0.5 * (sqrt_x / (2.0 * torch.pi)).sinc().real.square()
        grad = (d.pow(3) - 7.0 * fx) / (2.0 * regular_x)

        series_x = x.where(small, 0.0)
        coefficients = NormalizedJ3._coefficients
        series = (len(coefficients) - 1) * coefficients[-1]
        for n in range(len(coefficients) - 2, 0, -1):
            series = series * series_x + n * coefficients[n]
        return grad.where(~small, series)

    @staticmethod
    def backward(ctx, grad_output):
        x, fx = ctx.saved_tensors
        return grad_output * NormalizedJ3._derivative(x, fx)

    @staticmethod
    def jvp(ctx, grad_input):
        x, fx = ctx.saved_tensors
        return NormalizedJ3._derivative(x, fx) * grad_input


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
        a, b = inputs  # inputs is always passed as a tuple

        ctx.save_for_backward(a, b, output)
        ctx.save_for_forward(a, b, output)

    @staticmethod
    def forward(a, b):
        sqrt_a = torch.complex(a, a.new_zeros(())).sqrt()
        sqrt_b = torch.complex(b, b.new_zeros(())).sqrt()

        sa = (sqrt_a / torch.pi).sinc().real
        ca = sqrt_a.cos().real
        cb = sqrt_b.cos().real

        return ((cb - ca) / (a - b)).where(a != b, 0.5 * sa)

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

        limit = ((ca - sa) / (8.0 * a)).where(a != 0, -1.0 / 24.0)

        grad_a = ((0.5 * sa * ab - cbca) / demoninator).where(a != b, limit)
        grad_b = (-(0.5 * sb * ab - cbca) / demoninator).where(a != b, limit)

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

        return (
            ((0.5 * ab * (grad_a * sa - grad_b * sb)) + (grad_a - grad_b) * (ca - cb))
            / ab.square()
        ).where(
            a != b,
            (((grad_b + grad_a) * cb - (grad_b + grad_a) * sb) / (8.0 * b)).where(
                b != 0, -(grad_a + grad_b) / 24.0
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
        a, b = inputs

        ctx.save_for_backward(a, b, output)
        ctx.save_for_forward(a, b, output)

    @staticmethod
    def forward(a, b):
        sqrt_a = torch.complex(a, a.new_zeros(())).sqrt()
        sqrt_b = torch.complex(b, b.new_zeros(())).sqrt()

        sa = (sqrt_a / torch.pi).sinc().real
        sb = (sqrt_b / torch.pi).sinc().real
        cb = sqrt_b.cos().real

        return ((sa - sb) / (b - a)).where(
            a != b, (0.5 * (sb - cb) / b).where(b != 0, 1.0 / 6.0)
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
        aeqb_limit = ((3.0 * cb + (b - 3.0) * sb) / (8.0 * b.square())).where(
            b != 0, a0_b0_limit
        )
        aneqb_a0_limit = (1.0 - b / 6.0 - sb) / b.square()
        aneqb_b0_limit = (1.0 - a / 6.0 - sa) / a.square()

        grad_a = ((ca - sa) / (2.0 * a * ba) + (sa - sb) / ba.square()).where(
            (a != b).logical_and(a != 0), aneqb_a0_limit.where(a != b, aeqb_limit)
        )
        grad_b = (-(cb - sb) / (2.0 * b * ba) + (sb - sa) / ba.square()).where(
            (a != b).logical_and(b != 0), aneqb_b0_limit.where(a != b, aeqb_limit)
        )

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

        ba = b - a

        aneqb_a0_limit = (  # At this point we know b != 0
            1.5 * grad_b - grad_a
        ) * sb / b.square() + (
            6.0 * grad_b * (1.0 - cb) / 2.0 + (6.0 - b) * grad_a - 9.0 * grad_b
        ) / (
            6.0 * b.square()
        )
        aneqb_b0_limit = (  # At this point we know a != 0
            1.5 * grad_a - grad_b
        ) * sa / a.square() + (
            6.0 * grad_a * (1.0 - ca) / 2.0 + (6.0 - a) * grad_b - 9.0 * grad_a
        ) / (
            6.0 * a.square()
        )
        a0_xor_b0_limit = (  # At this point we know only one of a or b is 0
            aneqb_a0_limit.where(a == 0, aneqb_b0_limit)
        )
        a0_and_b0_limit = -(grad_a + grad_b) / 120.0
        a0_or_b0_limit = a0_and_b0_limit.where(
            (a == 0).logical_and(b == 0), a0_xor_b0_limit
        )
        aeqb_limit = (  # At this point we know a != 0 and b != 0
            (grad_b + grad_a) * ((b - 3.0) * sb + 3.0 * cb) / (8.0 * b.square())
        )
        aeqb_or_a0_or_b0_limit = aeqb_limit.where(
            (b != 0).logical_and(a != 0), a0_or_b0_limit
        )

        return (
            (
                (sa - sb) * (grad_a - grad_b)
                + 0.5 * ba * ((ca - sa) * grad_a / a + (sb - cb) * grad_b / b)
            )
            / ba.square()
        ).where(
            (a != b).logical_and(a != 0).logical_and(b != 0), aeqb_or_a0_or_b0_limit
        )


class Si2MSi2DivDiff(torch.autograd.Function):
    """
    Custom autograd function for the compound expression
    `(si^2(sqrt(b)) - si^2(sqrt(a))) / (a - b)`. The singularity at`a == b` is replaced
    by its limit.

    NOTE: Forward mode AD currently doesn't work properly for a==0 or b==0, and might
        return weird gradients in those cases.
    """

    # Automatically generate a custom vmap implementation
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b = inputs

        ctx.save_for_backward(a, b, output)
        ctx.save_for_forward(a, b, output)

    @staticmethod
    def forward(a, b):
        sqrt_a = torch.complex(a, a.new_zeros(())).sqrt()
        sqrt_b = torch.complex(b, b.new_zeros(())).sqrt()

        sa = (sqrt_a / torch.pi).sinc().real
        sb = (sqrt_b / torch.pi).sinc().real
        cb = sqrt_b.cos().real

        aeqb_limit = ((1.0 - cb.square() - b * sb * cb) / b.square()).where(
            b != 0, 1.0 / 3.0
        )

        return ((sb.square() - sa.square()) / (a - b)).where(a != b, aeqb_limit)

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

        a0_limit = (b - 1 / 3 * b.square() + cb.square() - 1.0) / b.pow(3)
        b0_limit = (a - 1 / 3 * a.square() + ca.square() - 1.0) / a.pow(3)
        aeqbeq0_limit = -2.0 / 45.0
        aeqb_limit = (
            (5.0 * b * sb * cb - (b - 2.0) * (2.0 * cb.square() - 1.0) - 2.0)
            / (4.0 * b.pow(3))
        ).where(b != 0, aeqbeq0_limit)
        aeqb_or_a0_limit = aeqb_limit.where(a == b, a0_limit)
        aeqb_or_b0_limit = aeqb_limit.where(a == b, b0_limit)

        grad_a = (
            (-ab * sa * (ca - sa) / a + sa.square() - sb.square()) / ab.square()
        ).where((a != b).logical_and(a != 0), aeqb_or_a0_limit)
        grad_b = (
            (ab * sb * (cb - sb) / b + sb.square() - sa.square()) / ab.square()
        ).where((a != b).logical_and(b != 0), aeqb_or_b0_limit)

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

        # NOTE: There exists no two-sided limit for a==0 and b==0. The commented out
        # lines are left here for documentation purposes. This may be fixed in the
        # future.

        # a0_limit = torch.nan
        # b0_limit = torch.nan
        # a0_or_b0_limit = a0_limit.where(a == 0, b0_limit)
        aeqbeq0_limit = -2.0 / 45.0 * (grad_a + grad_b)
        aeqb_limit = (
            (sb * cb * 5.0 - cb.square() + (b - 4.0) * sb.square())
            * (grad_a + grad_b)
            / (4.0 * b.square())
        ).where(b != 0, aeqbeq0_limit)
        # aeqb_or_a0_or_b0_limit = aeqb_limit.where(a == 0, a0_or_b0_limit)

        return (
            (
                (sa.square() - sb.square()) * (grad_a - grad_b)
                + ab * (sb * grad_b * (cb - sb) / b - sa * grad_a * (ca - sa) / a)
            )
            / ab.square()
        ).where(a != b, aeqb_limit)


class SqrtA2MinusBDivA(torch.autograd.Function):
    """
    Custom autograd function for the compound expression
    `((sqrt(a^2 + b) - a) / b)`. The singularity at `b == 0` is replaced by its limit.
    """

    # Automatically generate a custom vmap implementation
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b = inputs

        ctx.save_for_backward(a, b, output)
        ctx.save_for_forward(a, b, output)

    @staticmethod
    def forward(a, b):
        return (((a.square() + b).sqrt() - a) / b).where(b != 0, (2.0 * a).reciprocal())

    @staticmethod
    def backward(ctx, grad_output):
        a, b, _ = ctx.saved_tensors

        grad_a = ((a / (a.square() + b).sqrt() - 1) / b).where(
            b != 0, -(2.0 * a.square()).reciprocal()
        )
        grad_b = (
            ((-2.0 * a.square() - b) / (a.square() + b).sqrt() + 2.0 * a)
            / (2.0 * b.square())
        ).where(b != 0, -(8.0 * a.pow(3)).reciprocal())

        return grad_output * grad_a, grad_output * grad_b

    @staticmethod
    def jvp(ctx, grad_a, grad_b):
        a, b, _ = ctx.saved_tensors

        return (
            (
                b
                * (
                    (2.0 * a * grad_a + grad_b) / (2.0 * (a.square() + b).sqrt())
                    - grad_a
                )
                - ((a.square() + b).sqrt() - a) * grad_b
            )
            / (b.square())
        ).where(b != 0, (-grad_b - 4.0 * a * grad_a) / (8.0 * a.pow(3)))
