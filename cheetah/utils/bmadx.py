import torch

double_precision_epsilon = torch.finfo(torch.float64).eps


def offset_particle_set(
    x_offset: torch.Tensor,
    y_offset: torch.Tensor,
    tilt: torch.Tensor,
    x_lab: torch.Tensor,
    px_lab: torch.Tensor,
    y_lab: torch.Tensor,
    py_lab: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Transforms particle coordinates from lab to element frame.

    :param x_offset: Element x-coordinate offset.
    :param y_offset: Element y-coordinate offset.
    :param tilt: Tilt angle (rad).
    :param x_lab: x-coordinate in lab frame.
    :param px_lab: x-momentum in lab frame.
    :param y_lab: y-coordinate in lab frame.
    :param py_lab: y-momentum in lab frame.
    :return: x, px, y, py coordinates in element frame.
    """
    s = tilt.sin()
    c = tilt.cos()
    x_ele_int = x_lab - x_offset.unsqueeze(-1)
    y_ele_int = y_lab - y_offset.unsqueeze(-1)
    x_ele = x_ele_int * c.unsqueeze(-1) + y_ele_int * s.unsqueeze(-1)
    y_ele = -x_ele_int * s.unsqueeze(-1) + y_ele_int * c.unsqueeze(-1)
    px_ele = px_lab * c.unsqueeze(-1) + py_lab * s.unsqueeze(-1)
    py_ele = -px_lab * s.unsqueeze(-1) + py_lab * c.unsqueeze(-1)

    return x_ele, px_ele, y_ele, py_ele


def offset_particle_unset(
    x_offset: torch.Tensor,
    y_offset: torch.Tensor,
    tilt: torch.Tensor,
    x_ele: torch.Tensor,
    px_ele: torch.Tensor,
    y_ele: torch.Tensor,
    py_ele: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Transforms particle coordinates from element to lab frame.

    :param x_offset: Element x-coordinate offset.
    :param y_offset: Element y-coordinate offset.
    :param tilt: Tilt angle (rad).
    :param x_ele: x-coordinate in element frame.
    :param px_ele: x-momentum in element frame.
    :param y_ele: y-coordinate in element frame.
    :param py_ele: y-momentum in element frame.
    :return: x, px, y, py coordinates in lab frame.
    """
    s = tilt.sin()
    c = tilt.cos()
    x_lab_int = x_ele * c.unsqueeze(-1) - y_ele * s.unsqueeze(-1)
    y_lab_int = x_ele * s.unsqueeze(-1) + y_ele * c.unsqueeze(-1)
    x_lab = x_lab_int + x_offset.unsqueeze(-1)
    y_lab = y_lab_int + y_offset.unsqueeze(-1)
    px_lab = px_ele * c.unsqueeze(-1) - py_ele * s.unsqueeze(-1)
    py_lab = px_ele * s.unsqueeze(-1) + py_ele * c.unsqueeze(-1)

    return x_lab, px_lab, y_lab, py_lab


def low_energy_z_correction(
    pz: torch.Tensor, p0c: torch.Tensor, mc2: torch.Tensor, ds: torch.Tensor
) -> torch.Tensor:
    """
    Corrects the change in z-coordinate due to speed < speed_of_light.

    :param pz: Particle longitudinal momentum.
    :param p0c: Reference particle momentum in eV.
    :param mc2: Particle mass in eV.
    :param ds: Drift length.
    :return: dz=(ds-d_particle) + ds*(beta - beta_ref)/beta_ref
    """
    beta = (
        (1 + pz)
        * p0c.unsqueeze(-1)
        / (((1 + pz) * p0c.unsqueeze(-1)).square() + mc2.square()).sqrt()
    )
    beta0 = p0c / (p0c.square() + mc2.square()).sqrt()
    e_tot = (p0c.square() + mc2.square()).sqrt()

    evaluation = mc2 * (beta0.unsqueeze(-1) * pz).square()
    dz = ds.unsqueeze(-1) * pz * (
        1
        - 3 * (pz * beta0.square().unsqueeze(-1)) / 2
        + pz.square()
        * beta0.square().unsqueeze(-1)
        * (2 * beta0.square().unsqueeze(-1) - (mc2 / e_tot.unsqueeze(-1)).square() / 2)
    ) * (mc2 / e_tot.unsqueeze(-1)).square() * (
        evaluation < 3e-7 * e_tot.unsqueeze(-1)
    ) + (
        ds.unsqueeze(-1) * (beta - beta0.unsqueeze(-1)) / beta0.unsqueeze(-1)
    ) * (
        evaluation >= 3e-7 * e_tot.unsqueeze(-1)
    )

    return dz


def calculate_quadrupole_coefficients(
    k1: torch.Tensor,
    length: torch.Tensor,
    rel_p: torch.Tensor,
    eps: float = double_precision_epsilon,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Returns 2x2 transfer matrix elements aij and the coefficients to calculate the
    change in z position.

    NOTE: Accumulated error due to machine epsilon.

    :param k1: Quadrupole strength (k1 > 0 ==> defocus).
    :param length: Quadrupole length.
    :param rel_p: Relative momentum P/P0.
    :param eps: Machine precision epsilon, default to double precision.
    :return: Tuple of transfer matrix elements and coefficients.
        a11, a12, a21, a22: Transfer matrix elements.
        c1, c2, c3: Second order derivatives of z such that
            z = c1 * x_0^2 + c2 * x_0 * px_0 + c3 * px_0^2.
    """
    # The sign of k1 is flipped compared to track_methods by convention
    kx = (torch.complex(-k1, torch.zeros_like(k1))).sqrt()
    cx = (kx * length.unsqueeze(-1)).cos().real
    sx = ((kx * length.unsqueeze(-1) / torch.pi).sinc() * length.unsqueeze(-1)).real

    a11 = cx
    a12 = sx / rel_p
    a21 = k1 * sx * rel_p
    a22 = cx

    c1 = k1 * (-cx * sx + length.unsqueeze(-1)) / 4
    c2 = -k1 * sx.square() / (2 * rel_p)
    c3 = -(cx * sx + length.unsqueeze(-1)) / (4 * rel_p.square())

    return [[a11, a12], [a21, a22]], [c1, c2, c3]


def sqrt_one(x):
    """Routine to calculate Sqrt[1+x] - 1 to machine precision."""
    sq = (1 + x).sqrt()
    rad = sq + 1

    return x / rad


def track_a_drift(
    length: torch.Tensor,
    x_in: torch.Tensor,
    px_in: torch.Tensor,
    y_in: torch.Tensor,
    py_in: torch.Tensor,
    tau_in: torch.Tensor,
    delta_in: torch.Tensor,
    p0c: torch.Tensor,
    mc2: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact drift tracking in Cheetah coordinates."""
    e_tot = (p0c.square() + mc2.square()).sqrt()
    energy = e_tot.unsqueeze(-1) + delta_in * p0c.unsqueeze(-1)
    p = (energy.square() - mc2.square()).sqrt()
    beta = p / energy
    P = p / p0c.unsqueeze(-1)
    pz = P - 1.0

    Px = px_in / P
    Py = py_in / P
    Pxy2 = Px.square() + Py.square()
    Pl = (1.0 - Pxy2).sqrt()

    dz = length.unsqueeze(-1) * (
        sqrt_one(
            (mc2.square() * (2.0 * pz + pz.square()))
            / ((p0c.unsqueeze(-1) * P).square() + mc2.square())
        )
        + sqrt_one(-Pxy2) / Pl
    )

    x_out = x_in + length.unsqueeze(-1) * Px / Pl
    y_out = y_in + length.unsqueeze(-1) * Py / Pl
    tau_out = tau_in - dz / beta

    return x_out, y_out, tau_out


def sinc(x):
    """sinc(x) = sin(x)/x."""
    return (x / torch.pi).sinc()


def cosc(x):
    """cosc(x) = (cos(x)-1)/x^2 = -1/2 [sinc(x/2)]^2"""
    return -0.5 * sinc(x / 2).square()
