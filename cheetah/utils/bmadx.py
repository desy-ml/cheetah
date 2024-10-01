import torch
from scipy.constants import speed_of_light

double_precision_epsilon = torch.finfo(torch.float64).eps


def cheetah_to_bmad_z_pz(
    tau: torch.Tensor, delta: torch.Tensor, ref_energy: torch.Tensor, mc2: float
) -> torch.Tensor:
    """
    Transforms Cheetah longitudinal coordinates to Bmad coordinates
    and computes p0c.

    :param tau: Cheetah longitudinal coordinate (c*delta_t).
    :param delta: Cheetah longitudinal momentum (delta_E/p0c).
    :param ref_energy: Reference energy in eV.
    :param mc2: Particle mass in eV/c^2.
    """
    # TODO This can probably be moved to the `ParticleBeam` class at some point

    # Compute p0c and Bmad z, pz
    p0c = torch.sqrt(ref_energy**2 - mc2**2)
    energy = ref_energy.unsqueeze(-1) + delta * p0c.unsqueeze(-1)
    p = torch.sqrt(energy**2 - mc2**2)
    beta = p / energy
    z = -beta * tau
    pz = (p - p0c.unsqueeze(-1)) / p0c.unsqueeze(-1)

    return z, pz, p0c


def bmad_to_cheetah_z_pz(
    z: torch.Tensor, pz: torch.Tensor, p0c: torch.Tensor, mc2: float
) -> tuple[torch.Tensor]:
    """
    Transforms Bmad longitudinal coordinates to Cheetah coordinates
    and computes reference energy.

    :param z: Bmad longitudinal coordinate (c*delta_t).
    :param pz: Bmad longitudinal momentum (delta_E/p0c).
    :param p0c: Reference momentum in eV/c.
    :param mc2: Particle mass in eV/c^2.
    """
    # TODO This can probably be moved to the `ParticleBeam` class at some point

    # Compute ref_energy and Cheetah tau, delta
    ref_energy = torch.sqrt(p0c**2 + mc2**2)
    p = (1 + pz) * p0c.unsqueeze(-1)
    energy = torch.sqrt(p**2 + mc2**2)
    beta = p / energy
    tau = -z / beta
    delta = (energy - ref_energy.unsqueeze(-1)) / p0c.unsqueeze(-1)

    return tau, delta, ref_energy


def cheetah_to_bmad_coords(
    cheetah_coords: torch.Tensor, ref_energy: torch.Tensor, mc2: torch.Tensor
) -> torch.Tensor:
    """
    Transforms Cheetah coordinates to Bmad coordinates.

    :param cheetah_coords: 7-dimensional particle vectors in Cheetah coordinates.
    :param ref_energy: Reference energy in eV.
    """
    # TODO This can probably be moved to the `ParticleBeam` class at some point

    # Initialize Bmad coordinates
    bmad_coords = cheetah_coords[..., :6].clone()

    # Cheetah longitudinal coordinates
    tau = cheetah_coords[..., 4]
    delta = cheetah_coords[..., 5]

    # Compute p0c and Bmad z, pz
    z, pz, p0c = cheetah_to_bmad_z_pz(tau, delta, ref_energy, mc2)

    # Bmad coordinates
    bmad_coords[..., 4] = z
    bmad_coords[..., 5] = pz

    return bmad_coords, p0c


def bmad_to_cheetah_coords(
    bmad_coords: torch.Tensor, p0c: torch.Tensor, mc2: torch.Tensor
) -> torch.Tensor:
    """
    Transforms Bmad coordinates to Cheetah coordinates.

    :param bmad_coords: 6-dimensional particle vectors in Bmad coordinates.
    :param p0c: Reference momentum in eV/c.
    """
    # TODO This can probably be moved to the `ParticleBeam` class at some point

    # Initialize Cheetah coordinates
    cheetah_coords = torch.ones(
        (*bmad_coords.shape[:-1], 7), dtype=bmad_coords.dtype, device=bmad_coords.device
    )
    cheetah_coords[..., :6] = bmad_coords.clone()

    # Bmad longitudinal coordinates
    z = bmad_coords[..., 4]
    pz = bmad_coords[..., 5]

    # Compute ref_energy and Cheetah tau, delta
    tau, delta, ref_energy = bmad_to_cheetah_z_pz(z, pz, p0c, mc2)

    # Cheetah coordinates
    cheetah_coords[..., 4] = tau
    cheetah_coords[..., 5] = delta

    return cheetah_coords, ref_energy


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
    s = torch.sin(tilt)
    c = torch.cos(tilt)
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
    s = torch.sin(tilt)
    c = torch.cos(tilt)
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
        / torch.sqrt(((1 + pz) * p0c.unsqueeze(-1)) ** 2 + mc2**2)
    )
    beta0 = p0c / torch.sqrt(p0c**2 + mc2**2)
    e_tot = torch.sqrt(p0c**2 + mc2**2)

    evaluation = mc2 * (beta0.unsqueeze(-1) * pz) ** 2
    dz = ds.unsqueeze(-1) * pz * (
        1
        - 3 * (pz * beta0.unsqueeze(-1) ** 2) / 2
        + pz**2
        * beta0.unsqueeze(-1) ** 2
        * (2 * beta0.unsqueeze(-1) ** 2 - (mc2 / e_tot.unsqueeze(-1)) ** 2 / 2)
    ) * (mc2 / e_tot.unsqueeze(-1)) ** 2 * (evaluation < 3e-7 * e_tot.unsqueeze(-1)) + (
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
    # TODO: Revisit to fix accumulated error due to machine epsilon
    sqrt_k = torch.sqrt(torch.absolute(k1) + eps)
    sk_l = sqrt_k * length.unsqueeze(-1)

    cx = torch.cos(sk_l) * (k1 <= 0) + torch.cosh(sk_l) * (k1 > 0)
    sx = (torch.sin(sk_l) / (sqrt_k)) * (k1 <= 0) + (torch.sinh(sk_l) / (sqrt_k)) * (
        k1 > 0
    )

    a11 = cx
    a12 = sx / rel_p
    a21 = k1 * sx * rel_p
    a22 = cx

    c1 = k1 * (-cx * sx + length.unsqueeze(-1)) / 4
    c2 = -k1 * sx**2 / (2 * rel_p)
    c3 = -(cx * sx + length.unsqueeze(-1)) / (4 * rel_p**2)

    return [[a11, a12], [a21, a22]], [c1, c2, c3]


def sqrt_one(x):
    """Routine to calculate Sqrt[1+x] - 1 to machine precision."""
    sq = torch.sqrt(1 + x)
    rad = sq + 1

    return x / rad


def track_a_drift(
    length: torch.Tensor,
    x_in: torch.Tensor,
    px_in: torch.Tensor,
    y_in: torch.Tensor,
    py_in: torch.Tensor,
    z_in: torch.Tensor,
    pz_in: torch.Tensor,
    p0c: torch.Tensor,
    mc2: torch.Tensor,
) -> tuple[torch.Tensor]:
    """Exact drift tracking used in different elements."""

    P = 1.0 + pz_in  # Particle's total momentum over p0
    Px = px_in / P  # Particle's 'x' momentum over p0
    Py = py_in / P  # Particle's 'y' momentum over p0
    Pxy2 = Px**2 + Py**2  # Particle's transverse mometum^2 over p0^2
    Pl = torch.sqrt(1.0 - Pxy2)  # Particle's longitudinal momentum over p0

    # z = z + L * ( beta / beta_ref - 1.0 / Pl ) but numerically accurate:
    dz = length.unsqueeze(-1) * (
        sqrt_one(
            (mc2**2 * (2 * pz_in + pz_in**2)) / ((p0c.unsqueeze(-1) * P) ** 2 + mc2**2)
        )
        + sqrt_one(-Pxy2) / Pl
    )

    x_out = x_in + length.unsqueeze(-1) * Px / Pl
    y_out = y_in + length.unsqueeze(-1) * Py / Pl
    z_out = z_in + dz

    return x_out, y_out, z_out


def particle_rf_time(z, pz, p0c, mc2):
    """Returns rf time of Particle p."""
    beta = (
        (1 + pz)
        * p0c.unsqueeze(-1)
        / torch.sqrt(((1 + pz) * p0c.unsqueeze(-1)) ** 2 + mc2**2)
    )
    time = -z / (beta * speed_of_light)

    return time


def sinc(x):
    """sinc(x) = sin(x)/x."""
    return torch.sinc(x / torch.pi)


def cosc(x):
    """cosc(x) = (cos(x)-1)/x**2 = -1/2 [sinc(x/2)]**2"""
    return -0.5 * sinc(x / 2) ** 2
