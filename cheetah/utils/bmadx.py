import numpy as np
import torch


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
    p0c = torch.sqrt(ref_energy**2 - mc2**2)
    energy = ref_energy.unsqueeze(-1) + delta * p0c.unsqueeze(-1)
    p = torch.sqrt(energy**2 - mc2**2)
    beta = p / energy
    z = -beta * tau
    pz = (p - p0c.unsqueeze(-1)) / p0c.unsqueeze(-1)

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
    ref_energy = torch.sqrt(p0c**2 + mc2**2)
    p = (1 + pz) * p0c.unsqueeze(-1)
    energy = torch.sqrt(p**2 + mc2**2)
    beta = p / energy
    tau = -z / beta
    delta = (energy - ref_energy.unsqueeze(-1)) / p0c.unsqueeze(-1)

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
    Corrects the change in z-coordinate due to speed < c_light.

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
    eps: float = np.finfo(np.float64).eps,
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
