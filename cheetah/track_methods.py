"""Utility functions for creating transfer maps for elements."""

import torch

from cheetah.particles import Species
from cheetah.utils import compute_relativistic_factors


def base_rmatrix(
    length: torch.Tensor,
    k1: torch.Tensor,
    hx: torch.Tensor,
    species: Species,
    tilt: torch.Tensor | None = None,
    energy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create a first order universal transfer map for a beamline element.

    :param length: Length of the element in m.
    :param k1: Quadrupole strength in 1/m**2.
    :param hx: Curvature (1/radius) of the element in 1/m.
    :param species: Particle species of the beam.
    :param tilt: Roation of the element relative to the longitudinal axis in rad.
    :param energy: Beam energy in eV.
    :return: First order transfer map for the element.
    """
    zero = length.new_zeros(())

    if tilt is None:
        tilt = zero
    if energy is None:
        energy = species.mass_eV

    _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)
    ibeta2 = torch.reciprocal(torch.square(beta))

    kx2 = k1 + hx * hx
    ky2 = -k1
    kx = torch.sqrt(torch.complex(kx2, zero))
    ky = torch.sqrt(torch.complex(ky2, zero))
    kLx = kx * length
    kLy = ky * length
    cx = torch.cos(kLx).real
    cy = torch.cos(kLy).real
    kLxpi = kLx / torch.pi
    sx = (torch.sinc(kLxpi) * length).real
    sy = (torch.sinc(kLy / torch.pi) * length).real

    r = torch.sinc(0.5 * kLxpi)
    dx = hx * 0.5 * length * length * (r * r).real

    kx2_is_not_zero = kx2 != 0
    r56 = (
        torch.addcmul(
            torch.where(kx2_is_not_zero, torch.square(hx) * (length - sx) / kx2, zero),
            length,
            igamma2,
            value=-1,
        )
        * ibeta2
    )

    dx_ibeta2 = dx * ibeta2
    sx_hx_ibeta2 = sx * hx * ibeta2

    cx, sx, dx, cy, sy, r56, dx_ibeta2, sx_hx_ibeta2 = torch.broadcast_tensors(
        cx, sx, dx, cy, sy, r56, dx_ibeta2, sx_hx_ibeta2
    )

    R = torch.eye(7, dtype=cx.dtype, device=cx.device).expand(*cx.shape, 7, 7).clone()
    R[
        ...,
        (0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4),
        (0, 1, 5, 0, 1, 5, 2, 3, 2, 3, 0, 1, 5),
    ] = torch.stack(
        [
            cx,
            sx,
            dx_ibeta2,
            -kx2 * sx,
            cx,
            sx_hx_ibeta2,
            cy,
            sy,
            -ky2 * sy,
            cy,
            sx_hx_ibeta2,
            dx_ibeta2,
            r56,
        ],
        dim=-1,
    )

    rotation = rotation_matrix(tilt)
    R = rotation.mT @ R @ rotation

    return R


def base_ttensor(
    length: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    hx: torch.Tensor,
    species: Species,
    tilt: torch.Tensor | None = None,
    energy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create a second order universal transfer map for a beamline element. Uses MAD
    convention.

    :param length: Length of the element in m.
    :param k1: Quadrupole strength in 1/m**2.
    :param k2: Sextupole strength in 1/m**3.
    :param hx: Curvature (1/radius) of the element in 1/m.
    :param species: Particle species of the beam.
    :param tilt: Roation of the element relative to the longitudinal axis in rad.
    :param energy: Beam energy in eV.
    :return: Second order transfer map for the element.
    """
    device = length.device
    dtype = length.dtype

    zero = torch.tensor(0.0, device=device, dtype=dtype)

    tilt = tilt if tilt is not None else zero
    energy = energy if energy is not None else zero

    _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

    kx2 = k1 + hx**2
    ky2 = -k1
    kx = torch.sqrt(torch.complex(kx2, zero))
    ky = torch.sqrt(torch.complex(ky2, zero))
    cx = torch.cos(kx * length).real
    cy = torch.cos(ky * length).real
    sx = (torch.sinc(kx * length / torch.pi) * length).real
    sy = (torch.sinc(ky * length / torch.pi) * length).real
    dx = torch.where(kx != 0, (1.0 - cx) / kx2, length**2 / 2.0)

    d2y = 0.5 * sy**2
    s2y = sy * cy
    c2y = torch.cos(2 * ky * length).real
    fx = torch.where(kx2 != 0, (length - sx) / kx2, length**3 / 6.0)
    f2y = torch.where(ky2 != 0, (length - s2y) / ky2, length**3 / 6.0)

    j1 = torch.where(kx2 != 0, (length - sx) / kx2, length**3 / 6.0)
    j2 = torch.where(
        kx2 != 0,
        (3.0 * length - 4.0 * sx + sx * cx) / (2 * kx2**2),
        length**5 / 20.0,
    )
    j3 = torch.where(
        kx2 != 0,
        (15.0 * length - 22.5 * sx + 9.0 * sx * cx - 1.5 * sx * cx**2 + kx2 * sx**3)
        / (6.0 * kx2**3),
        length**7 / 56.0,
    )
    j_denominator = kx2 - 4 * ky2
    jc = torch.where(j_denominator != 0, (c2y - cx) / j_denominator, 0.5 * length**2)
    js = torch.where(
        j_denominator != 0, (cy * sy - sx) / j_denominator, length**3 / 6.0
    )
    jd = torch.where(j_denominator != 0, (d2y - dx) / j_denominator, length**4 / 24.0)
    jf = torch.where(j_denominator != 0, (f2y - fx) / j_denominator, length**5 / 120.0)

    khk = k2 + 2 * hx * k1

    vector_shape = torch.broadcast_shapes(
        length.shape, k1.shape, k2.shape, hx.shape, tilt.shape, energy.shape
    )

    T = torch.zeros((7, 7, 7), dtype=dtype, device=device).repeat(
        *vector_shape, 1, 1, 1
    )
    T[..., 0, 0, 0] = -1 / 6 * khk * (sx**2 + dx) - 0.5 * hx * kx2 * sx**2
    T[..., 0, 0, 1] = 2 * (-1 / 6 * khk * sx * dx + 0.5 * hx * sx * cx)
    T[..., 0, 1, 1] = -1 / 6 * khk * dx**2 + 0.5 * hx * dx * cx
    T[..., 0, 0, 5] = 2 * (
        -hx / 12 / beta * khk * (3 * sx * j1 - dx**2)
        + 0.5 * hx**2 / beta * sx**2
        + 0.25 / beta * k1 * length * sx
    )
    T[..., 0, 1, 5] = 2 * (
        -hx / 12 / beta * khk * (sx * dx**2 - 2 * cx * j2)
        + 0.25 * hx**2 / beta * (sx * dx + cx * j1)
        - 0.25 / beta * (sx + length * cx)
    )
    T[..., 0, 5, 5] = (
        -(hx**2) / 6 / beta**2 * khk * (dx**2 * dx - 2 * sx * j2)
        + 0.5 * hx**3 / beta**2 * sx * j1
        - 0.5 * hx / beta**2 * length * sx
        - 0.5 * hx / (beta**2) * igamma2 * dx
    )
    T[..., 0, 2, 2] = k1 * k2 * jd + 0.5 * (k2 + hx * k1) * dx
    T[..., 0, 2, 3] = 2 * (0.5 * k2 * js)
    T[..., 0, 3, 3] = k2 * jd - 0.5 * hx * dx
    T[..., 1, 0, 0] = -1 / 6 * khk * sx * (1 + 2 * cx)
    T[..., 1, 0, 1] = 2 * (-1 / 6 * khk * dx * (1 + 2 * cx))
    T[..., 1, 1, 1] = -1 / 3 * khk * sx * dx - 0.5 * hx * sx
    T[..., 1, 0, 5] = 2 * (
        -hx / 12 / beta * khk * (3 * cx * j1 + sx * dx)
        - 0.25 / beta * k1 * (sx - length * cx)
    )
    T[..., 1, 1, 5] = 2 * (
        -hx / 12 / beta * khk * (3 * sx * j1 + dx**2) + 0.25 / beta * k1 * length * sx
    )
    T[..., 1, 5, 5] = (
        -(hx**2) / 6 / beta**2 * khk * (sx * dx**2 - 2 * cx * j2)
        - 0.5 * hx / beta**2 * k1 * (cx * j1 - sx * dx)
        - 0.5 * hx / beta**2 * igamma2 * sx
    )
    T[..., 1, 2, 2] = k1 * k2 * js + 0.5 * (k2 + hx * k1) * sx
    T[..., 1, 2, 3] = 2 * (0.5 * k2 * jc)
    T[..., 1, 3, 3] = k2 * js - 0.5 * hx * sx
    T[..., 2, 0, 2] = 2 * (
        0.5 * k2 * (cy * jc - 2 * k1 * sy * js) + 0.5 * hx * k1 * sx * sy
    )
    T[..., 2, 0, 3] = 2 * (0.5 * k2 * (sy * jc - 2 * cy * js) + 0.5 * hx * sx * cy)
    T[..., 2, 1, 2] = 2 * (
        0.5 * k2 * (cy * js - 2 * k1 * sy * jd) + 0.5 * hx * k1 * dx * sy
    )
    T[..., 2, 1, 3] = 2 * (0.5 * k2 * (sy * js - 2 * cy * jd) + 0.5 * hx * dx * cy)
    T[..., 2, 2, 5] = 2 * (
        0.5 * hx / beta * k2 * (cy * jd - 2 * k1 * sy * jf)
        + 0.5 * hx**2 / beta * k1 * j1 * sy
        - 0.25 / beta * k1 * length * sy
    )
    T[..., 2, 3, 5] = 2 * (
        0.5 * hx / beta * k2 * (sy * jd - 2 * cy * jf)
        + 0.5 * hx**2 / beta * j1 * cy
        - 0.25 / beta * (sy + length * cy)
    )
    T[..., 3, 0, 2] = 2 * (
        0.5 * k1 * k2 * (2 * cy * js - sy * jc) + 0.5 * (k2 + hx * k1) * sx * cy
    )
    T[..., 3, 0, 3] = 2 * (
        0.5 * k2 * (2 * k1 * sy * js - cy * jc) + 0.5 * (k2 + hx * k1) * sx * sy
    )
    T[..., 3, 1, 2] = 2 * (
        0.5 * k1 * k2 * (2 * cy * jd - sy * js) + 0.5 * (k2 + hx * k1) * dx * cy
    )
    T[..., 3, 1, 3] = 2 * (
        0.5 * k2 * (2 * k1 * sy * jd - cy * js) + 0.5 * (k2 + hx * k1) * dx * sy
    )
    T[..., 3, 2, 5] = 2 * (
        0.5 * hx / beta * k1 * k2 * (2 * cy * jf - sy * jd)
        + 0.5 * hx / beta * (k2 + hx * k1) * j1 * cy
        + 0.25 / beta * k1 * (sy - length * cy)
    )
    T[..., 3, 3, 5] = 2 * (
        0.5 * hx / beta * k2 * (2 * k1 * sy * jf - cy * jd)
        + 0.5 * hx / beta * (k2 + hx * k1) * j1 * sy
        - 0.25 / beta * k1 * length * sy
    )
    T[..., 4, 0, 0] = -1 * (
        hx / 12 / beta * khk * (sx * dx + 3 * j1)
        - 0.25 / beta * k1 * (length - sx * cx)
    )
    T[..., 4, 0, 1] = -2 * (hx / 12 / beta * khk * dx**2 + 0.25 / beta * k1 * sx**2)
    T[..., 4, 1, 1] = -1 * (
        hx / 6 / beta * khk * j2 - 0.5 / beta * sx - 0.25 / beta * k1 * (j1 - sx * dx)
    )
    T[..., 4, 0, 5] = -2 * (
        hx**2 / 12 / beta**2 * khk * (3 * dx * j1 - 4 * j2)
        + 0.25 * hx / beta**2 * k1 * j1 * (1 + cx)
        + 0.5 * hx / beta**2 * igamma2 * sx
    )
    T[..., 4, 1, 5] = -2 * (
        hx**2 / 12 / beta**2 * khk * (dx * dx**2 - 2 * sx * j2)
        + 0.25 * hx / beta**2 * k1 * sx * j1
        + 0.5 * hx / beta**2 * igamma2 * dx
    )
    T[..., 4, 5, 5] = -1 * (
        hx**3 / 6 / beta**3 * khk * (3 * j3 - 2 * dx * j2)
        + hx**2 / 6 / beta**3 * k1 * (sx * dx**2 - j2 * (1 + 2 * cx))
        + 1.5 / beta**3 * igamma2 * (hx**2 * j1 - length)
    )
    T[..., 4, 2, 2] = -1 * (
        -hx / beta * k1 * k2 * jf
        - 0.5 * hx / beta * (k2 + hx * k1) * j1
        + 0.25 / beta * k1 * (length - cy * sy)
    )
    T[..., 4, 2, 3] = -2 * (-0.5 * hx / beta * k2 * jd - 0.25 / beta * k1 * sy**2)
    T[..., 4, 3, 3] = -1 * (
        -hx / beta * k2 * jf
        + 0.5 * hx**2 / beta * j1
        - 0.25 / beta * (length + cy * sy)
    )

    rotation = rotation_matrix(tilt)
    T = torch.einsum(
        "...ji,...jkl,...kn,...lm->...inm",
        rotation,  # Switch index labels in einsum instead of transpose (faster)
        T,
        rotation,
        rotation,
    )

    return T


def drift_matrix(
    length: torch.Tensor, energy: torch.Tensor, species: Species
) -> torch.Tensor:
    """Create a first order transfer map for a drift space."""

    _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

    length, beta, igamma2 = torch.broadcast_tensors(length, beta, igamma2)
    tm = (
        torch.eye(7, device=length.device, dtype=length.dtype)
        .expand((*length.shape, 7, 7))
        .clone()
    )
    tm[..., (0, 2, 4), (1, 3, 5)] = torch.stack(
        [length, length, -length / beta**2 * igamma2], dim=-1
    )

    return tm


def rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """
    Rotate the coordinate system in the x-y plane.

    :param angle: Rotation angle in rad, for example `angle = np.pi/2` for vertical =
        dipole.
    :return: Rotation matrix to be multiplied to the element's transfer matrix.
    """
    cs = torch.cos(angle)
    sn = torch.sin(angle)

    tm = (
        torch.eye(7, dtype=angle.dtype, device=angle.device)
        .expand((*angle.shape, 7, 7))
        .clone()
    )
    tm[..., (0, 0, 1, 1, 2, 2, 3, 3), (0, 2, 1, 3, 0, 2, 1, 3)] = torch.stack(
        [cs, sn, cs, sn, -sn, cs, -sn, cs], dim=-1
    )

    return tm


def misalignment_matrix(
    misalignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift the beam for tracking beam through misaligned elements."""
    device = misalignment.device
    dtype = misalignment.dtype

    vector_shape = misalignment.shape[:-1]

    R_exit = (
        torch.eye(7, device=device, dtype=dtype).expand(*vector_shape, 7, 7).clone()
    )
    R_exit[..., (0, 2), (6, 6)] = misalignment

    R_entry = (
        torch.eye(7, device=device, dtype=dtype).expand(*vector_shape, 7, 7).clone()
    )
    R_entry[..., (0, 2), (6, 6)] = -misalignment

    return R_entry, R_exit
