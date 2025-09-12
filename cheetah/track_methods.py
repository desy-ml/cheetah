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

    tilt = tilt if tilt is not None else zero
    energy = energy if energy is not None else zero

    _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

    kx2 = k1 + hx.square()
    ky2 = -k1
    kx = (torch.complex(kx2, zero)).sqrt()
    ky = (torch.complex(ky2, zero)).sqrt()
    cx = (kx * length).cos().real
    cy = (ky * length).cos().real
    sx = ((kx * length / torch.pi).sinc() * length).real
    sy = ((ky * length / torch.pi).sinc() * length).real
    dx = torch.where(kx2 != 0, hx / kx2 * (1.0 - cx), zero)
    r56 = torch.where(kx2 != 0, hx.square() * (length - sx) / kx2 / beta.square(), zero)

    r56 = r56 - length / beta.square() * igamma2

    cx, sx, dx, cy, sy, r56 = torch.broadcast_tensors(cx, sx, dx, cy, sy, r56)

    R = torch.eye(7, dtype=cx.dtype, device=cx.device).expand(*cx.shape, 7, 7).clone()
    R[
        ...,
        (0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4),
        (0, 1, 5, 0, 1, 5, 2, 3, 2, 3, 0, 1, 5),
    ] = torch.stack(
        [
            cx,
            sx,
            dx / beta,
            -kx2 * sx,
            cx,
            sx * hx / beta,
            cy,
            sy,
            -ky2 * sy,
            cy,
            sx * hx / beta,
            dx / beta,
            r56,
        ],
        dim=-1,
    )

    # Rotate the R matrix for skew / vertical magnets. The rotation only has an effect
    # if hx != 0 or k1 != 0. Note that the first if is here to improve speed when no
    # rotation needs to be applied accross all vector dimensions. The torch.where is
    # here to improve numerical stability for the vector elements where no rotation
    # needs to be applied.
    if ((tilt != 0) & ((hx != 0) | (k1 != 0))).any():
        rotation = rotation_matrix(tilt)
        R = torch.where(
            ((tilt != 0) & ((hx != 0) | (k1 != 0))).unsqueeze(-1).unsqueeze(-1),
            rotation.mT @ R @ rotation,
            R,
        )

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
    zero = length.new_zeros(())

    tilt = tilt if tilt is not None else zero
    energy = energy if energy is not None else zero

    _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

    kx2 = k1 + hx.square()
    ky2 = -k1
    kx = (torch.complex(kx2, zero)).sqrt()
    ky = (torch.complex(ky2, zero)).sqrt()
    cx = (kx * length).cos().real
    cy = (ky * length).cos().real
    sx = ((kx * length / torch.pi).sinc() * length).real
    sy = ((ky * length / torch.pi).sinc() * length).real
    dx = torch.where(kx != 0, (1.0 - cx) / kx2, length.square() / 2.0)

    d2y = 0.5 * sy.square()
    s2y = sy * cy
    c2y = (2 * ky * length).cos().real
    fx = torch.where(kx2 != 0, (length - sx) / kx2, length**3 / 6.0)
    f2y = torch.where(ky2 != 0, (length - s2y) / ky2, length**3 / 6.0)

    j1 = torch.where(kx2 != 0, (length - sx) / kx2, length**3 / 6.0)
    j2 = torch.where(
        kx2 != 0,
        (3.0 * length - 4.0 * sx + sx * cx) / (2 * kx2.square()),
        length**5 / 20.0,
    )
    j3 = torch.where(
        kx2 != 0,
        (
            15.0 * length
            - 22.5 * sx
            + 9.0 * sx * cx
            - 1.5 * sx * cx.square()
            + kx2 * sx**3
        )
        / (6.0 * kx2**3),
        length**7 / 56.0,
    )
    j_denominator = kx2 - 4 * ky2
    jc = torch.where(
        j_denominator != 0, (c2y - cx) / j_denominator, 0.5 * length.square()
    )
    js = torch.where(
        j_denominator != 0, (cy * sy - sx) / j_denominator, length**3 / 6.0
    )
    jd = torch.where(j_denominator != 0, (d2y - dx) / j_denominator, length**4 / 24.0)
    jf = torch.where(j_denominator != 0, (f2y - fx) / j_denominator, length**5 / 120.0)

    khk = k2 + 2 * hx * k1

    vector_shape = torch.broadcast_shapes(
        length.shape, k1.shape, k2.shape, hx.shape, tilt.shape, energy.shape
    )

    T = length.new_zeros((7, 7, 7)).expand(*vector_shape, 7, 7, 7).clone()
    T[..., 0, 0, 0] = -1 / 6 * khk * (sx.square() + dx) - 0.5 * hx * kx2 * sx.square()
    T[..., 0, 0, 1] = 2 * (-1 / 6 * khk * sx * dx + 0.5 * hx * sx * cx)
    T[..., 0, 1, 1] = -1 / 6 * khk * dx.square() + 0.5 * hx * dx * cx
    T[..., 0, 0, 5] = 2 * (
        -hx / 12 / beta * khk * (3 * sx * j1 - dx.square())
        + 0.5 * hx.square() / beta * sx.square()
        + 0.25 / beta * k1 * length * sx
    )
    T[..., 0, 1, 5] = 2 * (
        -hx / 12 / beta * khk * (sx * dx.square() - 2 * cx * j2)
        + 0.25 * hx.square() / beta * (sx * dx + cx * j1)
        - 0.25 / beta * (sx + length * cx)
    )
    T[..., 0, 5, 5] = (
        -(hx.square()) / 6 / beta.square() * khk * (dx.square() * dx - 2 * sx * j2)
        + 0.5 * hx**3 / beta.square() * sx * j1
        - 0.5 * hx / beta.square() * length * sx
        - 0.5 * hx / (beta.square()) * igamma2 * dx
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
        -hx / 12 / beta * khk * (3 * sx * j1 + dx.square())
        + 0.25 / beta * k1 * length * sx
    )
    T[..., 1, 5, 5] = (
        -(hx.square()) / 6 / beta.square() * khk * (sx * dx.square() - 2 * cx * j2)
        - 0.5 * hx / beta.square() * k1 * (cx * j1 - sx * dx)
        - 0.5 * hx / beta.square() * igamma2 * sx
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
        + 0.5 * hx.square() / beta * k1 * j1 * sy
        - 0.25 / beta * k1 * length * sy
    )
    T[..., 2, 3, 5] = 2 * (
        0.5 * hx / beta * k2 * (sy * jd - 2 * cy * jf)
        + 0.5 * hx.square() / beta * j1 * cy
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
    T[..., 4, 0, 1] = -2 * (
        hx / 12 / beta * khk * dx.square() + 0.25 / beta * k1 * sx.square()
    )
    T[..., 4, 1, 1] = -1 * (
        hx / 6 / beta * khk * j2 - 0.5 / beta * sx - 0.25 / beta * k1 * (j1 - sx * dx)
    )
    T[..., 4, 0, 5] = -2 * (
        hx.square() / 12 / beta.square() * khk * (3 * dx * j1 - 4 * j2)
        + 0.25 * hx / beta.square() * k1 * j1 * (1 + cx)
        + 0.5 * hx / beta.square() * igamma2 * sx
    )
    T[..., 4, 1, 5] = -2 * (
        hx.square() / 12 / beta.square() * khk * (dx * dx.square() - 2 * sx * j2)
        + 0.25 * hx / beta.square() * k1 * sx * j1
        + 0.5 * hx / beta.square() * igamma2 * dx
    )
    T[..., 4, 5, 5] = -1 * (
        hx**3 / 6 / beta**3 * khk * (3 * j3 - 2 * dx * j2)
        + hx.square() / 6 / beta**3 * k1 * (sx * dx.square() - j2 * (1 + 2 * cx))
        + 1.5 / beta**3 * igamma2 * (hx.square() * j1 - length)
    )
    T[..., 4, 2, 2] = -1 * (
        -hx / beta * k1 * k2 * jf
        - 0.5 * hx / beta * (k2 + hx * k1) * j1
        + 0.25 / beta * k1 * (length - cy * sy)
    )
    T[..., 4, 2, 3] = -2 * (-0.5 * hx / beta * k2 * jd - 0.25 / beta * k1 * sy.square())
    T[..., 4, 3, 3] = -1 * (
        -hx / beta * k2 * jf
        + 0.5 * hx.square() / beta * j1
        - 0.25 / beta * (length + cy * sy)
    )

    # Rotate the T tensor for skew / vertical magnets. The rotation only has an effect
    # if hx != 0, k1 != 0 or k2 != 0. Note that the first if is here to improve speed
    # when no rotation needs to be applied accross all vector dimensions. The
    # torch.where is here to improve numerical stability for the vector elements where
    # no rotation needs to be applied.
    if ((tilt != 0) & ((hx != 0) | (k1 != 0) | (k2 != 0))).any():
        rotation = rotation_matrix(tilt)
        T = torch.where(
            ((tilt != 0) & ((hx != 0) | (k1 != 0) | (k2 != 0)))
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1),
            torch.einsum(
                "...ij,...jkl,...kn,...lm->...inm", rotation.mT, T, rotation, rotation
            ),
            T,
        )
    return T


def drift_matrix(
    length: torch.Tensor, energy: torch.Tensor, species: Species
) -> torch.Tensor:
    """Create a first order transfer map for a drift space."""
    factory_kwargs = {"device": length.device, "dtype": length.dtype}

    _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

    length, beta, igamma2 = torch.broadcast_tensors(length, beta, igamma2)

    tm = torch.eye(7, **factory_kwargs).expand((*length.shape, 7, 7)).clone()
    tm[..., (0, 2, 4), (1, 3, 5)] = torch.stack(
        [length, length, -length / beta.square() * igamma2], dim=-1
    )

    return tm


def rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """
    Rotate the coordinate system in the x-y plane.

    :param angle: Rotation angle in rad, for example `angle = np.pi/2` for vertical =
        dipole.
    :return: Rotation matrix to be multiplied to the element's transfer matrix.
    """
    cs = angle.cos()
    sn = angle.sin()

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
    factory_kwargs = {"device": misalignment.device, "dtype": misalignment.dtype}

    vector_shape = misalignment.shape[:-1]

    R_exit = torch.eye(7, **factory_kwargs).expand(*vector_shape, 7, 7).clone()
    R_exit[..., (0, 2), (6, 6)] = misalignment

    R_entry = torch.eye(7, **factory_kwargs).expand(*vector_shape, 7, 7).clone()
    R_entry[..., (0, 2), (6, 6)] = -misalignment

    return R_entry, R_exit
