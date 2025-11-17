"""Utility functions for creating transfer maps for elements."""

import torch

from cheetah.particles import Species
from cheetah.utils import compute_relativistic_factors
from cheetah.utils.autograd import cos1mprodbdiva, si1mdiv


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
    factory_kwargs = {"device": length.device, "dtype": length.dtype}

    zero = torch.tensor(0.0, **factory_kwargs)

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

    r = (0.5 * kx * length / torch.pi).sinc()
    dx = hx * 0.5 * length.square() * r.square().real

    r56 = (
        hx.square() * length.pow(3) * si1mdiv(kx2) / beta.square()
    ) - length / beta.square() * igamma2

    vector_shape = torch.broadcast_shapes(
        length.shape, k1.shape, hx.shape, tilt.shape, energy.shape
    )

    R = torch.eye(7, **factory_kwargs).repeat(*vector_shape, 1, 1)
    R[..., 0, 0] = cx
    R[..., 0, 1] = sx
    R[..., 0, 5] = dx / beta
    R[..., 1, 0] = -kx2 * sx
    R[..., 1, 1] = cx
    R[..., 1, 5] = sx * hx / beta
    R[..., 2, 2] = cy
    R[..., 2, 3] = sy
    R[..., 3, 2] = -ky2 * sy
    R[..., 3, 3] = cy
    R[..., 4, 0] = sx * hx / beta
    R[..., 4, 1] = dx / beta
    R[..., 4, 5] = r56

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
    factory_kwargs = {"device": length.device, "dtype": length.dtype}

    zero = torch.tensor(0.0, **factory_kwargs)

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
    dx = cos1mprodbdiva(kx2, length)

    d2y = 0.5 * sy.square()
    s2y = sy * cy
    c2y = (2 * ky * length).cos().real
    fx = torch.where(kx2 != 0, (length - sx) / kx2, length.pow(3) / 6.0)
    f2y = torch.where(ky2 != 0, (length - s2y) / ky2, length.pow(3) / 6.0)

    j1 = torch.where(kx2 != 0, (length - sx) / kx2, length.pow(3) / 6.0)
    j2 = torch.where(
        kx2 != 0,
        (3.0 * length - 4.0 * sx + sx * cx) / (2 * kx2.square()),
        length.pow(5) / 20.0,
    )
    j3 = torch.where(
        kx2 != 0,
        (
            15.0 * length
            - 22.5 * sx
            + 9.0 * sx * cx
            - 1.5 * sx * cx.square()
            + kx2 * sx.pow(3)
        )
        / (6.0 * kx2.pow(3)),
        length.pow(7) / 56.0,
    )
    j_denominator = kx2 - 4 * ky2
    jc = torch.where(
        j_denominator != 0, (c2y - cx) / j_denominator, 0.5 * length.square()
    )
    js = torch.where(
        j_denominator != 0, (cy * sy - sx) / j_denominator, length.pow(3) / 6.0
    )
    jd = torch.where(
        j_denominator != 0, (d2y - dx) / j_denominator, length.pow(4) / 24.0
    )
    jf = torch.where(
        j_denominator != 0, (f2y - fx) / j_denominator, length.pow(5) / 120.0
    )

    khk = k2 + 2 * hx * k1

    vector_shape = torch.broadcast_shapes(
        length.shape, k1.shape, k2.shape, hx.shape, tilt.shape, energy.shape
    )

    T = torch.zeros((7, 7, 7), **factory_kwargs).repeat(*vector_shape, 1, 1, 1)
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
        + 0.5 * hx.pow(3) / beta.square() * sx * j1
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
        hx.pow(3) / 6 / beta.pow(3) * khk * (3 * j3 - 2 * dx * j2)
        + hx.square() / 6 / beta.pow(3) * k1 * (sx * dx.square() - j2 * (1 + 2 * cx))
        + 1.5 / beta.pow(3) * igamma2 * (hx.square() * j1 - length)
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

    rotation = rotation_matrix(tilt)
    T = torch.einsum(
        "...ji,...jkl,...kn,...lm->...inm", rotation, T, rotation, rotation
    )

    return T


def drift_matrix(
    length: torch.Tensor, energy: torch.Tensor, species: Species
) -> torch.Tensor:
    """Create a first order transfer map for a drift space."""
    factory_kwargs = {"device": length.device, "dtype": length.dtype}

    _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

    vector_shape = torch.broadcast_shapes(length.shape, igamma2.shape)

    tm = torch.eye(7, **factory_kwargs).repeat((*vector_shape, 1, 1))
    tm[..., 0, 1] = length
    tm[..., 2, 3] = length
    tm[..., 4, 5] = -length / beta.square() * igamma2

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

    tm = torch.eye(7, dtype=angle.dtype, device=angle.device).repeat(*angle.shape, 1, 1)
    tm[..., 0, 0] = cs
    tm[..., 0, 2] = sn
    tm[..., 1, 1] = cs
    tm[..., 1, 3] = sn
    tm[..., 2, 0] = -sn
    tm[..., 2, 2] = cs
    tm[..., 3, 1] = -sn
    tm[..., 3, 3] = cs

    return tm


def misalignment_matrix(
    misalignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift the beam for tracking beam through misaligned elements."""
    factory_kwargs = {"device": misalignment.device, "dtype": misalignment.dtype}

    vector_shape = misalignment.shape[:-1]

    R_exit = torch.eye(7, **factory_kwargs).repeat(*vector_shape, 1, 1)
    R_exit[..., 0, 6] = misalignment[..., 0]
    R_exit[..., 2, 6] = misalignment[..., 1]

    R_entry = torch.eye(7, **factory_kwargs).repeat(*vector_shape, 1, 1)
    R_entry[..., 0, 6] = -misalignment[..., 0]
    R_entry[..., 2, 6] = -misalignment[..., 1]

    return R_entry, R_exit
