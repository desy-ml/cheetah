"""Utility functions for creating transfer maps for elements."""

import torch
from scipy.constants import speed_of_light

from cheetah.particles import Species
from cheetah.utils import compute_relativistic_factors
from cheetah.utils.autograd import (
    cossqrtmcosdivdiff,
    si1mdiv,
    si2msi2divdiff,
    sicos1mdiv,
    simsidivdiff,
    sipsicos3mdiv,
)


def base_rmatrix(
    length: torch.Tensor,
    k1: torch.Tensor,
    hx: torch.Tensor,
    species: Species,
    energy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create a first order universal transfer map for a beamline element.

    :param length: Length of the element in m.
    :param k1: Quadrupole strength in 1/m**2.
    :param hx: Curvature (1/radius) of the element in 1/m.
    :param species: Particle species of the beam.
    :param energy: Beam energy in eV.
    :return: First order transfer map for the element.
    """
    factory_kwargs = {"device": length.device, "dtype": length.dtype}

    zero = length.new_zeros(())

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
        hx.square() * length.pow(3) * si1mdiv(kx2 * length.square()) / beta.square()
    ) - length / beta.square() * igamma2

    vector_shape = torch.broadcast_shapes(
        length.shape, k1.shape, hx.shape, energy.shape
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

    return R


def base_ttensor(
    length: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    hx: torch.Tensor,
    species: Species,
    energy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create a second order universal transfer map for a beamline element. Uses MAD
    convention.

    NOTE: It is possible that gradient components are missing when computing the
        over second order T tensors or beams tracked with them in very few highly
        specific scenarios. Even when it happens, it is unlikely to have a meaningful
        impact. This comment is left here to make sure this caveat is known.

    :param length: Length of the element in m.
    :param k1: Quadrupole strength in 1/m**2.
    :param k2: Sextupole strength in 1/m**3.
    :param hx: Curvature (1/radius) of the element in 1/m.
    :param species: Particle species of the beam.
    :param energy: Beam energy in eV.
    :return: Second order transfer map for the element.
    """
    zero = length.new_zeros(())

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
    dx = 0.5 * length.square() * r.square().real

    fx = length.pow(3) * si1mdiv(kx2 * length.square())
    f2y = length.pow(3) * sicos1mdiv(ky2 * length.square())

    j1 = fx
    j2 = length.pow(3) * sipsicos3mdiv(kx2 * length.square())
    # NOTE: Previously, an effort has been made to implement a custom autograd function
    # for `j3` as `sicoskuddelmuddel15mdiv`, but it was found that no proper limit
    # exists. Instead, a different model will have to be found in the future that
    # avoids these issues altogether.
    j3 = (
        (
            15.0 * length
            - 22.5 * sx
            + 9.0 * sx * cx
            - 1.5 * sx * cx.square()
            + kx2 * sx.pow(3)
        )
        / (6.0 * kx2.pow(3))
    ).where(kx2 != 0, length.pow(7) / 56.0)
    j_denominator = kx2 - 4.0 * ky2
    jc = length.square() * cossqrtmcosdivdiff(
        kx2 * length.square(), ky2 * length.square()
    )
    js = length.pow(3) * simsidivdiff(kx2 * length.square(), ky2 * length.square())
    jd = length.pow(4) * si2msi2divdiff(kx2 * length.square(), ky2 * length.square())
    jf = ((f2y - fx) / j_denominator).where(j_denominator != 0, length.pow(5) / 120.0)

    khk = k2 + 2.0 * hx * k1

    vector_shape = torch.broadcast_shapes(
        length.shape, k1.shape, k2.shape, hx.shape, energy.shape
    )

    T = length.new_zeros((7, 7, 7)).repeat(*vector_shape, 1, 1, 1)
    T[..., 0, 0, 0] = -khk * (sx.square() + dx) / 6.0 - 0.5 * hx * kx2 * sx.square()
    T[..., 0, 0, 1] = 2.0 * (-khk * sx * dx / 6.0 + 0.5 * hx * sx * cx)
    T[..., 0, 1, 1] = -khk * dx.square() / 6.0 + 0.5 * hx * dx * cx
    T[..., 0, 0, 5] = 2.0 * (
        -hx / 12.0 / beta * khk * (3.0 * sx * j1 - dx.square())
        + 0.5 * hx.square() / beta * sx.square()
        + 0.25 / beta * k1 * length * sx
    )
    T[..., 0, 1, 5] = 2.0 * (
        -hx / 12.0 / beta * khk * (sx * dx.square() - 2.0 * cx * j2)
        + 0.25 * hx.square() / beta * (sx * dx + cx * j1)
        - 0.25 / beta * (sx + length * cx)
    )
    T[..., 0, 5, 5] = (
        -(hx.square()) / 6.0 / beta.square() * khk * (dx.square() * dx - 2.0 * sx * j2)
        + 0.5 * hx.pow(3) / beta.square() * sx * j1
        - 0.5 * hx / beta.square() * length * sx
        - 0.5 * hx / (beta.square()) * igamma2 * dx
    )
    T[..., 0, 2, 2] = k1 * k2 * jd + 0.5 * (k2 + hx * k1) * dx
    T[..., 0, 2, 3] = 2.0 * (0.5 * k2 * js)
    T[..., 0, 3, 3] = k2 * jd - 0.5 * hx * dx
    T[..., 1, 0, 0] = -khk * sx * (1.0 + 2.0 * cx) / 6.0
    T[..., 1, 0, 1] = -2.0 * khk * dx * (1.0 + 2.0 * cx) / 6.0
    T[..., 1, 1, 1] = -khk * sx * dx / 3.0 - 0.5 * hx * sx
    T[..., 1, 0, 5] = 2.0 * (
        -hx / 12.0 / beta * khk * (3.0 * cx * j1 + sx * dx)
        - 0.25 / beta * k1 * (sx - length * cx)
    )
    T[..., 1, 1, 5] = 2.0 * (
        -hx / 12.0 / beta * khk * (3.0 * sx * j1 + dx.square())
        + 0.25 / beta * k1 * length * sx
    )
    T[..., 1, 5, 5] = (
        -(hx.square()) / 6.0 / beta.square() * khk * (sx * dx.square() - 2.0 * cx * j2)
        - 0.5 * hx / beta.square() * k1 * (cx * j1 - sx * dx)
        - 0.5 * hx / beta.square() * igamma2 * sx
    )
    T[..., 1, 2, 2] = k1 * k2 * js + 0.5 * (k2 + hx * k1) * sx
    T[..., 1, 2, 3] = 2.0 * (0.5 * k2 * jc)
    T[..., 1, 3, 3] = k2 * js - 0.5 * hx * sx
    T[..., 2, 0, 2] = 2.0 * (
        0.5 * k2 * (cy * jc - 2.0 * k1 * sy * js) + 0.5 * hx * k1 * sx * sy
    )
    T[..., 2, 0, 3] = 2.0 * (0.5 * k2 * (sy * jc - 2.0 * cy * js) + 0.5 * hx * sx * cy)
    T[..., 2, 1, 2] = 2.0 * (
        0.5 * k2 * (cy * js - 2.0 * k1 * sy * jd) + 0.5 * hx * k1 * dx * sy
    )
    T[..., 2, 1, 3] = 2.0 * (0.5 * k2 * (sy * js - 2.0 * cy * jd) + 0.5 * hx * dx * cy)
    T[..., 2, 2, 5] = 2.0 * (
        0.5 * hx / beta * k2 * (cy * jd - 2.0 * k1 * sy * jf)
        + 0.5 * hx.square() / beta * k1 * j1 * sy
        - 0.25 / beta * k1 * length * sy
    )
    T[..., 2, 3, 5] = 2.0 * (
        0.5 * hx / beta * k2 * (sy * jd - 2.0 * cy * jf)
        + 0.5 * hx.square() / beta * j1 * cy
        - 0.25 / beta * (sy + length * cy)
    )
    T[..., 3, 0, 2] = 2.0 * (
        0.5 * k1 * k2 * (2.0 * cy * js - sy * jc) + 0.5 * (k2 + hx * k1) * sx * cy
    )
    T[..., 3, 0, 3] = 2.0 * (
        0.5 * k2 * (2.0 * k1 * sy * js - cy * jc) + 0.5 * (k2 + hx * k1) * sx * sy
    )
    T[..., 3, 1, 2] = 2.0 * (
        0.5 * k1 * k2 * (2.0 * cy * jd - sy * js) + 0.5 * (k2 + hx * k1) * dx * cy
    )
    T[..., 3, 1, 3] = 2.0 * (
        0.5 * k2 * (2.0 * k1 * sy * jd - cy * js) + 0.5 * (k2 + hx * k1) * dx * sy
    )
    T[..., 3, 2, 5] = 2.0 * (
        0.5 * hx / beta * k1 * k2 * (2.0 * cy * jf - sy * jd)
        + 0.5 * hx / beta * (k2 + hx * k1) * j1 * cy
        + 0.25 / beta * k1 * (sy - length * cy)
    )
    T[..., 3, 3, 5] = 2.0 * (
        0.5 * hx / beta * k2 * (2.0 * k1 * sy * jf - cy * jd)
        + 0.5 * hx / beta * (k2 + hx * k1) * j1 * sy
        - 0.25 / beta * k1 * length * sy
    )
    T[..., 4, 0, 0] = -(
        hx / 12.0 / beta * khk * (sx * dx + 3.0 * j1)
        - 0.25 / beta * k1 * (length - sx * cx)
    )
    T[..., 4, 0, 1] = -2.0 * (
        hx / 12.0 / beta * khk * dx.square() + 0.25 / beta * k1 * sx.square()
    )
    T[..., 4, 1, 1] = -(
        hx / 6.0 / beta * khk * j2 - 0.5 / beta * sx - 0.25 / beta * k1 * (j1 - sx * dx)
    )
    T[..., 4, 0, 5] = -2.0 * (
        hx.square() / 12.0 / beta.square() * khk * (3.0 * dx * j1 - 4.0 * j2)
        + 0.25 * hx / beta.square() * k1 * j1 * (1.0 + cx)
        + 0.5 * hx / beta.square() * igamma2 * sx
    )
    T[..., 4, 1, 5] = -2.0 * (
        hx.square() / 12.0 / beta.square() * khk * (dx * dx.square() - 2.0 * sx * j2)
        + 0.25 * hx / beta.square() * k1 * sx * j1
        + 0.5 * hx / beta.square() * igamma2 * dx
    )
    T[..., 4, 5, 5] = -(
        hx.pow(3) / 6.0 / beta.pow(3) * khk * (3.0 * j3 - 2.0 * dx * j2)
        + hx.square()
        / 6.0
        / beta.pow(3)
        * k1
        * (sx * dx.square() - j2 * (1.0 + 2.0 * cx))
        + 1.5 / beta.pow(3) * igamma2 * (hx.square() * j1 - length)
    )
    T[..., 4, 2, 2] = -(
        -hx / beta * k1 * k2 * jf
        - 0.5 * hx / beta * (k2 + hx * k1) * j1
        + 0.25 / beta * k1 * (length - cy * sy)
    )
    T[..., 4, 2, 3] = -2.0 * (
        -0.5 * hx / beta * k2 * jd - 0.25 / beta * k1 * sy.square()
    )
    T[..., 4, 3, 3] = -(
        -hx / beta * k2 * jf
        + 0.5 * hx.square() / beta * j1
        - 0.25 / beta * (length + cy * sy)
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


def combined_rotation_misalignment_matrix(
    angle: torch.Tensor, misalignment: torch.Tensor
) -> torch.Tensor:
    """
    Apply misalignment and then rotation in the x-y plane.

    :param angle: Rotation angle in rad, for example `angle = np.pi/2` for vertical =
        dipole.
    :param misalignment: Misalignment vector [dx, dy] in m.
    :return: Combined rotation and misalignment matrix.
    """
    cs = angle.cos()
    sn = angle.sin()

    vector_shape = torch.broadcast_shapes(angle.shape, misalignment.shape[:-1])
    tm_entry = torch.eye(7, dtype=angle.dtype, device=angle.device).repeat(
        *vector_shape, 1, 1
    )

    # Rotation
    tm_entry[..., 0, 0] = cs
    tm_entry[..., 0, 2] = sn
    tm_entry[..., 1, 1] = cs
    tm_entry[..., 1, 3] = sn
    tm_entry[..., 2, 0] = -sn
    tm_entry[..., 2, 2] = cs
    tm_entry[..., 3, 1] = -sn
    tm_entry[..., 3, 3] = cs

    # Save for exit
    tm_exit = tm_entry.clone().mT

    # Misalignment
    tm_entry[..., 0, 6] = -misalignment[..., 0] * cs - misalignment[..., 1] * sn
    tm_entry[..., 2, 6] = misalignment[..., 0] * sn - misalignment[..., 1] * cs
    tm_exit[..., [0, 2], 6] = misalignment

    return tm_entry, tm_exit


def patch_rotation_matrix(
    pitch: torch.Tensor, tilt: torch.Tensor
) -> torch.Tensor:
    """
    Compute the 3D rotation matrix for a patch element based on its pitch and tilt.

    :param pitch: Pitch angles `(x_pitch, y_pitch)` in radians.
    :param tilt: Tilt angle in the x-y plane in radians.
    :return: 3D rotation matrix of shape `(..., 3, 3)`.
    """
    broadcast_batch = torch.broadcast_shapes(pitch.shape[:-1], tilt.shape)
    factory_kwargs = {"device": pitch.device, "dtype": pitch.dtype}

    if broadcast_batch:
        pitch = pitch.expand(*broadcast_batch, 2)
        tilt = tilt.expand(broadcast_batch)

    rotation_y = torch.zeros((*broadcast_batch, 3, 3), **factory_kwargs)
    rotation_y[..., 0, 0] = pitch[..., 0].cos()
    rotation_y[..., 0, 2] = pitch[..., 0].sin()
    rotation_y[..., 1, 1] = 1.0
    rotation_y[..., 2, 0] = -pitch[..., 0].sin()
    rotation_y[..., 2, 2] = pitch[..., 0].cos()

    rotation_neg_x = torch.zeros((*broadcast_batch, 3, 3), **factory_kwargs)
    rotation_neg_x[..., 0, 0] = 1.0
    rotation_neg_x[..., 1, 1] = pitch[..., 1].cos()
    rotation_neg_x[..., 1, 2] = pitch[..., 1].sin()
    rotation_neg_x[..., 2, 1] = -pitch[..., 1].sin()
    rotation_neg_x[..., 2, 2] = pitch[..., 1].cos()

    rotation_z = torch.zeros((*broadcast_batch, 3, 3), **factory_kwargs)
    rotation_z[..., 0, 0] = tilt.cos()
    rotation_z[..., 0, 1] = -tilt.sin()
    rotation_z[..., 1, 0] = tilt.sin()
    rotation_z[..., 1, 1] = tilt.cos()
    rotation_z[..., 2, 2] = 1.0

    return rotation_y @ rotation_neg_x @ rotation_z


def patch_transfer_map(
    offset: torch.Tensor,
    pitch: torch.Tensor,
    tilt: torch.Tensor,
    time_offset: torch.Tensor,
    drift_to_exit: bool = True,
    energy_offset: torch.Tensor | None = None,
    energy_setpoint: torch.Tensor | None = None,
    energy: torch.Tensor | None = None,
    species: Species | None = None,
) -> torch.Tensor:
    """
    Compute the first-order 7x7 transfer map for a `Patch` element.

    :param offset: Exit face offset `(x, y, z)` in meters.
    :param pitch: Pitch angles `(x_pitch, y_pitch)` in radians.
    :param tilt: Tilt angle in radians.
    :param time_offset: Reference time offset in seconds.
    :param drift_to_exit: Whether to drift particles to the exit face.
    :param energy_offset: Reference energy offset in eV.
    :param energy_setpoint: Target reference energy setpoint in eV.
    :param energy: Incoming reference energy in eV.
    :param species: Particle species of the beam.
    :return: First-order transfer map of shape `(..., 7, 7)`.
    """
    broadcast_batch = torch.broadcast_shapes(
        offset.shape[:-1], pitch.shape[:-1], tilt.shape, time_offset.shape
    )
    factory_kwargs = {"device": offset.device, "dtype": offset.dtype}

    if broadcast_batch:
        offset = offset.expand(*broadcast_batch, 3)
        pitch = pitch.expand(*broadcast_batch, 2)
        tilt = tilt.expand(broadcast_batch)
        time_offset = time_offset.expand(broadcast_batch)

    R = patch_rotation_matrix(pitch, tilt)
    RT = R.mT

    p0 = RT[..., :, 2]
    px0, py0, pz0 = p0[..., 0], p0[..., 1], p0[..., 2]

    z0 = -(
        RT[..., 2, 0] * offset[..., 0]
        + RT[..., 2, 1] * offset[..., 1]
        + RT[..., 2, 2] * offset[..., 2]
    )
    x_exit0 = -(
        RT[..., 0, 0] * offset[..., 0]
        + RT[..., 0, 1] * offset[..., 1]
        + RT[..., 0, 2] * offset[..., 2]
    )
    y_exit0 = -(
        RT[..., 1, 0] * offset[..., 0]
        + RT[..., 1, 1] * offset[..., 1]
        + RT[..., 1, 2] * offset[..., 2]
    )

    length = (R[..., :, 2] * offset).sum(dim=-1)

    tm = torch.eye(7, **factory_kwargs).repeat(*broadcast_batch, 1, 1)

    tm[..., 1, 1] = RT[..., 0, 0]
    tm[..., 1, 3] = RT[..., 0, 1]
    tm[..., 1, 5] = RT[..., 0, 2]

    tm[..., 3, 1] = RT[..., 1, 0]
    tm[..., 3, 3] = RT[..., 1, 1]
    tm[..., 3, 5] = RT[..., 1, 2]

    tm[..., 5, 5] = 1.0

    if drift_to_exit:
        pz0_safe = pz0.clamp(min=1e-12)
        pz0_sq = pz0_safe.square()

        for col, dr_idx, dp_idx, ddelta in [
            (0, 0, None, 0.0),
            (1, None, 0, 0.0),
            (2, 1, None, 0.0),
            (3, None, 1, 0.0),
            (4, None, None, 0.0),
            (5, None, 2, 1.0),
        ]:
            dr = RT[..., :, dr_idx] if dr_idx is not None else torch.zeros_like(p0)
            dp = RT[..., :, dp_idx] if dp_idx is not None else torch.zeros_like(p0)

            term_rx = dr[..., 0] - (px0 / pz0_safe) * dr[..., 2]
            term_px = -z0 * (dp[..., 0] / pz0_safe - px0 * dp[..., 2] / pz0_sq)
            tm[..., 0, col] = term_rx + term_px

            term_ry = dr[..., 1] - (py0 / pz0_safe) * dr[..., 2]
            term_py = -z0 * (dp[..., 1] / pz0_safe - py0 * dp[..., 2] / pz0_sq)
            tm[..., 2, col] = term_ry + term_py

            dtau = 1.0 if col == 4 else 0.0
            term_tau = (
                dtau
                + dr[..., 2] / pz0_safe
                + z0 * (ddelta / pz0_safe - dp[..., 2] / pz0_sq)
            )
            tm[..., 4, col] = term_tau

        tm[..., 0, 6] = x_exit0 - z0 * px0 / pz0_safe
        tm[..., 1, 6] = px0
        tm[..., 2, 6] = y_exit0 - z0 * py0 / pz0_safe
        tm[..., 3, 6] = py0
        tm[..., 4, 6] = time_offset * speed_of_light + z0 / pz0_safe + length
    else:
        tm[..., 0, 0] = RT[..., 0, 0]
        tm[..., 0, 2] = RT[..., 0, 1]
        tm[..., 2, 0] = RT[..., 1, 0]
        tm[..., 2, 2] = RT[..., 1, 1]
        tm[..., 0, 6] = x_exit0
        tm[..., 1, 6] = px0
        tm[..., 2, 6] = y_exit0
        tm[..., 3, 6] = py0
        tm[..., 4, 6] = time_offset * speed_of_light

    if energy is not None and species is not None:
        e_set = (
            energy_setpoint
            if energy_setpoint is not None
            else energy.new_tensor(0.0)
        )
        e_off = (
            energy_offset if energy_offset is not None else energy.new_tensor(0.0)
        )
        if torch.any(e_set > 0.0) or torch.any(e_off != 0.0):
            outgoing_energy = torch.where(e_set > 0.0, e_set, energy + e_off)
            _, _, beta0 = compute_relativistic_factors(energy, species.mass_eV)
            _, _, beta1 = compute_relativistic_factors(
                outgoing_energy, species.mass_eV
            )
            p_ratio = (energy * beta0) / (outgoing_energy * beta1)
            p_ratio_u = p_ratio.unsqueeze(-1)
            tm[..., [1, 3], :] = tm[..., [1, 3], :] * p_ratio_u
            tm[..., 5, 5] = p_ratio
            tm[..., 5, 6] = p_ratio - 1.0

    return tm
