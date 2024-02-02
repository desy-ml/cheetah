"""Utility functions for creating transfer maps for the elements."""

from typing import Optional

import torch
from scipy import constants

REST_ENERGY = torch.tensor(
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass


def rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """Rotate the transfer map in x-y plane

    :param angle: Rotation angle in rad, for example `angle = np.pi/2` for vertical =
        dipole.
    :return: Rotation matrix to be multiplied to the element's transfer matrix.
    """
    cs = torch.cos(angle)
    sn = torch.sin(angle)

    tm = torch.eye(7, dtype=angle.dtype, device=angle.device)
    tm[0, 0] = cs
    tm[0, 2] = sn
    tm[1, 1] = cs
    tm[1, 3] = sn
    tm[2, 0] = -sn
    tm[2, 2] = cs
    tm[3, 1] = -sn
    tm[3, 3] = cs

    return tm


def base_rmatrix(
    length: torch.Tensor,
    k1: torch.Tensor,
    hx: torch.Tensor,
    tilt: Optional[torch.Tensor] = None,
    energy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Create a universal transfer matrix for a beamline element.

    :param length: Length of the element in m.
    :param k1: Quadrupole strength in 1/m**2.
    :param hx: Curvature (1/radius) of the element in 1/m**2.
    :param tilt: Roation of the element relative to the longitudinal axis in rad.
    :param energy: Beam energy in eV.
    :return: Transfer matrix for the element.
    """
    device = length.device
    dtype = length.dtype

    tilt = tilt if tilt is not None else torch.tensor(0.0, device=device, dtype=dtype)
    energy = (
        energy if energy is not None else torch.tensor(0.0, device=device, dtype=dtype)
    )

    gamma = energy / REST_ENERGY.to(device=device, dtype=dtype)
    igamma2 = (
        1 / gamma**2 if gamma != 0 else torch.tensor(0.0, device=device, dtype=dtype)
    )

    beta = torch.sqrt(1 - igamma2)

    if k1 == 0:
        k1 = k1 + torch.tensor(
            1e-12, device=device, dtype=dtype
        )  # Avoid division by zero
    kx2 = k1 + hx**2
    ky2 = -k1
    kx = torch.sqrt(torch.complex(kx2, torch.tensor(0.0, device=device, dtype=dtype)))
    ky = torch.sqrt(torch.complex(ky2, torch.tensor(0.0, device=device, dtype=dtype)))
    cx = torch.cos(kx * length).real
    cy = torch.cos(ky * length).real
    sy = (torch.sin(ky * length) / ky).real if ky != 0 else length

    sx = (torch.sin(kx * length) / kx).real
    dx = hx / kx2 * (1.0 - cx)
    r56 = hx**2 * (length - sx) / kx2 / beta**2

    r56 = r56 - length / beta**2 * igamma2

    R = torch.eye(7, dtype=dtype, device=device)
    R[0, 0] = cx
    R[0, 1] = sx
    R[0, 5] = dx / beta
    R[1, 0] = -kx2 * sx
    R[1, 1] = cx
    R[1, 5] = sx * hx / beta
    R[2, 2] = cy
    R[2, 3] = sy
    R[3, 2] = -ky2 * sy
    R[3, 3] = cy
    R[4, 0] = sx * hx / beta
    R[4, 1] = dx / beta
    R[4, 5] = r56

    # Rotate the R matrix for skew / vertical magnets
    if tilt != 0:
        R = torch.matmul(torch.matmul(rotation_matrix(-tilt), R), rotation_matrix(tilt))
    return R


def misalignment_matrix(
    misalignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift the beam for tracking beam through misaligned elements"""
    device = misalignment.device
    dtype = misalignment.dtype

    R_exit = torch.eye(7, device=device, dtype=dtype)
    R_exit[0, 6] = misalignment[0]
    R_exit[2, 6] = misalignment[1]

    R_entry = torch.eye(7, device=device, dtype=dtype)
    R_entry[0, 6] = -misalignment[0]
    R_entry[2, 6] = -misalignment[1]

    return R_exit, R_entry  # TODO: This order is confusing, should be entry, exit
