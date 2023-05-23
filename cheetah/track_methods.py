"""Utility functions for creating transfer maps for the elements."""

from typing import Union

import numpy as np
import torch
from scipy import constants

REST_ENERGY = (
    constants.electron_mass
    * constants.speed_of_light**2
    / constants.elementary_charge
)  # electron mass


def rotation_matrix(angle: float, device: Union[str, torch.device] = "auto"):
    """Rotate the transfer map in x-y plane

    Parameters
    ----------
    angle : float [rad]
        rotation angle of the element, for example
        `angle = torch.pi/2` for vertical dipole
    device : [str, torch.device], optional
        device used for tracking, by default "auto"

    Returns
    -------
    torch.tensor
       rotation matrix to be multiplied to the element's transfer matrix
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cs = np.cos(angle)
    sn = np.sin(angle)
    return torch.tensor(
        [
            [cs, 0, sn, 0, 0, 0, 0],
            [0, cs, 0, sn, 0, 0, 0],
            [-sn, 0, cs, 0, 0, 0, 0],
            [0, -sn, 0, cs, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


def base_rmatrix(
    length: float,
    k1: float,
    hx: float,
    tilt: float = 0.0,
    energy: float = 0.0,
    device: Union[str, torch.device] = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gamma = energy / REST_ENERGY
    igamma2 = 1 / gamma**2 if gamma != 0 else 0

    beta = np.sqrt(1 - igamma2)

    kx2 = k1 + hx**2
    ky2 = -k1
    kx = np.sqrt(kx2 + 0.0j)
    ky = np.sqrt(ky2 + 0.0j)
    cx = np.cos(kx * length).real
    cy = np.cos(ky * length).real
    sy = (np.sin(ky * length) / ky).real if ky != 0 else length

    if kx != 0:
        sx = (np.sin(kx * length) / kx).real
        dx = hx / kx2 * (1.0 - cx)
        r56 = hx**2 * (length - sx) / kx2 / beta**2
    else:
        sx = length
        dx = length**2 * hx / 2
        r56 = hx**2 * length**3 / 6 / beta**2

    r56 -= length / beta**2 * igamma2

    R = torch.tensor(
        [
            [cx, sx, 0, 0, 0, dx / beta, 0],
            [-kx2 * sx, cx, 0, 0, 0, sx * hx / beta, 0],
            [0, 0, cy, sy, 0, 0, 0],
            [0, 0, -ky2 * sy, cy, 0, 0, 0],
            [sx * hx / beta, dx / beta, 0, 0, 1, r56, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )

    # Rotate the R matrix for skew / vertical magnets
    if tilt != 0:
        R = torch.dot(torch.dot(rotation_matrix(-tilt), R), rotation_matrix(tilt))
    return R


def misalignment_matrix(
    misalignment: tuple[float, float], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift the beam for tracking beam through misaligned elements"""
    R_exit = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, misalignment[0]],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, misalignment[1]],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    R_entry = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, -misalignment[0]],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, -misalignment[1]],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    return R_exit, R_entry
