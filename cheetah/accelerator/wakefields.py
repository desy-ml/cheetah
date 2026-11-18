"""
Wakefield physics used by Cavity.

The wake function w(s) depends only on the structure. The beam enters through its line
charge density lambda, in the wake potential W = w * lambda, and
through the grid, which is sized from the bunch length.

Signs: w_parallel > 0 means a trailing particle loses energy.
The separation is s = tau_test - tau_source >= 0, so only particles ahead contribute by causality.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
from scipy import constants

from cheetah.utils import cloud_in_cell_charge_deposition

__all__ = [
    "SPEED_OF_LIGHT",
    "ResonantModeWake",
    "LongitudinalGrid",
    "longitudinal_grid_from_beam",
    "deposit_line_density",
    "convolve_causal_wake",
    "longitudinal_wake_potential",
    "longitudinal_energy_kick",
    "reference_momentum",
    "gaussian_resonant_wake_potential",
]

SPEED_OF_LIGHT = constants.speed_of_light


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _like(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    """Cast 'value' to the device and dtype of 'reference', keeping gradients."""
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _scalar(value: torch.Tensor | float) -> float:
    """Detached Python float, for grid decisions only."""
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(value)


def _next_power_of_two(n: int) -> int:
    """Smallest power of two that is at least 'n'."""
    return 1 << max(0, int(n - 1)).bit_length()


def reference_momentum(
    energy_eV: torch.Tensor, mass_eV: torch.Tensor | float
) -> torch.Tensor:
    """Reference momentum 'p0c' in eV from total energy and rest mass."""
    mass_eV = _like(mass_eV, energy_eV)
    return torch.sqrt(torch.clamp(energy_eV**2 - mass_eV**2, min=0.0))


# --------------------------------------------------------------------------------------
# Wake Green functions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResonantModeWake:
    """
    Wake of a single resonant mode.

    'w(s) = wake_strength exp(-Gamma s / c) [cos(omega_bar s / c) - (Gamma / omega_bar)
    sin(omega_bar s / c)]' with 'Gamma = omega_r / 2Q'. With 'Q = inf' this is the pure
    cosine wake 'wake_strength cos(omega_r s / c)'.

    :param frequency: Mode frequency in Hz.
    :param wake_strength: Wake amplitude 'w(0+)' in V/C.
    :param quality_factor: Mode quality factor 'Q'. Infinite means no damping.
    """

    frequency: float | torch.Tensor
    wake_strength: float | torch.Tensor
    quality_factor: float | torch.Tensor = math.inf

    # -- geometry-free derived quantities ------------------------------------------

    @property
    def wavelength(self) -> float:
        """Mode wavelength in meters."""
        return SPEED_OF_LIGHT / _scalar(self.frequency)

    def sample(self, separations: torch.Tensor) -> torch.Tensor:
        """'w_parallel' at separations 's', zero ahead of the source."""
        omega_r = _like(2.0 * math.pi * _like(self.frequency, separations), separations)
        wake_strength = _like(self.wake_strength, separations)
        q = _scalar(self.quality_factor)

        if math.isinf(q):
            phase = omega_r * separations / SPEED_OF_LIGHT
            wake = wake_strength * torch.cos(phase)
        else:
            gamma = omega_r / (2.0 * _like(q, separations))
            omega_bar = torch.sqrt(torch.clamp(omega_r**2 - gamma**2, min=1e-30))
            phase = omega_bar * separations / SPEED_OF_LIGHT
            envelope = torch.exp(-gamma * separations / SPEED_OF_LIGHT)
            wake = (
                wake_strength
                * envelope
                * (torch.cos(phase) - (gamma / omega_bar) * torch.sin(phase))
            )
        return torch.where(separations >= 0.0, wake, torch.zeros_like(wake))

    @property
    def feature_length(self) -> float:
        """One radian of phase, 'lambda / 2 pi'."""
        return self.wavelength / (2.0 * math.pi)


# --------------------------------------------------------------------------------------
# Grid: chosen from the beam and from the kernel
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LongitudinalGrid:
    """Uniform grid along 'tau', with the numbers needed to check it."""

    z_min: float
    dz: float
    n_grid: int
    sigma_tau: float
    target_dz: float
    resolved: bool

    @property
    def z_max(self) -> float:
        """Position of the last node."""
        return self.z_min + (self.n_grid - 1) * self.dz

    @property
    def cells_per_sigma(self) -> float:
        """Grid cells per rms bunch length."""
        return self.sigma_tau / self.dz if self.dz > 0 else math.inf

    def nodes(self, device=None, dtype=None) -> torch.Tensor:
        """Node positions as a tensor."""
        return torch.linspace(
            self.z_min, self.z_max, self.n_grid, device=device, dtype=dtype
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        """Short summary of the grid."""
        return (
            f"LongitudinalGrid(n_grid={self.n_grid}, dz={self.dz:.3e} m, "
            f"extent=[{self.z_min:.3e}, {self.z_max:.3e}] m, "
            f"cells_per_sigma={self.cells_per_sigma:.1f}, resolved={self.resolved})"
        )


def longitudinal_grid_from_beam(
    tau: torch.Tensor,
    *,
    feature_lengths: tuple[float, ...] = (),
    cells_per_feature: float = 8.0,
    margin_cells: int = 4,
    min_grid: int = 256,
    max_grid: int = 1 << 17,
    warn: bool = True,
) -> LongitudinalGrid:
    """
    Size the grid from the bunch and the wake's shortest length scale.

    'dz' resolves the shorter of the rms bunch length and the wake's feature lengths
    with 'cells_per_feature' cells. The wake is causal, so there is no padding ahead of
    the bunch, and spare cells from rounding up go behind it.
    """
    tau_flat = tau.detach().reshape(-1)
    tau_min = float(tau_flat.min())
    tau_max = float(tau_flat.max())
    sigma_tau = float(tau_flat.std()) if tau_flat.numel() > 1 else 0.0

    candidates = [sigma_tau] + [f for f in feature_lengths]
    candidates = [
        c for c in candidates if c is not None and math.isfinite(c) and c > 0.0
    ]
    if not candidates:
        raise ValueError(
            "Cannot size the grid: no finite length scale from the beam or kernel."
        )
    target_dz = min(candidates) / cells_per_feature

    span = tau_max - tau_min
    if span <= 0.0:
        span = 2.0 * margin_cells * target_dz

    n_required = int(math.ceil(span / target_dz)) + 2 * margin_cells + 1
    n_grid = min(max(_next_power_of_two(n_required), min_grid), max_grid)

    if n_grid >= n_required:
        dz = target_dz
        resolved = True
    else:
        # Clamped by 'max_grid', so resolution is lost and the user is warned
        dz = (span + 2 * margin_cells * target_dz) / (n_grid - 1)
        resolved = False
        if warn:
            warnings.warn(
                f"Wake grid clamped to max_grid={max_grid}: dz = {dz:.3e} m gives "
                f"only {sigma_tau / dz:.2f} cells per sigma_tau (target "
                f"{cells_per_feature:.0f}). Raise max_grid or lower "
                "cells_per_feature.",
                stacklevel=2,
            )

    return LongitudinalGrid(
        z_min=tau_min - margin_cells * dz,
        dz=dz,
        n_grid=n_grid,
        sigma_tau=sigma_tau,
        target_dz=target_dz,
        resolved=resolved,
    )


# --------------------------------------------------------------------------------------
# Deposition, convolution, gather
# --------------------------------------------------------------------------------------


def deposit_line_density(
    tau: torch.Tensor,
    charges: torch.Tensor,
    grid: LongitudinalGrid,
) -> torch.Tensor:
    """Deposit charges onto the grid with cloud-in-cell, as a line density in C/m."""
    tau = tau.reshape(-1)
    charges = charges.reshape(-1).to(device=tau.device, dtype=tau.dtype)

    extent = torch.tensor(
        [[grid.z_min - 0.5 * grid.dz, grid.z_max + 0.5 * grid.dz]],
        device=tau.device,
        dtype=tau.dtype,
    )
    deposited = cloud_in_cell_charge_deposition(
        positions=tau.reshape(-1, 1),
        bins=grid.n_grid,
        extent=extent,
        charges=charges,
    )
    return deposited / grid.dz


def convolve_causal_wake(
    line_density: torch.Tensor,
    kernel_samples: torch.Tensor,
    dz: float,
    *,
    trapezoidal: bool = True,
) -> torch.Tensor:
    """
    Causal convolution 'W_i = sum_j w_j lambda_(i-j) dz', using FFTs.

    :param trapezoidal: Halve the 'j = 0' sample, so a particle sees half of its own
        slice's wake.
    """
    if trapezoidal:
        kernel_samples = kernel_samples.clone()
        kernel_samples[0] = 0.5 * kernel_samples[0]

    n = line_density.numel()
    m = kernel_samples.numel()
    length = n + m - 1
    n_fft = _next_power_of_two(length)
    convolution = torch.fft.irfft(
        torch.fft.rfft(line_density, n=n_fft) * torch.fft.rfft(kernel_samples, n=n_fft),
        n=n_fft,
    )
    return convolution[:n] * dz


def _gather_linear(
    values_on_grid: torch.Tensor, tau: torch.Tensor, grid: LongitudinalGrid
) -> torch.Tensor:
    """Linearly interpolate a grid quantity back to the particles."""
    u = (tau.reshape(-1) - grid.z_min) / grid.dz
    i0 = torch.clamp(torch.floor(u.detach()).long(), 0, grid.n_grid - 2)
    frac = (u - i0).to(values_on_grid.dtype)
    return (1.0 - frac) * values_on_grid[i0] + frac * values_on_grid[i0 + 1]


# --------------------------------------------------------------------------------------
# Wake potentials and kicks
# --------------------------------------------------------------------------------------


def longitudinal_wake_potential(
    tau: torch.Tensor,
    particle_charges: torch.Tensor,
    kernel: ResonantModeWake,
    *,
    grid: LongitudinalGrid | None = None,
    cells_per_feature: float = 8.0,
    max_grid: int = 1 << 17,
    trapezoidal: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Longitudinal wake potential 'W' in volts at each particle.

    :return: 'W' per particle, and a dict of diagnostics (grid, line density, kernel and
        'W' on the grid) for plotting.
    """
    tau = tau.reshape(-1)
    charges = particle_charges.reshape(-1).abs()

    if grid is None:
        grid = longitudinal_grid_from_beam(
            tau,
            feature_lengths=(kernel.feature_length,),
            cells_per_feature=cells_per_feature,
            max_grid=max_grid,
        )

    line_density = deposit_line_density(tau, charges, grid)

    separations = torch.arange(
        grid.n_grid, device=tau.device, dtype=tau.dtype
    ) * grid.dz
    kernel_samples = kernel.sample(separations)

    wake_on_grid = convolve_causal_wake(
        line_density, kernel_samples, grid.dz, trapezoidal=trapezoidal
    )
    wake_per_particle = _gather_linear(wake_on_grid, tau, grid)

    diagnostics = {
        "grid": grid,
        "z_grid": grid.nodes(device=tau.device, dtype=tau.dtype),
        "lambda": line_density,
        "kernel": kernel_samples,
        "W_grid": wake_on_grid,
    }
    return wake_per_particle, diagnostics


def longitudinal_energy_kick(
    tau: torch.Tensor,
    particle_charges: torch.Tensor,
    kernel: ResonantModeWake,
    *,
    num_elementary_charges: float = -1.0,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Energy change in eV of each particle from the longitudinal wake.

    'dE = -|Z| W': a positive wake decelerates for any species, because
    the source and test charges share a sign.
    """
    wake, diagnostics = longitudinal_wake_potential(
        tau, particle_charges, kernel, **kwargs
    )
    return -abs(num_elementary_charges) * wake, diagnostics


# --------------------------------------------------------------------------------------
# Analytic reference, for tests and for the validation notebook
# --------------------------------------------------------------------------------------


def gaussian_resonant_wake_potential(
    tau,
    *,
    sigma_tau: float,
    wavelength: float,
    total_charge: float,
    wake_strength: float,
):
    """
    Closed-form 'W(tau)' in volts for a Gaussian bunch in an undamped cosine wake.

    Used as the analytical reference in the tests.
    """
    import numpy as np
    from scipy.special import erf

    tau = np.asarray(tau, dtype=float)
    k = 2.0 * np.pi / wavelength
    argument = (tau + 1j * k * sigma_tau**2) / (sigma_tau * np.sqrt(2.0))
    return (
        (wake_strength * total_charge / 2.0)
        * np.exp(-0.5 * k**2 * sigma_tau**2)
        * np.real(np.exp(1j * k * tau) * (1.0 + erf(argument)))
    )
