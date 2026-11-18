import math
from typing import Literal

import matplotlib.pyplot as plt
import torch
import torch.fft
from matplotlib.patches import Rectangle
from scipy.constants import electron_mass, elementary_charge, mu_0, speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.utils import (
    UniqueNameGenerator,
    cache_transfer_map,
    compute_relativistic_factors,
)
from cheetah.utils.cloud_in_cell import cloud_in_cell_charge_deposition

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class RadiationField:
    """
    Representation of a transverse optical or X-ray radiation field in an FEL.

    :param field: Complex transverse electric field envelope on a 2D grid with shape
        `(..., H, W)` or time-dependent 3D grid `(..., num_slices, H, W)` in V/m.
    :param wavelength: Radiation wavelength in meters.
    :param extent: Transverse half-size of the grid in meters.
    :param slice_spacing: Longitudinal spacing between radiation slices in meters for
        time-dependent simulations, or `None` for steady-state.
    """

    def __init__(
        self,
        field: torch.Tensor,
        wavelength: torch.Tensor,
        extent: torch.Tensor,
        slice_spacing: torch.Tensor | None = None,
    ) -> None:
        self.field = field
        self.wavelength = wavelength
        self.extent = extent
        self.slice_spacing = slice_spacing

    @property
    def slice_power(self) -> torch.Tensor:
        """Radiation power in Watts for each slice (or total for steady-state)."""
        grid_shape = self.field.shape[-2:]
        dx = 2.0 * self.extent / (grid_shape[-1] - 1)
        dy = 2.0 * self.extent / (grid_shape[-2] - 1)
        if self.slice_spacing is not None:
            dx = dx[..., None]
            dy = dy[..., None]
        z0 = mu_0 * speed_of_light
        return (0.5 / z0) * self.field.abs().square().sum(dim=(-2, -1)) * dx * dy

    @property
    def power(self) -> torch.Tensor:
        """Mean radiation power in Watts across slices (or total for steady-state)."""
        sp = self.slice_power
        if self.slice_spacing is not None:
            return sp.mean(dim=-1)
        return sp

    @property
    def pulse_energy(self) -> torch.Tensor | None:
        """Total pulse energy in Joules for time-dependent fields, else `None`."""
        if self.slice_spacing is None:
            return None
        return self.slice_power.sum(dim=-1) * (self.slice_spacing / speed_of_light)

    @property
    def peak_intensity(self) -> torch.Tensor:
        """Peak intensity of the radiation field in W/m^2."""
        z0 = mu_0 * speed_of_light
        return (0.5 / z0) * self.field.abs().square().amax(dim=(-2, -1))

    def diffract(
        self, distance: torch.Tensor, absorbing_boundary: bool = True
    ) -> "RadiationField":
        """
        Propagate the radiation field in free space over `distance` in meters using
        paraxial diffraction in Fourier space.
        """
        grid_shape = self.field.shape[-2:]
        ny, nx = grid_shape
        dx = 2.0 * self.extent / (nx - 1)
        dy = 2.0 * self.extent / (ny - 1)
        k_r = 2.0 * torch.pi / self.wavelength

        kx = (
            2.0
            * torch.pi
            * torch.fft.fftfreq(nx, d=1.0, device=self.field.device)
            / dx[..., None]
        )
        ky = (
            2.0
            * torch.pi
            * torch.fft.fftfreq(ny, d=1.0, device=self.field.device)
            / dy[..., None]
        )
        k_perp2 = ky[..., :, None].square() + kx[..., None, :].square()

        phase_arg = -0.5j * (k_perp2 / k_r[..., None, None]) * distance[..., None, None]
        if self.slice_spacing is not None:
            phase_arg = phase_arg.unsqueeze(-3)
        phase = torch.exp(phase_arg)
        new_field = torch.fft.ifft2(torch.fft.fft2(self.field) * phase)
        if absorbing_boundary:
            new_field = new_field * self._absorbing_mask(
                ny, nx, device=self.field.device, dtype=self.field.real.dtype
            )
        return RadiationField(
            field=new_field,
            wavelength=self.wavelength,
            extent=self.extent,
            slice_spacing=self.slice_spacing,
        )

    @staticmethod
    def _absorbing_mask(
        ny: int, nx: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Smooth super-Gaussian boundary absorber to suppress FFT wrap-around."""
        ux = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype)
        uy = torch.linspace(-1.0, 1.0, ny, device=device, dtype=dtype)
        mask_x = torch.exp(-((ux / 0.98) ** 64))
        mask_y = torch.exp(-((uy / 0.98) ** 64))
        return mask_y[:, None] * mask_x[None, :]

    def clone(self) -> "RadiationField":
        """Create a clone of the radiation field."""
        return RadiationField(
            field=self.field.clone(),
            wavelength=self.wavelength.clone(),
            extent=self.extent.clone(),
            slice_spacing=(
                self.slice_spacing.clone() if self.slice_spacing is not None else None
            ),
        )


class Undulator(Element):
    """
    Element representing an undulator in a particle accelerator.

    Implements a linear paraxial undulator field approximation taken from Ocelot, see
    S. Tomin, Varenna, 2017, as well as a fast, differentiable 3D FEL simulation mode
    matching Genesis 1.3 v4.

    :param length: Length in meters.
    :param period: Undulator period in meters.
    :param kx: Vertical undulator parameter.
    :param ky: Horizontal undulator parameter.
    :param tracking_method: Method to use for tracking through the element. Can be
        `"linear"` (default) or `"fel"`.
    :param step_size: Integration step size in meters for FEL tracking. If `None`,
        defaults to 0.12 m.
    :param wavelength: Radiation wavelength in meters. If `None`, resonant wavelength
        is determined automatically from electron energy and undulator parameter.
    :param seed_power: Initial radiation seed power in Watts.
    :param bunching: Initial macroparticle bunching factor (dimensionless).
    :param current: Electron beam peak current in Amperes.
    :param grid_extent: Transverse half-size of the radiation field grid in meters.
    :param grid_shape: Grid resolution tuple `(num_y, num_x)`.
    :param num_macroparticles: Number of macroparticles used for FEL slice tracking.
    :param nbins: Number of quiet-start phase bins per slice.
    :param num_slices: Number of longitudinal slices (`1` for steady-state, `> 1` for
        time-dependent SASE simulation with slippage).
    :param slice_spacing: Longitudinal spacing between slices in meters for
        time-dependent simulations.
    :param shot_noise: Whether to add Penman-McNeil shot-noise phase perturbations in
        time-dependent (`num_slices > 1`) simulations.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment. If `None` (default), a warning is raised for
        invalid names. Set to `True` to sanitise, or `False` to silence the warning.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element (e.g. control-system addresses or PVs). This information is *not* used
        in simulation and may contain any extra data the user wants to store along with
        the lattice. See :doc:`/examples/including_metadata` for more information.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.
    """

    supported_tracking_methods = ["linear", "fel"]

    def __init__(
        self,
        length: torch.Tensor,
        period: torch.Tensor | None = None,
        kx: torch.Tensor | None = None,
        ky: torch.Tensor | None = None,
        tracking_method: Literal["linear", "fel"] = "linear",
        step_size: torch.Tensor | None = None,
        wavelength: torch.Tensor | None = None,
        seed_power: torch.Tensor | None = None,
        bunching: torch.Tensor | None = None,
        current: torch.Tensor | None = None,
        grid_extent: torch.Tensor | None = None,
        grid_shape: tuple[int, int] = (65, 65),
        num_macroparticles: int = 2048,
        nbins: int = 8,
        num_slices: int = 1,
        slice_spacing: torch.Tensor | None = None,
        shot_noise: bool = True,
        name: str | None = None,
        sanitize_name: bool | None = None,
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            name=name, sanitize_name=sanitize_name, metadata=metadata, **factory_kwargs
        )

        self.length = length

        self.register_buffer_or_parameter(
            "period",
            period if period is not None else torch.tensor(1.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "kx", kx if kx is not None else torch.tensor(0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "ky", ky if ky is not None else torch.tensor(0.0, **factory_kwargs)
        )

        self.register_buffer_or_parameter("step_size", step_size)
        self.register_buffer_or_parameter("wavelength", wavelength)
        self.register_buffer_or_parameter("seed_power", seed_power)
        self.register_buffer_or_parameter("bunching", bunching)
        self.register_buffer_or_parameter("current", current)
        self.register_buffer_or_parameter("grid_extent", grid_extent)
        self.grid_shape = grid_shape
        self.num_macroparticles = num_macroparticles
        self.nbins = nbins
        self.num_slices = num_slices
        self.register_buffer_or_parameter("slice_spacing", slice_spacing)
        self.shot_noise = shot_noise

        self.radiation: RadiationField | None = None
        self.tracking_method = tracking_method

    @property
    def is_active(self) -> bool:
        return torch.logical_or(self.kx != 0.0, self.ky != 0.0).any().item()

    @cache_transfer_map
    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        factory_kwargs = {"device": self.length.device, "dtype": self.length.dtype}

        gamma, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

        vector_shape = torch.broadcast_shapes(
            self.length.shape,
            igamma2.shape,
            self.kx.shape,
            self.ky.shape,
            self.period.shape,
        )

        tm = torch.eye(7, **factory_kwargs).repeat((*vector_shape, 1, 1))
        tm[..., 4, 5] = (
            -self.length
            * igamma2
            * (beta.square().reciprocal() + 0.5 * (self.kx.square() + self.ky.square()))
        )

        spatial_frequency = torch.where(
            self.period > 0.0,
            math.sqrt(2.0) * torch.pi / (self.period * gamma * beta),
            torch.tensor(0.0, **factory_kwargs),
        )

        # Transverse focusing from vertical field (Kx > 0.0)
        omega_x = spatial_frequency * self.kx
        cos_omega_x = (omega_x * self.length).cos()

        tm[..., 2, 2] = cos_omega_x
        tm[..., 2, 3] = (omega_x * self.length / torch.pi).sinc() * self.length
        tm[..., 3, 2] = -(omega_x * self.length).sin() * omega_x
        tm[..., 3, 3] = cos_omega_x

        # Transverse focusing from horizontal field (Ky > 0.0)
        omega_y = spatial_frequency * self.ky
        cos_omega_y = (omega_y * self.length).cos()

        tm[..., 0, 0] = cos_omega_y
        tm[..., 0, 1] = (omega_y * self.length / torch.pi).sinc() * self.length
        tm[..., 1, 0] = -(omega_y * self.length).sin() * omega_y
        tm[..., 1, 1] = cos_omega_y

        return tm

    @property
    def is_skippable(self) -> bool:
        return self.tracking_method != "fel"

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the undulator. If `tracking_method` is `"fel"`,
        simulates the 3D FEL amplification and beam microbunching; otherwise tracks with
        linear transfer map.

        :param incoming: Beam of particles entering the undulator.
        :return: Beam of particles exiting the undulator.
        """
        if self.tracking_method == "fel":
            return self._track_fel(incoming)
        return super().track(incoming)

    def _track_fel(self, incoming: Beam) -> Beam:
        """Track beam through undulator with 3D FEL physics."""
        factory_kwargs = {
            "device": incoming.energy.device,
            "dtype": incoming.energy.dtype,
        }
        gamma0 = incoming.energy / incoming.species.mass_eV
        z0 = mu_0 * speed_of_light

        # Undulator K parameter and Bessel coupling factor
        k_param = (self.kx.square() + self.ky.square()).sqrt()
        aw = k_param / math.sqrt(2.0)
        xi = aw.square() / (2.0 * (1.0 + aw.square()))
        is_helical = torch.isclose(self.kx.abs(), self.ky.abs()) & (k_param > 0.0)
        jj = torch.where(
            is_helical,
            torch.ones_like(xi),
            torch.special.bessel_j0(xi) - torch.special.bessel_j1(xi),
        )
        lambdau = self.period
        k_u = 2.0 * torch.pi / lambdau

        # Wavelength determination
        if self.wavelength is not None:
            lambda0 = self.wavelength
        elif hasattr(incoming, "radiation") and incoming.radiation is not None:
            lambda0 = incoming.radiation.wavelength
        else:
            lambda0 = (lambdau / (2.0 * gamma0.square())) * (1.0 + aw.square())

        k_r = 2.0 * torch.pi / lambda0

        complex_dtype = (
            torch.complex128
            if incoming.energy.dtype == torch.float64
            else torch.complex64
        )

        # Radiation field and transverse grid setup
        beam_vector_shape = (
            incoming.mu.shape[:-1]
            if isinstance(incoming, ParameterBeam)
            else incoming.particles.shape[:-2]
        )
        vector_shape = torch.broadcast_shapes(
            beam_vector_shape,
            incoming.energy.shape,
            self.length.shape,
            self.period.shape,
            self.kx.shape,
            self.ky.shape,
        )

        has_in_rad = hasattr(incoming, "radiation") and incoming.radiation is not None
        is_time_dependent = (self.num_slices > 1) or (
            has_in_rad and incoming.radiation.slice_spacing is not None
        )
        if has_in_rad and incoming.radiation.slice_spacing is not None:
            num_slices = incoming.radiation.field.shape[-3]
            slice_spacing = incoming.radiation.slice_spacing.expand(vector_shape)
        elif is_time_dependent:
            num_slices = self.num_slices
            if self.slice_spacing is not None:
                slice_spacing = self.slice_spacing.expand(vector_shape)
            else:
                has_sigma_tau = incoming.sigma_tau > 0.0
                slice_spacing = torch.where(
                    has_sigma_tau,
                    6.0 * incoming.sigma_tau / num_slices,
                    10.0 * lambda0,
                ).expand(vector_shape)
        else:
            num_slices = 1
            slice_spacing = None

        slice_shape = (num_slices,) if is_time_dependent else ()
        field_batch_shape = (*vector_shape, *slice_shape)

        if has_in_rad:
            grid_extent = incoming.radiation.extent.expand(vector_shape)
            field = incoming.radiation.field.to(dtype=complex_dtype)
            ny, nx = field.shape[-2:]
            field = field.expand(*field_batch_shape, ny, nx).clone()
            dx = 2.0 * grid_extent / (nx - 1)
            dy = 2.0 * grid_extent / (ny - 1)
        else:
            if self.grid_extent is not None:
                grid_extent = self.grid_extent.expand(vector_shape)
            else:
                sigma_max = torch.maximum(incoming.sigma_x, incoming.sigma_y)
                grid_extent = torch.clamp(6.0 * sigma_max, min=3e-4).expand(
                    vector_shape
                )
            ny, nx = self.grid_shape
            dx = 2.0 * grid_extent / (nx - 1)
            dy = 2.0 * grid_extent / (ny - 1)
            default_seed = 0.0 if (is_time_dependent and self.shot_noise) else 1.0
            seed_power = (
                self.seed_power
                if self.seed_power is not None
                else torch.tensor(default_seed, **factory_kwargs)
            )
            w0 = grid_extent / 4.0
            e0 = torch.sqrt(4.0 * z0 * seed_power / (math.pi * w0.square()))

            ux = torch.linspace(-1.0, 1.0, nx, **factory_kwargs)
            uy = torch.linspace(-1.0, 1.0, ny, **factory_kwargs)
            grid_uy, grid_ux = torch.meshgrid(uy, ux, indexing="ij")
            grid_x = grid_ux * grid_extent[..., None, None]
            grid_y = grid_uy * grid_extent[..., None, None]
            field_2d = (
                e0[..., None, None]
                * torch.exp(
                    -(grid_x.square() + grid_y.square()) / w0[..., None, None].square()
                )
            ).to(dtype=complex_dtype)
            if is_time_dependent:
                field = (
                    field_2d.unsqueeze(-3).expand(*field_batch_shape, ny, nx).clone()
                )
            else:
                field = field_2d

        # Macroparticles setup
        nbins = self.nbins
        num_transverse = max(1, self.num_macroparticles // nbins)
        num_particles = num_transverse * nbins

        if self.current is not None:
            current = self.current
        else:
            has_beam_current = (incoming.total_charge > 0.0) & (
                incoming.sigma_tau > 0.0
            )
            safe_sigma_tau = torch.where(
                has_beam_current,
                incoming.sigma_tau,
                torch.ones_like(incoming.sigma_tau),
            )
            beam_peak_current = (
                incoming.total_charge
                * speed_of_light
                / (math.sqrt(2.0 * math.pi) * safe_sigma_tau)
            )
            current = torch.where(
                has_beam_current,
                beam_peak_current,
                torch.tensor(1000.0, **factory_kwargs),
            )

        if is_time_dependent:
            s_idx = torch.linspace(
                -0.5 * (num_slices - 1),
                0.5 * (num_slices - 1),
                num_slices,
                **factory_kwargs,
            )
            s_coords = s_idx * slice_spacing[..., None]
            if self.current is None and (incoming.sigma_tau > 0.0).any():
                safe_st = torch.where(
                    incoming.sigma_tau > 0.0,
                    incoming.sigma_tau,
                    torch.ones_like(incoming.sigma_tau),
                )
                gauss_env = torch.exp(-0.5 * (s_coords / safe_st[..., None]).square())
                current_profile = torch.where(
                    (incoming.sigma_tau > 0.0)[..., None],
                    current[..., None] * gauss_env,
                    current[..., None].expand(*field_batch_shape),
                )
            else:
                current_profile = current[..., None].expand(*field_batch_shape)
        else:
            current_profile = current

        bunch = (
            self.bunching
            if self.bunching is not None
            else torch.tensor(1e-6, **factory_kwargs)
        )

        if hasattr(incoming, "_fel_particles") and incoming._fel_particles is not None:
            x, px, y, py, gammas, theta = incoming._fel_particles
        elif isinstance(incoming, ParticleBeam):
            x = incoming.particles[..., 0].expand(*vector_shape, -1).contiguous()
            px = incoming.particles[..., 1].expand(*vector_shape, -1).contiguous()
            y = incoming.particles[..., 2].expand(*vector_shape, -1).contiguous()
            py = incoming.particles[..., 3].expand(*vector_shape, -1).contiguous()
            gammas = (
                gamma0.unsqueeze(-1) * (1.0 + incoming.particles[..., 5].contiguous())
            ).expand(*vector_shape, -1)
            if is_time_dependent:
                x = x.unsqueeze(-2).expand(*field_batch_shape, -1).clone()
                px = px.unsqueeze(-2).expand(*field_batch_shape, -1).clone()
                y = y.unsqueeze(-2).expand(*field_batch_shape, -1).clone()
                py = py.unsqueeze(-2).expand(*field_batch_shape, -1).clone()
                gammas = gammas.unsqueeze(-2).expand(*field_batch_shape, -1).clone()
            num_parts = x.shape[-1]
            theta0 = torch.linspace(
                0.0, 2.0 * math.pi, num_parts + 1, **factory_kwargs
            )[:-1]
            theta = (theta0 - 2.0 * bunch[..., None] * torch.sin(theta0)).expand(
                *field_batch_shape, -1
            )
            num_particles = num_parts
        elif isinstance(incoming, ParameterBeam):
            cov_x = incoming.cov[..., :2, :2].expand(*vector_shape, 2, 2)
            cov_y = incoming.cov[..., 2:4, 2:4].expand(*vector_shape, 2, 2)
            mu_x = incoming.mu[..., :2].expand(*vector_shape, 2)
            mu_y = incoming.mu[..., 2:4].expand(*vector_shape, 2)
            if is_time_dependent:
                cov_x = cov_x.unsqueeze(-3).expand(*field_batch_shape, 2, 2)
                cov_y = cov_y.unsqueeze(-3).expand(*field_batch_shape, 2, 2)
                mu_x = mu_x.unsqueeze(-2).expand(*field_batch_shape, 2)
                mu_y = mu_y.unsqueeze(-2).expand(*field_batch_shape, 2)

            gen = torch.Generator(device=incoming.energy.device).manual_seed(42)
            # Share whitened normalized draws across vector_shape (CRN)
            z_x = torch.randn(
                *slice_shape, num_transverse, 2, generator=gen, **factory_kwargs
            )
            z_y = torch.randn(
                *slice_shape, num_transverse, 2, generator=gen, **factory_kwargs
            )
            if num_transverse > 2:
                z_x = z_x - z_x.mean(dim=-2, keepdim=True)
                z_y = z_y - z_y.mean(dim=-2, keepdim=True)
                cov_zx = (z_x.mT @ z_x) / num_transverse
                cov_zy = (z_y.mT @ z_y) / num_transverse
                z_x = z_x @ torch.linalg.inv(torch.linalg.cholesky(cov_zx)).mT
                z_y = z_y @ torch.linalg.inv(torch.linalg.cholesky(cov_zy)).mT
            l_x = torch.linalg.cholesky(cov_x)
            l_y = torch.linalg.cholesky(cov_y)
            sample_x = mu_x.unsqueeze(-2) + z_x @ l_x.mT
            sample_y = mu_y.unsqueeze(-2) + z_y @ l_y.mT

            x = sample_x[..., 0].repeat_interleave(nbins, dim=-1)
            px = sample_x[..., 1].repeat_interleave(nbins, dim=-1)
            y = sample_y[..., 0].repeat_interleave(nbins, dim=-1)
            py = sample_y[..., 1].repeat_interleave(nbins, dim=-1)

            delgam = (incoming.sigma_p * gamma0).expand(vector_shape)
            mu_gamma = (gamma0 * (1.0 + incoming.mu[..., 5])).expand(vector_shape)
            if is_time_dependent:
                delgam = delgam.unsqueeze(-1).expand(field_batch_shape)
                mu_gamma = mu_gamma.unsqueeze(-1).expand(field_batch_shape)
            z_gamma = torch.randn(
                *slice_shape, num_transverse, generator=gen, **factory_kwargs
            )
            if num_transverse > 2:
                z_gamma = z_gamma - z_gamma.mean(dim=-1, keepdim=True)
                z_gamma = z_gamma / z_gamma.std(dim=-1, unbiased=False, keepdim=True)
            gammas_trans = mu_gamma.unsqueeze(-1) + z_gamma * delgam.unsqueeze(-1)
            gammas = gammas_trans.repeat_interleave(nbins, dim=-1)

            theta_bins = torch.linspace(
                0.0, 2.0 * math.pi, nbins + 1, **factory_kwargs
            )[:-1]
            theta0 = theta_bins.repeat(num_transverse)
            bunch_b = bunch[..., None] if is_time_dependent else bunch
            theta = (theta0 - 2.0 * bunch_b[..., None] * torch.sin(theta0)).expand(
                *field_batch_shape, -1
            )
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

        if is_time_dependent and self.shot_noise and not has_in_rad:
            gen_sn = torch.Generator(device=incoming.energy.device).manual_seed(137)
            n_e = (
                current_profile
                * slice_spacing[..., None]
                / (elementary_charge * speed_of_light)
            ).clamp(min=1.0)
            sigma_theta = torch.sqrt(num_particles / n_e).clamp(max=math.pi)
            theta = (
                theta
                + torch.randn(
                    *field_batch_shape,
                    num_particles,
                    generator=gen_sn,
                    **factory_kwargs,
                )
                * sigma_theta[..., None]
            )

        # Spectral grid for diffraction
        kx = (
            2.0
            * torch.pi
            * torch.fft.fftfreq(nx, d=1.0, device=field.device)
            / dx[..., None]
        )
        ky = (
            2.0
            * torch.pi
            * torch.fft.fftfreq(ny, d=1.0, device=field.device)
            / dy[..., None]
        )
        k_perp2 = ky[..., :, None].square() + kx[..., None, :].square()
        absorbing_mask = RadiationField._absorbing_mask(
            ny, nx, device=field.device, dtype=field.real.dtype
        )
        batch_size = math.prod(field_batch_shape) if len(field_batch_shape) > 0 else 1

        grid_extent_p = grid_extent.unsqueeze(-1) if is_time_dependent else grid_extent
        dx_f = dx.unsqueeze(-1) if is_time_dependent else dx
        dy_f = dy.unsqueeze(-1) if is_time_dependent else dy
        gamma0_p = gamma0.unsqueeze(-1) if is_time_dependent else gamma0

        def interp_field(
            fld: torch.Tensor, x_p: torch.Tensor, y_p: torch.Tensor
        ) -> torch.Tensor:
            norm_x = (x_p / grid_extent_p[..., None]).to(dtype=fld.real.dtype)
            norm_y = (y_p / grid_extent_p[..., None]).to(dtype=fld.real.dtype)
            grid_sample_coords = torch.stack([norm_x, norm_y], dim=-1).reshape(
                batch_size, 1, -1, 2
            )
            fld_2c = torch.stack([fld.real, fld.imag], dim=-3).reshape(
                batch_size, 2, ny, nx
            )
            e_both = torch.nn.functional.grid_sample(
                fld_2c, grid_sample_coords, align_corners=True, mode="bilinear"
            )
            return torch.complex(
                e_both[:, 0].reshape(*field_batch_shape, -1),
                e_both[:, 1].reshape(*field_batch_shape, -1),
            )

        cic_extent_y = torch.stack(
            [-grid_extent_p - 0.5 * dy_f, grid_extent_p + 0.5 * dy_f], dim=-1
        )
        cic_extent_x = torch.stack(
            [-grid_extent_p - 0.5 * dx_f, grid_extent_p + 0.5 * dx_f], dim=-1
        )
        cic_extent = torch.stack([cic_extent_y, cic_extent_x], dim=-2).unsqueeze(-3)

        def deposit_source(
            x_p: torch.Tensor,
            y_p: torch.Tensor,
            gammas_p: torch.Tensor,
            theta_p: torch.Tensor,
        ) -> torch.Tensor:
            amp = gamma0_p[..., None] / gammas_p
            charges_2 = torch.stack(
                [amp * torch.cos(theta_p), -amp * torch.sin(theta_p)], dim=-2
            )
            pos_2 = (
                torch.stack([y_p, x_p], dim=-1)
                .unsqueeze(-3)
                .expand(*field_batch_shape, 2, -1, 2)
            )
            grid_2 = cloud_in_cell_charge_deposition(
                pos_2,
                bins=(ny, nx),
                extent=cic_extent,
                charges=charges_2,
            )
            j_bunch = (
                torch.complex(grid_2[..., 0, :, :], grid_2[..., 1, :, :])
                * (current_profile / (num_particles * dx_f * dy_f))[..., None, None]
            )
            return (z0 * aw * jj / (math.sqrt(2.0) * gamma0))[..., None, None] * (
                j_bunch if not is_time_dependent else j_bunch
            )

        step_size = self.step_size if self.step_size is not None else 2.0 * self.period
        n_steps = max(1, int(round((self.length / step_size).max().item())))
        dz = self.length / n_steps

        phase_half_arg = (
            -0.5j * (k_perp2 / k_r[..., None, None]) * (0.5 * dz)[..., None, None]
        )
        if is_time_dependent:
            phase_half_arg = phase_half_arg.unsqueeze(-3)
        phase_half = torch.exp(phase_half_arg)
        phase_full = phase_half * phase_half

        k_beta_x = (self.ky / math.sqrt(2.0)) * k_u / gamma0
        k_beta_y = (self.kx / math.sqrt(2.0)) * k_u / gamma0
        half_dz = 0.5 * dz
        cos_bx_half = torch.cos(k_beta_x * half_dz)[..., None]
        sinc_bx_half = ((k_beta_x * half_dz / torch.pi).sinc() * half_dz)[..., None]
        cos_by_half = torch.cos(k_beta_y * half_dz)[..., None]
        sinc_by_half = ((k_beta_y * half_dz / torch.pi).sinc() * half_dz)[..., None]

        cos_bx = torch.cos(k_beta_x * dz)[..., None]
        sinc_bx = ((k_beta_x * dz / torch.pi).sinc() * dz)[..., None]
        k_sin_bx = (k_beta_x * torch.sin(k_beta_x * dz))[..., None]
        cos_by = torch.cos(k_beta_y * dz)[..., None]
        sinc_by = ((k_beta_y * dz / torch.pi).sinc() * dz)[..., None]
        k_sin_by = (k_beta_y * torch.sin(k_beta_y * dz))[..., None]

        e_charge_over_mc2 = elementary_charge / (electron_mass * speed_of_light**2)
        aw_jj = (aw * jj)[..., None]
        k_u_b = k_u[..., None]
        k_r_b = k_r[..., None]
        aw2_b = aw.square()[..., None]
        dz_p = dz[..., None]
        dz_f = dz[..., None, None]
        if is_time_dependent:
            cos_bx_half = cos_bx_half.unsqueeze(-2)
            sinc_bx_half = sinc_bx_half.unsqueeze(-2)
            cos_by_half = cos_by_half.unsqueeze(-2)
            sinc_by_half = sinc_by_half.unsqueeze(-2)
            cos_bx = cos_bx.unsqueeze(-2)
            sinc_bx = sinc_bx.unsqueeze(-2)
            k_sin_bx = k_sin_bx.unsqueeze(-2)
            cos_by = cos_by.unsqueeze(-2)
            sinc_by = sinc_by.unsqueeze(-2)
            k_sin_by = k_sin_by.unsqueeze(-2)
            aw_jj = aw_jj.unsqueeze(-2)
            k_u_b = k_u_b.unsqueeze(-2)
            k_r_b = k_r_b.unsqueeze(-2)
            aw2_b = aw2_b.unsqueeze(-2)
            dz_p = dz_p.unsqueeze(-2)
            dz_f = dz_f.unsqueeze(-3)

        def compute_rhs(
            fld: torch.Tensor,
            x_p: torch.Tensor,
            px_p: torch.Tensor,
            y_p: torch.Tensor,
            py_p: torch.Tensor,
            gammas_p: torch.Tensor,
            theta_p: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            e_p = interp_field(fld, x_p, y_p)
            dgamma_dz = (
                -e_charge_over_mc2
                * (aw_jj / (math.sqrt(2.0) * gammas_p))
                * (e_p * torch.exp(1.0j * theta_p)).real
            )
            dtheta_dz = k_u_b - (k_r_b / (2.0 * gammas_p.square())) * (
                1.0 + aw2_b + gammas_p.square() * (px_p.square() + py_p.square())
            )
            s_source = deposit_source(x_p, y_p, gammas_p, theta_p)
            return dgamma_dz, dtheta_dz, s_source

        slip_accum = 0.0
        slip_per_step = (
            (dz * lambda0 / (lambdau * slice_spacing)).max().item()
            if is_time_dependent
            else 0.0
        )

        field = torch.fft.ifft2(torch.fft.fft2(field) * phase_half)
        for step in range(n_steps):
            # 2nd-order midpoint (RK2) integration for particles and field source
            dg1, dt1, s1 = compute_rhs(field, x, px, y, py, gammas, theta)

            x_mid = x * cos_bx_half + px * sinc_bx_half
            y_mid = y * cos_by_half + py * sinc_by_half
            gammas_mid = gammas + 0.5 * dg1 * dz_p
            theta_mid = theta + 0.5 * dt1 * dz_p
            field_mid = field + 0.5 * s1 * dz_f

            dg2, dt2, s2 = compute_rhs(
                field_mid, x_mid, px, y_mid, py, gammas_mid, theta_mid
            )

            gammas = gammas + dg2 * dz_p
            theta = theta + dt2 * dz_p
            field = field + s2 * dz_f

            phase = phase_full if step < n_steps - 1 else phase_half
            field = torch.fft.ifft2(torch.fft.fft2(field) * phase) * absorbing_mask

            if is_time_dependent:
                slip_accum += slip_per_step
                if slip_accum >= 1.0:
                    shift = int(slip_accum)
                    slip_accum -= shift
                    if shift < num_slices:
                        field = torch.nn.functional.pad(
                            field[..., :-shift, :, :], (0, 0, 0, 0, shift, 0)
                        )
                    else:
                        field = torch.zeros_like(field)

            x_new = x * cos_bx + px * sinc_bx
            px = -x * k_sin_bx + px * cos_bx
            x = x_new

            y_new = y * cos_by + py * sinc_by
            py = -y * k_sin_by + py * cos_by
            y = y_new

        radiation = RadiationField(
            field=field,
            wavelength=lambda0,
            extent=grid_extent,
            slice_spacing=slice_spacing,
        )
        self.radiation = radiation

        if isinstance(incoming, ParameterBeam):
            new_mu = incoming.mu.expand(*vector_shape, 7).clone()
            new_cov = incoming.cov.expand(*vector_shape, 7, 7).clone()
            delta_p = (gammas - gamma0_p[..., None]) / gamma0_p[..., None]
            if is_time_dependent:
                coords = torch.stack([x, px, y, py, delta_p], dim=-1).reshape(
                    *vector_shape, num_slices * num_particles, 5
                )
            else:
                coords = torch.stack([x, px, y, py, delta_p], dim=-1)
            mean_coords = coords.mean(dim=-2)
            centered = coords - mean_coords.unsqueeze(-2)
            cov_5 = (centered.mT @ centered) / max(1, coords.shape[-2] - 1)

            new_mu[..., :4] = mean_coords[..., :4]
            new_mu[..., 5] = mean_coords[..., 4]
            new_cov[..., :4, :4] = cov_5[..., :4, :4]
            new_cov[..., :4, 5] = cov_5[..., :4, 4]
            new_cov[..., 5, :4] = cov_5[..., 4, :4]
            new_cov[..., 5, 5] = cov_5[..., 4, 4]
            new_s = incoming.s + self.length

            outgoing = ParameterBeam(
                new_mu,
                new_cov,
                incoming.energy,
                total_charge=incoming.total_charge,
                s=new_s,
                species=incoming.species.clone(),
            )
        elif isinstance(incoming, ParticleBeam):
            new_particles = incoming.particles.expand(*vector_shape, -1, 7).clone()
            idx_s = num_slices // 2 if is_time_dependent else None
            new_particles[..., 0] = x[..., idx_s, :] if is_time_dependent else x
            new_particles[..., 1] = px[..., idx_s, :] if is_time_dependent else px
            new_particles[..., 2] = y[..., idx_s, :] if is_time_dependent else y
            new_particles[..., 3] = py[..., idx_s, :] if is_time_dependent else py
            p_out = (gammas - gamma0_p[..., None]) / gamma0_p[..., None]
            new_particles[..., 5] = p_out[..., idx_s, :] if is_time_dependent else p_out
            new_s = incoming.s + self.length
            outgoing = ParticleBeam(
                new_particles,
                incoming.energy,
                particle_charges=incoming.particle_charges,
                survival_probabilities=incoming.survival_probabilities,
                s=new_s,
                species=incoming.species.clone(),
            )
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

        outgoing.radiation = radiation
        outgoing._fel_particles = (x, px, y, py, gammas, theta)
        return outgoing

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:purple", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "period", "kx", "ky"]
