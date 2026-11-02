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

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class RadiationField:
    """
    Representation of a transverse optical or X-ray radiation field in an FEL.

    :param field: Complex transverse electric field envelope on a 2D grid with shape
        `(H, W)` in V/m.
    :param wavelength: Radiation wavelength in meters.
    :param extent: Transverse half-size of the grid in meters.
    """

    def __init__(
        self,
        field: torch.Tensor,
        wavelength: torch.Tensor,
        extent: torch.Tensor,
    ) -> None:
        self.field = field
        self.wavelength = wavelength
        self.extent = extent

    @property
    def power(self) -> torch.Tensor:
        """Total radiation power in Watts integrated over the transverse grid."""
        grid_shape = self.field.shape[-2:]
        dx = 2.0 * self.extent / (grid_shape[-1] - 1)
        dy = 2.0 * self.extent / (grid_shape[-2] - 1)
        z0 = mu_0 * speed_of_light
        return (0.5 / z0) * self.field.abs().square().sum(dim=(-2, -1)) * dx * dy

    @property
    def peak_intensity(self) -> torch.Tensor:
        """Peak intensity of the radiation field in W/m^2."""
        z0 = mu_0 * speed_of_light
        return (0.5 / z0) * self.field.abs().square().amax(dim=(-2, -1))

    def diffract(self, distance: torch.Tensor) -> "RadiationField":
        """
        Propagate the radiation field in free space over `distance` in meters using
        paraxial diffraction in Fourier space.
        """
        grid_shape = self.field.shape[-2:]
        dx = 2.0 * self.extent / (grid_shape[-1] - 1)
        dy = 2.0 * self.extent / (grid_shape[-2] - 1)
        k_r = 2.0 * torch.pi / self.wavelength

        kx = (
            2.0
            * torch.pi
            * torch.fft.fftfreq(grid_shape[-1], d=dx, device=self.field.device)
        )
        ky = (
            2.0
            * torch.pi
            * torch.fft.fftfreq(grid_shape[-2], d=dy, device=self.field.device)
        )
        k_perp2 = ky[:, None].square() + kx[None, :].square()

        phase = torch.exp(-0.5j * (k_perp2 / k_r) * distance)
        new_field = torch.fft.ifft2(torch.fft.fft2(self.field) * phase)
        return RadiationField(
            field=new_field, wavelength=self.wavelength, extent=self.extent
        )

    def clone(self) -> "RadiationField":
        """Create a clone of the radiation field."""
        return RadiationField(
            field=self.field.clone(),
            wavelength=self.wavelength.clone(),
            extent=self.extent.clone(),
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
        jj = torch.special.bessel_j0(xi) - torch.special.bessel_j1(xi)
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

        # Transverse grid setup
        grid_extent = (
            self.grid_extent
            if self.grid_extent is not None
            else torch.tensor(2e-4, **factory_kwargs)
        )
        ngrid = self.grid_shape[0]
        dx = 2.0 * grid_extent / (ngrid - 1)
        dy = dx

        complex_dtype = (
            torch.complex128
            if incoming.energy.dtype == torch.float64
            else torch.complex64
        )

        # Radiation field initialization
        if hasattr(incoming, "radiation") and incoming.radiation is not None:
            radiation = incoming.radiation.diffract(torch.tensor(0.0, **factory_kwargs))
            field = radiation.field.to(dtype=complex_dtype)
        else:
            seed_power = (
                self.seed_power
                if self.seed_power is not None
                else torch.tensor(1.090578e5, **factory_kwargs)
            )
            w0 = grid_extent / 4.0
            e0 = torch.sqrt(4.0 * z0 * seed_power / (math.pi * w0.square()))

            x_coords = torch.linspace(
                -grid_extent, grid_extent, ngrid, **factory_kwargs
            )
            y_coords = torch.linspace(
                -grid_extent, grid_extent, ngrid, **factory_kwargs
            )
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            field = (
                e0 * torch.exp(-(grid_x.square() + grid_y.square()) / w0.square())
            ).to(dtype=complex_dtype)
            radiation = RadiationField(
                field=field, wavelength=lambda0, extent=grid_extent
            )

        # Macroparticles setup
        nbins = self.nbins
        num_transverse = max(1, self.num_macroparticles // nbins)
        num_particles = num_transverse * nbins

        current = (
            self.current
            if self.current is not None
            else torch.tensor(5000.0, **factory_kwargs)
        )
        bunch = (
            self.bunching
            if self.bunching is not None
            else torch.tensor(1e-6, **factory_kwargs)
        )

        if hasattr(incoming, "_fel_particles") and incoming._fel_particles is not None:
            x, px, y, py, gammas, theta = incoming._fel_particles
        elif isinstance(incoming, ParticleBeam):
            x = incoming.particles[..., 0].contiguous()
            px = incoming.particles[..., 1].contiguous()
            y = incoming.particles[..., 2].contiguous()
            py = incoming.particles[..., 3].contiguous()
            gammas = gamma0 * (1.0 + incoming.particles[..., 5].contiguous())
            num_parts = x.shape[-1]
            theta0 = torch.linspace(
                0.0, 2.0 * math.pi, num_parts + 1, **factory_kwargs
            )[:-1]
            theta = theta0 + 2.0 * bunch * torch.sin(theta0)
        elif isinstance(incoming, ParameterBeam):
            cov_x = incoming.cov[..., :2, :2].squeeze()
            cov_y = incoming.cov[..., 2:4, 2:4].squeeze()

            gen = torch.Generator(device=incoming.energy.device).manual_seed(42)
            mvn_x = torch.distributions.MultivariateNormal(
                torch.zeros(2, **factory_kwargs), cov_x
            )
            mvn_y = torch.distributions.MultivariateNormal(
                torch.zeros(2, **factory_kwargs), cov_y
            )

            sample_x = mvn_x.sample((num_transverse,))
            sample_y = mvn_y.sample((num_transverse,))

            x = sample_x[:, 0].repeat_interleave(nbins)
            px = sample_x[:, 1].repeat_interleave(nbins)
            y = sample_y[:, 0].repeat_interleave(nbins)
            py = sample_y[:, 1].repeat_interleave(nbins)

            delgam = (
                incoming.sigma_p * gamma0
                if hasattr(incoming, "sigma_p")
                else torch.tensor(5.0, **factory_kwargs)
            )
            gammas_trans = (
                gamma0
                + torch.randn(num_transverse, generator=gen, **factory_kwargs) * delgam
            )
            gammas = gammas_trans.repeat_interleave(nbins)

            theta_bins = torch.linspace(
                0.0, 2.0 * math.pi, nbins + 1, **factory_kwargs
            )[:-1]
            theta0 = theta_bins.repeat(num_transverse)
            theta = theta0 + 2.0 * bunch * torch.sin(theta0)
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

        # Spectral grid for diffraction
        kx = 2.0 * torch.pi * torch.fft.fftfreq(ngrid, d=dx, device=field.device)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(ngrid, d=dy, device=field.device)
        grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing="ij")
        k_perp2 = grid_kx.square() + grid_ky.square()

        def diffract_half(fld: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
            phase = torch.exp(-0.5j * (k_perp2 / k_r) * dist)
            return torch.fft.ifft2(torch.fft.fft2(fld) * phase)

        def interp_field(
            fld: torch.Tensor, x_p: torch.Tensor, y_p: torch.Tensor
        ) -> torch.Tensor:
            norm_x = (x_p / grid_extent).to(dtype=fld.real.dtype)
            norm_y = (y_p / grid_extent).to(dtype=fld.real.dtype)
            grid_sample_coords = (
                torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)
            )
            real_part = fld.real.unsqueeze(0).unsqueeze(0)
            imag_part = fld.imag.unsqueeze(0).unsqueeze(0)
            e_real = torch.nn.functional.grid_sample(
                real_part, grid_sample_coords, align_corners=True, mode="bilinear"
            ).squeeze()
            e_imag = torch.nn.functional.grid_sample(
                imag_part, grid_sample_coords, align_corners=True, mode="bilinear"
            ).squeeze()
            return torch.complex(e_real, e_imag)

        def deposit_source(
            x_p: torch.Tensor,
            y_p: torch.Tensor,
            gammas_p: torch.Tensor,
            theta_p: torch.Tensor,
        ) -> torch.Tensor:
            weight = (gamma0 / gammas_p).to(dtype=complex_dtype) * torch.exp(
                -1.0j * theta_p.to(dtype=complex_dtype)
            )
            gx = (x_p + grid_extent) / dx
            gy = (y_p + grid_extent) / dy

            gx0 = gx.floor().long()
            gy0 = gy.floor().long()
            gx1 = gx0 + 1
            gy1 = gy0 + 1

            fx1 = (gx - gx0).to(dtype=factory_kwargs["dtype"])
            fx0 = 1.0 - fx1
            fy1 = (gy - gy0).to(dtype=factory_kwargs["dtype"])
            fy0 = 1.0 - fy1

            mask = (gx0 >= 0) & (gx1 < ngrid) & (gy0 >= 0) & (gy1 < ngrid)
            flat_grid_bunch = torch.zeros(
                ngrid * ngrid, dtype=complex_dtype, device=field.device
            )

            w00 = (fx0 * fy0).to(dtype=complex_dtype) * weight
            w01 = (fx1 * fy0).to(dtype=complex_dtype) * weight
            w10 = (fx0 * fy1).to(dtype=complex_dtype) * weight
            w11 = (fx1 * fy1).to(dtype=complex_dtype) * weight

            for w, ix, iy in [
                (w00, gx0, gy0),
                (w01, gx1, gy0),
                (w10, gx0, gy1),
                (w11, gx1, gy1),
            ]:
                idx = iy[mask] * ngrid + ix[mask]
                flat_grid_bunch.scatter_add_(0, idx, w[mask])

            j_bunch = flat_grid_bunch.reshape(ngrid, ngrid) * (
                current / (num_particles * dx * dy)
            )
            return (0.5j * z0 * aw * jj / gamma0) * j_bunch

        step_size = (
            self.step_size
            if self.step_size is not None
            else torch.tensor(0.12, **factory_kwargs)
        )
        n_steps = max(1, int(round((self.length / step_size).item())))
        dz = self.length / n_steps

        k_beta = aw * k_u / gamma0

        for _ in range(n_steps):
            field = diffract_half(field, 0.5 * dz)
            e_p = interp_field(field, x, y)

            dgamma_dz = (
                -(elementary_charge / (electron_mass * speed_of_light**2))
                * (aw * jj / gammas)
                * (e_p * torch.exp(1.0j * theta)).real
            )
            dtheta_dz = k_u - (k_r / (2.0 * gammas.square())) * (
                1.0 + aw.square() + gammas.square() * (px.square() + py.square())
            )

            s_source = deposit_source(x, y, gammas, theta)

            gammas = gammas + dgamma_dz * dz
            theta = theta + dtheta_dz * dz
            field = field + s_source * dz

            field = diffract_half(field, 0.5 * dz)

            x = x + px * dz
            cos_b = torch.cos(k_beta * dz)
            sin_b = torch.sin(k_beta * dz)
            y_new = y * cos_b + (py / k_beta) * sin_b
            py = -y * k_beta * sin_b + py * cos_b
            y = y_new

        radiation = RadiationField(field=field, wavelength=lambda0, extent=grid_extent)
        self.radiation = radiation

        if isinstance(incoming, ParameterBeam):
            new_mu = incoming.mu.clone()
            new_cov = incoming.cov.clone()
            delta_gamma = gammas - gamma0
            mean_delta = delta_gamma.mean()
            std_delta = delta_gamma.std()

            new_mu[..., 5] = mean_delta / gamma0
            new_cov[..., 5, 5] = (std_delta / gamma0).square()
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
            new_particles = incoming.particles.clone()
            new_particles[..., : len(x), 0] = x
            new_particles[..., : len(px), 1] = px
            new_particles[..., : len(y), 2] = y
            new_particles[..., : len(py), 3] = py
            new_particles[..., : len(gammas), 5] = (gammas - gamma0) / gamma0
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
