import torch

from cheetah.particles.beam import Beam
from cheetah.particles.particle_beam import ParticleBeam
from cheetah.particles.species import Species


class ParameterBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param mu: Mu vector of the beam with shape `(..., 7)`.
    :param cov: Covariance matrix of the beam with shape `(..., 7, 7)`.
    :param energy: Reference energy of the beam in eV.
    :param total_charge: Total charge of the beam in C.
    :param s: Position along the beamline of the reference particle in meters.
    :param species: Particle species of the beam. Defaults to electron.
    :param device: Device that the beam creates its tensors on.
    :param dtype: Data type of the tensors created by the beam.
    """

    UNVECTORIZED_NUM_ATTR_DIMS = Beam.UNVECTORIZED_NUM_ATTR_DIMS | {
        "mu": 1,
        "cov": 2,
    }

    def __init__(
        self,
        mu: torch.Tensor,
        cov: torch.Tensor,
        energy: torch.Tensor,
        total_charge: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.species = (
            species if species is not None else Species("electron", **factory_kwargs)
        )

        self.register_buffer_or_parameter("mu", mu)
        self.register_buffer_or_parameter("cov", cov)

        self.register_buffer_or_parameter("energy", energy)
        self.register_buffer_or_parameter(
            "total_charge",
            (
                total_charge
                if total_charge is not None
                else torch.tensor(0.0, **factory_kwargs)
            ),
        )
        self.register_buffer_or_parameter(
            "s", s if s is not None else torch.tensor(0.0, **factory_kwargs)
        )

    @classmethod
    def from_parameters(
        cls,
        mu_x: torch.Tensor | None = None,
        mu_px: torch.Tensor | None = None,
        mu_y: torch.Tensor | None = None,
        mu_py: torch.Tensor | None = None,
        mu_tau: torch.Tensor | None = None,
        mu_p: torch.Tensor | None = None,
        sigma_x: torch.Tensor | None = None,
        sigma_px: torch.Tensor | None = None,
        sigma_y: torch.Tensor | None = None,
        sigma_py: torch.Tensor | None = None,
        sigma_tau: torch.Tensor | None = None,
        sigma_p: torch.Tensor | None = None,
        cov_xpx: torch.Tensor | None = None,
        cov_ypy: torch.Tensor | None = None,
        cov_taup: torch.Tensor | None = None,
        cov_xp: torch.Tensor | None = None,
        cov_pxp: torch.Tensor | None = None,
        cov_yp: torch.Tensor | None = None,
        cov_pyp: torch.Tensor | None = None,
        cov_xy: torch.Tensor | None = None,
        cov_xpy: torch.Tensor | None = None,
        cov_xtau: torch.Tensor | None = None,
        cov_pxy: torch.Tensor | None = None,
        cov_pxpy: torch.Tensor | None = None,
        cov_pxtau: torch.Tensor | None = None,
        cov_ytau: torch.Tensor | None = None,
        cov_pytau: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParameterBeam":
        factory_kwargs = {"device": device, "dtype": dtype}

        # Set default values without function call in function signature
        mu_x = mu_x if mu_x is not None else torch.tensor(0.0, **factory_kwargs)
        mu_px = mu_px if mu_px is not None else torch.tensor(0.0, **factory_kwargs)
        mu_y = mu_y if mu_y is not None else torch.tensor(0.0, **factory_kwargs)
        mu_py = mu_py if mu_py is not None else torch.tensor(0.0, **factory_kwargs)
        mu_tau = mu_tau if mu_tau is not None else torch.tensor(0.0, **factory_kwargs)
        mu_p = mu_p if mu_p is not None else torch.tensor(0.0, **factory_kwargs)
        sigma_x = (
            sigma_x if sigma_x is not None else torch.tensor(175e-6, **factory_kwargs)
        )
        sigma_px = (
            sigma_px if sigma_px is not None else torch.tensor(4e-6, **factory_kwargs)
        )
        sigma_y = (
            sigma_y if sigma_y is not None else torch.tensor(175e-6, **factory_kwargs)
        )
        sigma_py = (
            sigma_py if sigma_py is not None else torch.tensor(4e-6, **factory_kwargs)
        )
        sigma_tau = (
            sigma_tau if sigma_tau is not None else torch.tensor(8e-6, **factory_kwargs)
        )
        sigma_p = (
            sigma_p if sigma_p is not None else torch.tensor(2e-3, **factory_kwargs)
        )
        cov_xpx = (
            cov_xpx if cov_xpx is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_ypy = (
            cov_ypy if cov_ypy is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_taup = (
            cov_taup if cov_taup is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_xp = cov_xp if cov_xp is not None else torch.tensor(0.0, **factory_kwargs)
        cov_pxp = (
            cov_pxp if cov_pxp is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_yp = cov_yp if cov_yp is not None else torch.tensor(0.0, **factory_kwargs)
        cov_pyp = (
            cov_pyp if cov_pyp is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_xy = cov_xy if cov_xy is not None else torch.tensor(0.0, **factory_kwargs)
        cov_xpy = (
            cov_xpy if cov_xpy is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_xtau = (
            cov_xtau if cov_xtau is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_pxy = (
            cov_pxy if cov_pxy is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_pxpy = (
            cov_pxpy if cov_pxpy is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_pxtau = (
            cov_pxtau if cov_pxtau is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_ytau = (
            cov_ytau if cov_ytau is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_pytau = (
            cov_pytau if cov_pytau is not None else torch.tensor(0.0, **factory_kwargs)
        )
        energy = energy if energy is not None else torch.tensor(1e8, **factory_kwargs)
        total_charge = (
            total_charge
            if total_charge is not None
            else torch.tensor(0.0, **factory_kwargs)
        )

        mu_x, mu_px, mu_y, mu_py, mu_tau, mu_p = torch.broadcast_tensors(
            mu_x, mu_px, mu_y, mu_py, mu_tau, mu_p
        )
        mu = torch.stack(
            [mu_x, mu_px, mu_y, mu_py, mu_tau, mu_p, torch.ones_like(mu_x)],
            dim=-1,
        )

        (
            sigma_x,
            cov_xpx,
            sigma_px,
            sigma_y,
            cov_ypy,
            sigma_py,
            sigma_tau,
            cov_taup,
            sigma_p,
            cov_xp,
            cov_pxp,
            cov_yp,
            cov_pyp,
            cov_xy,
            cov_xpy,
            cov_xtau,
            cov_pxy,
            cov_pxpy,
            cov_pxtau,
            cov_ytau,
            cov_pytau,
        ) = torch.broadcast_tensors(
            sigma_x,
            cov_xpx,
            sigma_px,
            sigma_y,
            cov_ypy,
            sigma_py,
            sigma_tau,
            cov_taup,
            sigma_p,
            cov_xp,
            cov_pxp,
            cov_yp,
            cov_pyp,
            cov_xy,
            cov_xpy,
            cov_xtau,
            cov_pxy,
            cov_pxpy,
            cov_pxtau,
            cov_ytau,
            cov_pytau,
        )
        cov = torch.zeros(*sigma_x.shape, 7, 7, **factory_kwargs)
        cov[..., 0, 0] = sigma_x.square()
        cov[..., 0, 1] = cov_xpx
        cov[..., 1, 0] = cov_xpx
        cov[..., 1, 1] = sigma_px.square()
        cov[..., 2, 2] = sigma_y.square()
        cov[..., 2, 3] = cov_ypy
        cov[..., 3, 2] = cov_ypy
        cov[..., 3, 3] = sigma_py.square()
        cov[..., 4, 4] = sigma_tau.square()
        cov[..., 4, 5] = cov_taup
        cov[..., 5, 4] = cov_taup
        cov[..., 5, 5] = sigma_p.square()
        cov[..., 0, 5] = cov_xp
        cov[..., 5, 0] = cov_xp
        cov[..., 1, 5] = cov_pxp
        cov[..., 5, 1] = cov_pxp
        cov[..., 2, 5] = cov_yp
        cov[..., 5, 2] = cov_yp
        cov[..., 3, 5] = cov_pyp
        cov[..., 5, 3] = cov_pyp
        cov[..., 0, 2] = cov_xy
        cov[..., 2, 0] = cov_xy
        cov[..., 0, 3] = cov_xpy
        cov[..., 3, 0] = cov_xpy
        cov[..., 0, 4] = cov_xtau
        cov[..., 4, 0] = cov_xtau
        cov[..., 1, 2] = cov_pxy
        cov[..., 2, 1] = cov_pxy
        cov[..., 1, 3] = cov_pxpy
        cov[..., 3, 1] = cov_pxpy
        cov[..., 1, 4] = cov_pxtau
        cov[..., 4, 1] = cov_pxtau
        cov[..., 2, 4] = cov_ytau
        cov[..., 4, 2] = cov_ytau
        cov[..., 3, 4] = cov_pytau
        cov[..., 4, 3] = cov_pytau

        # Assert that the covariance matrix is positive (semi)-definite
        try:
            torch.linalg.cholesky(cov[..., :6, :6])
        except RuntimeError as e:
            raise ValueError(
                "The covariance matrix of the beam must be positive definite. Please "
                "check the input parameters to ensure that they are consistent."
            ) from e

        return cls(
            mu=mu,
            cov=cov,
            energy=energy,
            total_charge=total_charge,
            s=s,
            species=species,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_twiss(
        cls,
        beta_x: torch.Tensor | None = None,
        alpha_x: torch.Tensor | None = None,
        emittance_x: torch.Tensor | None = None,
        beta_y: torch.Tensor | None = None,
        alpha_y: torch.Tensor | None = None,
        emittance_y: torch.Tensor | None = None,
        sigma_tau: torch.Tensor | None = None,
        sigma_p: torch.Tensor | None = None,
        cov_taup: torch.Tensor | None = None,
        dispersion_x: torch.Tensor | None = None,
        dispersion_px: torch.Tensor | None = None,
        dispersion_y: torch.Tensor | None = None,
        dispersion_py: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParameterBeam":
        factory_kwargs = {"device": device, "dtype": dtype}

        # Set default values without function call in function signature
        beta_x = beta_x if beta_x is not None else torch.tensor(1.0, **factory_kwargs)
        alpha_x = (
            alpha_x if alpha_x is not None else torch.tensor(0.0, **factory_kwargs)
        )
        emittance_x = (
            emittance_x
            if emittance_x is not None
            else torch.tensor(7.1971891e-13, **factory_kwargs)
        )
        beta_y = beta_y if beta_y is not None else torch.tensor(1.0, **factory_kwargs)
        alpha_y = (
            alpha_y if alpha_y is not None else torch.tensor(0.0, **factory_kwargs)
        )
        emittance_y = (
            emittance_y
            if emittance_y is not None
            else torch.tensor(7.1971891e-13, **factory_kwargs)
        )
        sigma_tau = (
            sigma_tau if sigma_tau is not None else torch.tensor(1e-6, **factory_kwargs)
        )
        sigma_p = (
            sigma_p if sigma_p is not None else torch.tensor(1e-6, **factory_kwargs)
        )
        cov_taup = (
            cov_taup if cov_taup is not None else torch.tensor(0.0, **factory_kwargs)
        )
        dispersion_x = (
            dispersion_x
            if dispersion_x is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        dispersion_px = (
            dispersion_px
            if dispersion_px is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        dispersion_y = (
            dispersion_y
            if dispersion_y is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        dispersion_py = (
            dispersion_py
            if dispersion_py is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        energy = energy if energy is not None else torch.tensor(1e8, **factory_kwargs)
        total_charge = (
            total_charge
            if total_charge is not None
            else torch.tensor(0.0, **factory_kwargs)
        )

        assert (
            beta_x > 0
        ).all(), "Beta function in x direction must be larger than 0 everywhere."
        assert (
            beta_y > 0
        ).all(), "Beta function in y direction must be larger than 0 everywhere."

        sigma_x = (
            emittance_x * beta_x + dispersion_x.square() * sigma_p.square()
        ).sqrt()
        sigma_px = (
            emittance_x * (1 + alpha_x.square()) / beta_x
            + dispersion_px.square() * sigma_p.square()
        ).sqrt()
        sigma_y = (
            emittance_y * beta_y + dispersion_y.square() * sigma_p.square()
        ).sqrt()
        sigma_py = (
            emittance_y * (1 + alpha_y.square()) / beta_y
            + dispersion_py.square() * sigma_p.square()
        ).sqrt()
        cov_xpx = (
            -emittance_x * alpha_x + dispersion_x * dispersion_px * sigma_p.square()
        )
        cov_ypy = (
            -emittance_y * alpha_y + dispersion_y * dispersion_py * sigma_p.square()
        )
        cov_xp = dispersion_x * sigma_p.square()
        cov_pxp = dispersion_px * sigma_p.square()
        cov_yp = dispersion_y * sigma_p.square()
        cov_pyp = dispersion_py * sigma_p.square()

        return cls.from_parameters(
            sigma_x=sigma_x,
            sigma_px=sigma_px,
            sigma_y=sigma_y,
            sigma_py=sigma_py,
            sigma_tau=sigma_tau,
            sigma_p=sigma_p,
            energy=energy,
            cov_taup=cov_taup,
            cov_xpx=cov_xpx,
            cov_ypy=cov_ypy,
            cov_xp=cov_xp,
            cov_pxp=cov_pxp,
            cov_yp=cov_yp,
            cov_pyp=cov_pyp,
            total_charge=total_charge,
            s=s,
            species=species,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_ocelot(
        cls, parray, device: torch.device = None, dtype: torch.dtype = None
    ) -> "ParameterBeam":
        """Load an Ocelot ParticleArray `parray` as a Cheetah Beam."""
        factory_kwargs = {
            "device": device or torch.get_default_device(),
            "dtype": dtype or torch.get_default_dtype(),
        }

        mu = torch.ones(7, **factory_kwargs)
        mu[:6] = torch.as_tensor(parray.rparticles.mean(axis=1), **factory_kwargs)

        cov = torch.zeros(7, 7, **factory_kwargs)
        cov[:6, :6] = torch.as_tensor(parray.rparticles, **factory_kwargs).cov()

        energy = 1e9 * torch.as_tensor(parray.E, **factory_kwargs)
        total_charge = torch.as_tensor(parray.q_array, **factory_kwargs).sum()

        return cls(
            mu=mu,
            cov=cov,
            energy=energy,
            total_charge=total_charge,
            species=Species("electron", **factory_kwargs),
            **factory_kwargs,
        )

    @classmethod
    def from_astra(
        cls, path: str, device: torch.device = None, dtype: torch.dtype = None
    ) -> "ParameterBeam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        from cheetah.converters.astra import from_astrabeam

        factory_kwargs = {
            "device": device or torch.get_default_device(),
            "dtype": dtype or torch.get_default_dtype(),
        }

        particles, energy, particle_charges = from_astrabeam(path)

        mu = torch.ones(7, **factory_kwargs)
        mu[:6] = torch.as_tensor(particles.mean(axis=0), **factory_kwargs)

        cov = torch.zeros(7, 7, **factory_kwargs)
        cov[:6, :6] = torch.as_tensor(particles, **factory_kwargs).T.cov()

        energy = torch.as_tensor(energy, **factory_kwargs)
        total_charge = torch.as_tensor(particle_charges, **factory_kwargs).sum()

        return cls(
            mu=mu,
            cov=cov,
            energy=energy,
            total_charge=total_charge,
            species=Species("electron"),
            **factory_kwargs,
        )

    def transformed_to(
        self,
        mu_x: torch.Tensor | None = None,
        mu_px: torch.Tensor | None = None,
        mu_y: torch.Tensor | None = None,
        mu_py: torch.Tensor | None = None,
        mu_tau: torch.Tensor | None = None,
        mu_p: torch.Tensor | None = None,
        sigma_x: torch.Tensor | None = None,
        sigma_px: torch.Tensor | None = None,
        sigma_y: torch.Tensor | None = None,
        sigma_py: torch.Tensor | None = None,
        sigma_tau: torch.Tensor | None = None,
        sigma_p: torch.Tensor | None = None,
        cov_xpx: torch.Tensor | None = None,
        cov_ypy: torch.Tensor | None = None,
        cov_taup: torch.Tensor | None = None,
        cov_xp: torch.Tensor | None = None,
        cov_pxp: torch.Tensor | None = None,
        cov_yp: torch.Tensor | None = None,
        cov_pyp: torch.Tensor | None = None,
        cov_xy: torch.Tensor | None = None,
        cov_xpy: torch.Tensor | None = None,
        cov_xtau: torch.Tensor | None = None,
        cov_pxy: torch.Tensor | None = None,
        cov_pxpy: torch.Tensor | None = None,
        cov_pxtau: torch.Tensor | None = None,
        cov_ytau: torch.Tensor | None = None,
        cov_pytau: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParameterBeam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on yp, dimensionless.
        :param mu_tau: Center of the particle distribution on tau in meters.
        :param mu_p: Center of the particle distribution on p, dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_py: Sigma of the particle distribution in py direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param cov_xpx: Covariance between x and px.
        :param cov_ypy: Covariance between y and py.
        :param cov_taup: Covariance between tau and p.
        :param cov_xp: Covariance between x and p.
        :param cov_pxp: Covariance between px and p.
        :param cov_yp: Covariance between y and p.
        :param cov_pyp: Covariance between py and p.
        :param cov_xy: Covariance between x and y.
        :param cov_xpy: Covariance between x and py.
        :param cov_xtau: Covariance between x and tau.
        :param cov_pxy: Covariance between px and y.
        :param cov_pxpy: Covariance between px and py.
        :param cov_pxtau: Covariance between px and tau.
        :param cov_ytau: Covariance between y and tau.
        :param cov_pytau: Covariance between py and tau.
        :param energy: Reference energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param species: Particle species of the beam.
        :param device: Device that the beam creates its tensors on.
        :param dtype: Data type of the tensors created by the beam.
        """
        return self.__class__.from_parameters(
            mu_x=mu_x if mu_x is not None else self.mu_x,
            mu_px=mu_px if mu_px is not None else self.mu_px,
            mu_y=mu_y if mu_y is not None else self.mu_y,
            mu_py=mu_py if mu_py is not None else self.mu_py,
            mu_tau=mu_tau if mu_tau is not None else self.mu_tau,
            mu_p=mu_p if mu_p is not None else self.mu_p,
            sigma_x=sigma_x if sigma_x is not None else self.sigma_x,
            sigma_px=sigma_px if sigma_px is not None else self.sigma_px,
            sigma_y=sigma_y if sigma_y is not None else self.sigma_y,
            sigma_py=sigma_py if sigma_py is not None else self.sigma_py,
            sigma_tau=sigma_tau if sigma_tau is not None else self.sigma_tau,
            sigma_p=sigma_p if sigma_p is not None else self.sigma_p,
            cov_xpx=cov_xpx if cov_xpx is not None else self.cov_xpx,
            cov_ypy=cov_ypy if cov_ypy is not None else self.cov_ypy,
            cov_taup=cov_taup if cov_taup is not None else self.cov_taup,
            cov_xp=cov_xp if cov_xp is not None else self.cov_xp,
            cov_pxp=cov_pxp if cov_pxp is not None else self.cov_pxp,
            cov_yp=cov_yp if cov_yp is not None else self.cov_yp,
            cov_pyp=cov_pyp if cov_pyp is not None else self.cov_pyp,
            cov_xy=cov_xy if cov_xy is not None else self.cov_xy,
            cov_xpy=cov_xpy if cov_xpy is not None else self.cov_xpy,
            cov_xtau=cov_xtau if cov_xtau is not None else self.cov_xtau,
            cov_pxy=cov_pxy if cov_pxy is not None else self.cov_pxy,
            cov_pxpy=cov_pxpy if cov_pxpy is not None else self.cov_pxpy,
            cov_pxtau=cov_pxtau if cov_pxtau is not None else self.cov_pxtau,
            cov_ytau=cov_ytau if cov_ytau is not None else self.cov_ytau,
            cov_pytau=cov_pytau if cov_pytau is not None else self.cov_pytau,
            energy=energy if energy is not None else self.energy,
            total_charge=(
                total_charge if total_charge is not None else self.total_charge
            ),
            s=self.s,
            species=species if species is not None else self.species,
            device=device if device is not None else self.mu.device,
            dtype=dtype if dtype is not None else self.mu.dtype,
        )

    def as_particle_beam(self, num_particles: int) -> "ParticleBeam":  # noqa: F821
        """
        Convert this beam to a `ParticleBeam` beam with `num_particles` particles.

        :param num_particles: Number of macro particles to create.
        :return: `ParticleBeam` with `num_particles` particles and the same parameters
            as this beam.
        """
        from cheetah.particles.particle_beam import ParticleBeam  # No circular import

        return ParticleBeam.from_distribution(
            num_particles=num_particles,
            mu=self.mu[..., :6],
            cov=self.cov[..., :6, :6],
            energy=self.energy,
            total_charge=self.total_charge,
            s=self.s,
            species=self.species,
            device=self.mu.device,
            dtype=self.mu.dtype,
        )

    def linspaced(self, num_particles: int) -> "ParticleBeam":  # noqa: F821
        """
        Create a `ParticleBeam` beam with the same parameters as this beam and
        `num_particles` particles evenly distributed in the beam.

        :param num_particles: Number of particles to create.
        :return: `ParticleBeam` with `num_particles` particles.
        """
        from cheetah.particles.particle_beam import ParticleBeam  # No circular import

        return ParticleBeam.make_linspaced(
            num_particles=num_particles,
            mu_x=self.mu_x,
            mu_y=self.mu_y,
            mu_px=self.mu_px,
            mu_py=self.mu_py,
            mu_tau=self.mu_tau,
            mu_p=self.mu_p,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            sigma_px=self.sigma_px,
            sigma_py=self.sigma_py,
            sigma_tau=self.sigma_tau,
            sigma_p=self.sigma_p,
            energy=self.energy,
            total_charge=self.total_charge,
            s=self.s,
            species=self.species,
            device=self.mu.device,
            dtype=self.mu.dtype,
        )

    @property
    def mu_x(self) -> torch.Tensor:
        return self.mu[..., 0]

    @property
    def sigma_x(self) -> torch.Tensor:
        return self.cov[..., 0, 0].sqrt()

    @property
    def mu_px(self) -> torch.Tensor:
        return self.mu[..., 1]

    @property
    def sigma_px(self) -> torch.Tensor:
        return self.cov[..., 1, 1].sqrt()

    @property
    def mu_y(self) -> torch.Tensor:
        return self.mu[..., 2]

    @property
    def sigma_y(self) -> torch.Tensor:
        return self.cov[..., 2, 2].sqrt()

    @property
    def mu_py(self) -> torch.Tensor:
        return self.mu[..., 3]

    @property
    def sigma_py(self) -> torch.Tensor:
        return self.cov[..., 3, 3].sqrt()

    @property
    def mu_tau(self) -> torch.Tensor:
        return self.mu[..., 4]

    @property
    def sigma_tau(self) -> torch.Tensor:
        return self.cov[..., 4, 4].sqrt()

    @property
    def mu_p(self) -> torch.Tensor:
        return self.mu[..., 5]

    @property
    def sigma_p(self) -> torch.Tensor:
        return self.cov[..., 5, 5].sqrt()

    @property
    def cov_xpx(self) -> torch.Tensor:
        return self.cov[..., 0, 1]

    @property
    def cov_ypy(self) -> torch.Tensor:
        return self.cov[..., 2, 3]

    @property
    def cov_taup(self) -> torch.Tensor:
        return self.cov[..., 4, 5]

    @property
    def cov_xp(self) -> torch.Tensor:
        return self.cov[..., 0, 5]

    @property
    def cov_pxp(self) -> torch.Tensor:
        return self.cov[..., 1, 5]

    @property
    def cov_yp(self) -> torch.Tensor:
        return self.cov[..., 2, 5]

    @property
    def cov_pyp(self) -> torch.Tensor:
        return self.cov[..., 3, 5]

    @property
    def cov_xy(self) -> torch.Tensor:
        return self.cov[..., 0, 2]

    @property
    def cov_xpy(self) -> torch.Tensor:
        return self.cov[..., 0, 3]

    @property
    def cov_xtau(self) -> torch.Tensor:
        return self.cov[..., 0, 4]

    @property
    def cov_pxy(self) -> torch.Tensor:
        return self.cov[..., 1, 2]

    @property
    def cov_pxpy(self) -> torch.Tensor:
        return self.cov[..., 1, 3]

    @property
    def cov_pxtau(self) -> torch.Tensor:
        return self.cov[..., 1, 4]

    @property
    def cov_ytau(self) -> torch.Tensor:
        return self.cov[..., 2, 4]

    @property
    def cov_pytau(self) -> torch.Tensor:
        return self.cov[..., 3, 4]

    @property
    def defining_features(self):
        return ["mu", "cov", "energy", "total_charge", "s", "species"]

    def clone(self) -> "ParameterBeam":
        return self.__class__(
            mu=self.mu.clone(),
            cov=self.cov.clone(),
            energy=self.energy.clone(),
            total_charge=self.total_charge.clone(),
            s=self.s.clone(),
            species=self.species.clone(),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu={repr(self.mu)}, "
            + f"cov={repr(self.cov)}, "
            + f"energy={repr(self.energy)}, "
            + f"total_charge={repr(self.total_charge)}, "
            + f"s={repr(self.s)}, "
            + f"species={repr(self.species)})"
        )
