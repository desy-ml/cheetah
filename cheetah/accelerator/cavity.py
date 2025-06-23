import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from scipy import constants

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.track_methods import base_rmatrix
from cheetah.utils import (
    UniqueNameGenerator,
    compute_relativistic_factors,
    verify_device_and_dtype,
)

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Cavity(Element):
    """
    Accelerating cavity in a particle accelerator.

    :param length: Length in meters.
    :param voltage: Voltage of the cavity in volts. NOTE: This assumes the physical
        voltage. Positive voltage will accelerate electron-like particles.
        For particle with charge `n * e`, the energy gain on crest will be
        `n * voltage`.
    :param phase: Phase of the cavity in degrees.
    :param frequency: Frequency of the cavity in Hz.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python
        variable name. This is needed if you want to use the `segment.element_name`
        syntax to access the element in a segment.
    """

    def __init__(
        self,
        length: torch.Tensor,
        voltage: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        frequency: torch.Tensor | None = None,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [length, voltage, phase, frequency], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        self.register_buffer_or_parameter(
            "voltage",
            torch.as_tensor(voltage if voltage is not None else 0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "phase",
            torch.as_tensor(phase if phase is not None else 0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "frequency",
            torch.as_tensor(
                frequency if frequency is not None else 0.0, **factory_kwargs
            ),
        )

    @property
    def is_active(self) -> bool:
        return torch.any(self.voltage != 0).item()

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        return torch.where(
            (self.voltage != 0).unsqueeze(-1).unsqueeze(-1),
            self._cavity_rmatrix(energy, species),
            base_rmatrix(
                length=self.length,
                k1=torch.zeros_like(self.length),
                hx=torch.zeros_like(self.length),
                species=species,
                tilt=torch.zeros_like(self.length),
                energy=energy,
            ),
        )

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the cavity. The input can be a `ParameterBeam` or a
        `ParticleBeam`. For a cavity, this does a little more than just the transfer map
        multiplication done by most elements.

        :param incoming: Beam of particles entering the element.
        :return: Beam of particles exiting the element.
        """
        if isinstance(incoming, (ParameterBeam, ParticleBeam)):
            return self._track_beam(incoming)
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

    def _track_beam(self, incoming: Beam) -> Beam:
        """
        Track particles through the cavity. The input can be a `ParameterBeam` or a
        `ParticleBeam`.
        """
        gamma0, igamma2, beta0 = compute_relativistic_factors(
            incoming.energy, incoming.species.mass_eV
        )

        phi = torch.deg2rad(self.phase)

        tm = self.transfer_map(incoming.energy, incoming.species)
        if isinstance(incoming, ParameterBeam):
            outgoing_mu = (tm @ incoming.mu.unsqueeze(-1)).squeeze(-1)
            outgoing_cov = tm @ incoming.cov @ tm.transpose(-2, -1)
        else:  # ParticleBeam
            outgoing_particles = incoming.particles @ tm.transpose(-2, -1)
        delta_energy = (
            self.voltage * torch.cos(phi) * incoming.species.num_elementary_charges * -1
        )

        T566 = 1.5 * self.length * igamma2 / beta0**3
        T556 = torch.full_like(self.length, 0.0)
        T555 = torch.full_like(self.length, 0.0)

        if torch.any(incoming.energy + delta_energy > 0):
            k = 2 * torch.pi * self.frequency / constants.speed_of_light
            outgoing_energy = incoming.energy + delta_energy
            gamma1, _, beta1 = compute_relativistic_factors(
                outgoing_energy, incoming.species.mass_eV
            )

            if isinstance(incoming, ParameterBeam):
                outgoing_mu[..., 5] = incoming.mu[..., 5] * incoming.energy * beta0 / (
                    outgoing_energy * beta1
                ) + self.voltage * beta0 / (outgoing_energy * beta1) * (
                    torch.cos(-incoming.mu[..., 4] * beta0 * k + phi) - torch.cos(phi)
                )
                outgoing_cov[..., 5, 5] = incoming.cov[..., 5, 5]
            else:  # ParticleBeam
                outgoing_particles[..., 5] = incoming.particles[
                    ..., 5
                ] * incoming.energy.unsqueeze(-1) * beta0.unsqueeze(-1) / (
                    outgoing_energy.unsqueeze(-1) * beta1.unsqueeze(-1)
                ) + self.voltage.unsqueeze(
                    -1
                ) * beta0.unsqueeze(
                    -1
                ) / (
                    outgoing_energy.unsqueeze(-1) * beta1.unsqueeze(-1)
                ) * (
                    torch.cos(
                        -1
                        * incoming.particles[..., 4]
                        * beta0.unsqueeze(-1)
                        * k.unsqueeze(-1)
                        + phi.unsqueeze(-1)
                    )
                    - torch.cos(phi).unsqueeze(-1)
                )

            dgamma = self.voltage / incoming.species.mass_eV
            if torch.any(delta_energy > 0):
                T566 = (
                    self.length
                    * (beta0**3 * gamma0**3 - beta1**3 * gamma1**3)
                    / (2 * beta0 * beta1**3 * gamma0 * (gamma0 - gamma1) * gamma1**3)
                )
                T556 = (
                    beta0
                    * k
                    * self.length
                    * dgamma
                    * gamma0
                    * (beta1**3 * gamma1**3 + beta0 * (gamma0 - gamma1**3))
                    * torch.sin(phi)
                    / (beta1**3 * gamma1**3 * (gamma0 - gamma1) ** 2)
                )
                T555 = (
                    beta0**2
                    * k**2
                    * self.length
                    * dgamma
                    / 2.0
                    * (
                        dgamma
                        * (
                            2 * gamma0 * gamma1**3 * (beta0 * beta1**3 - 1)
                            + gamma0**2
                            + 3 * gamma1**2
                            - 2
                        )
                        / (beta1**3 * gamma1**3 * (gamma0 - gamma1) ** 3)
                        * torch.sin(phi) ** 2
                        - (gamma1 * gamma0 * (beta1 * beta0 - 1) + 1)
                        / (beta1 * gamma1 * (gamma0 - gamma1) ** 2)
                        * torch.cos(phi)
                    )
                )

            if isinstance(incoming, ParameterBeam):
                outgoing_mu[..., 4] = outgoing_mu[..., 4] + (
                    T566 * incoming.mu[..., 5] ** 2
                    + T556 * incoming.mu[..., 4] * incoming.mu[..., 5]
                    + T555 * incoming.mu[..., 4] ** 2
                )
                outgoing_cov[..., 4, 4] = (
                    T566 * incoming.cov[..., 5, 5] ** 2
                    + T556 * incoming.cov[..., 4, 5] * incoming.cov[..., 5, 5]
                    + T555 * incoming.cov[..., 4, 4] ** 2
                )
                outgoing_cov[..., 4, 5] = (
                    T566 * incoming.cov[..., 5, 5] ** 2
                    + T556 * incoming.cov[..., 4, 5] * incoming.cov[..., 5, 5]
                    + T555 * incoming.cov[..., 4, 4] ** 2
                )
                outgoing_cov[..., 5, 4] = outgoing_cov[..., 4, 5]
            else:  # ParticleBeam
                outgoing_particles[..., 4] = outgoing_particles[..., 4] + (
                    T566.unsqueeze(-1) * incoming.particles[..., 5] ** 2
                    + T556.unsqueeze(-1)
                    * incoming.particles[..., 4]
                    * incoming.particles[..., 5]
                    + T555.unsqueeze(-1) * incoming.particles[..., 4] ** 2
                )

        if isinstance(incoming, ParameterBeam):
            outgoing = ParameterBeam(
                mu=outgoing_mu,
                cov=outgoing_cov,
                energy=outgoing_energy,
                total_charge=incoming.total_charge,
                s=incoming.s + self.length,
                device=outgoing_mu.device,
                dtype=outgoing_mu.dtype,
            )
            return outgoing
        else:  # ParticleBeam
            outgoing = ParticleBeam(
                particles=outgoing_particles,
                energy=outgoing_energy,
                particle_charges=incoming.particle_charges,
                survival_probabilities=incoming.survival_probabilities,
                s=incoming.s + self.length,
                device=outgoing_particles.device,
                dtype=outgoing_particles.dtype,
            )
            return outgoing

    def _cavity_rmatrix(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        """Produces an R-matrix for a cavity when it is on, i.e. voltage > 0.0."""
        factory_kwargs = {"device": self.length.device, "dtype": self.length.dtype}

        phi = torch.deg2rad(self.phase)
        delta_energy = (
            self.voltage * torch.cos(phi) * species.num_elementary_charges * -1
        )
        # Comment from Ocelot: Pure pi-standing-wave case
        eta = torch.tensor(1.0, **factory_kwargs)
        Ei = energy / species.mass_eV
        Ef = (energy + delta_energy) / species.mass_eV
        Ep = (Ef - Ei) / self.length  # Derivative of the energy
        assert torch.all(Ei > 0), "Initial energy must be larger than 0"

        alpha = torch.sqrt(eta / 8) / torch.cos(phi) * torch.log(Ef / Ei)

        r11 = torch.cos(alpha) - torch.sqrt(2 / eta) * torch.cos(phi) * torch.sin(alpha)

        # In Ocelot r12 is defined as below only if abs(Ep) > 10, and self.length
        # otherwise. This is implemented differently here in order to achieve results
        # closer to Bmad.
        r12 = torch.sqrt(8 / eta) * Ei / Ep * torch.cos(phi) * torch.sin(alpha)

        r21 = (
            -Ep
            / Ef
            * (
                torch.cos(phi) / torch.sqrt(2 * eta)
                + torch.sqrt(eta / 8) / torch.cos(phi)
            )
            * torch.sin(alpha)
        )

        r22 = (
            Ei
            / Ef
            * (
                torch.cos(alpha)
                + torch.sqrt(2 / eta) * torch.cos(phi) * torch.sin(alpha)
            )
        )

        r56 = torch.tensor(0.0, **factory_kwargs)
        beta0 = torch.tensor(1.0, **factory_kwargs)
        beta1 = torch.tensor(1.0, **factory_kwargs)

        k = 2 * torch.pi * self.frequency / constants.speed_of_light
        r55_cor = torch.tensor(0.0, **factory_kwargs)
        if torch.any((self.voltage != 0) & (energy != 0)):  # TODO: Do we need this if?
            beta0 = torch.sqrt(1 - 1 / Ei**2)
            beta1 = torch.sqrt(1 - 1 / Ef**2)

            r56 = -self.length / (Ef**2 * Ei * beta1) * (Ef + Ei) / (beta1 + beta0)
            g0 = Ei
            g1 = Ef
            r55_cor = (
                k
                * self.length
                * beta0
                * self.voltage
                / species.mass_eV
                * torch.sin(phi)
                * (g0 * g1 * (beta0 * beta1 - 1) + 1)
                / (beta1 * g1 * (g0 - g1) ** 2)
            )

        r66 = Ei / Ef * beta0 / beta1
        r65 = k * torch.sin(phi) * self.voltage / (Ef * beta1 * species.mass_eV)

        # Make sure that all matrix elements have the same shape
        r11, r12, r21, r22, r55_cor, r56, r65, r66 = torch.broadcast_tensors(
            r11, r12, r21, r22, r55_cor, r56, r65, r66
        )

        R = torch.eye(7, **factory_kwargs).repeat((*r11.shape, 1, 1))
        R[..., 0, 0] = r11
        R[..., 0, 1] = r12
        R[..., 1, 0] = r21
        R[..., 1, 1] = r22
        R[..., 2, 2] = r11
        R[..., 2, 3] = r12
        R[..., 3, 2] = r21
        R[..., 3, 3] = r22
        R[..., 4, 4] = 1 + r55_cor
        R[..., 4, 5] = r56
        R[..., 5, 4] = r65
        R[..., 5, 5] = r66

        return R

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="gold", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "voltage", "phase", "frequency"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"voltage={repr(self.voltage)}, "
            + f"phase={repr(self.phase)}, "
            + f"frequency={repr(self.frequency)}, "
            + f"name={repr(self.name)})"
        )
