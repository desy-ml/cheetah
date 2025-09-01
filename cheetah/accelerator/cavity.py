import math
from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from scipy import constants

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.track_methods import drift_matrix
from cheetah.utils import UniqueNameGenerator, compute_relativistic_factors

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
    :param cavity_type: Type of the cavity.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    """

    def __init__(
        self,
        length: torch.Tensor,
        voltage: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        frequency: torch.Tensor | None = None,
        cavity_type: Literal["standing_wave", "traveling_wave"] = "standing_wave",
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = length

        self.register_buffer_or_parameter(
            "voltage",
            voltage if voltage is not None else torch.tensor(0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "phase",
            phase if phase is not None else torch.tensor(0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "frequency",
            frequency if frequency is not None else torch.tensor(0.0, **factory_kwargs),
        )

        self.cavity_type = cavity_type

    @property
    def is_active(self) -> torch.Tensor:
        return self.voltage.any()

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        is_active_cavity = torch.logical_and(
            self.voltage != 0, (self.phase / 90) % 2 != 1.0
        )
        return torch.where(
            is_active_cavity.view(*is_active_cavity.shape, 1, 1),
            self._cavity_rmatrix(energy, species),
            drift_matrix(self.length, energy=energy, species=species),
        )

    def track(self, incoming: Beam) -> Beam:
        gamma0, igamma2, beta0 = compute_relativistic_factors(
            incoming.energy, incoming.species.mass_eV
        )

        phi = torch.deg2rad(self.phase)

        tm = self.first_order_transfer_map(incoming.energy, incoming.species)
        if isinstance(incoming, ParameterBeam):
            outgoing_mu = (tm @ incoming.mu.unsqueeze(-1)).squeeze(-1)
            outgoing_cov = tm @ incoming.cov @ tm.mT
        else:  # ParticleBeam
            outgoing_particles = incoming.particles @ tm.mT
        delta_energy = (
            -self.voltage * phi.cos() * incoming.species.num_elementary_charges
        )

        T566 = 1.5 * self.length * igamma2 / beta0**3
        T556 = self.length.new_zeros(())
        T555 = self.length.new_zeros(())

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
                    (-incoming.mu[..., 4] * beta0 * k + phi).cos() - phi.cos()
                )
                outgoing_cov[..., 5, 5] = incoming.cov[..., 5, 5]
            else:  # ParticleBeam
                outgoing_particles[..., 5] = incoming.particles[..., 5] * (
                    incoming.energy * beta0 / (outgoing_energy * beta1)
                ).unsqueeze(-1) + (
                    self.voltage * beta0 / (outgoing_energy * beta1)
                ).unsqueeze(
                    -1
                ) * (
                    (
                        -incoming.particles[..., 4] * (beta0 * k).unsqueeze(-1)
                        + phi.unsqueeze(-1)
                    ).cos()
                    - phi.cos().unsqueeze(-1)
                )

            dgamma = self.voltage / incoming.species.mass_eV
            if (delta_energy > 0).any():
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
                    * phi.sin()
                    / (beta1**3 * gamma1**3 * (gamma0 - gamma1).square())
                )
                T555 = (
                    beta0.square()
                    * k.square()
                    * self.length
                    * dgamma
                    / 2.0
                    * (
                        dgamma
                        * (
                            2 * gamma0 * gamma1**3 * (beta0 * beta1**3 - 1)
                            + gamma0.square()
                            + 3 * gamma1.square()
                            - 2
                        )
                        / (beta1**3 * gamma1**3 * (gamma0 - gamma1) ** 3)
                        * phi.sin().square()
                        - (gamma1 * gamma0 * (beta1 * beta0 - 1) + 1)
                        / (beta1 * gamma1 * (gamma0 - gamma1).square())
                        * phi.cos()
                    )
                )

            if isinstance(incoming, ParameterBeam):
                outgoing_mu[..., 4] = outgoing_mu[..., 4] + (
                    T566 * incoming.mu[..., 5].square()
                    + T556 * incoming.mu[..., 4] * incoming.mu[..., 5]
                    + T555 * incoming.mu[..., 4].square()
                )
                outgoing_cov[..., 4, 4] = (
                    T566 * incoming.cov[..., 5, 5].square()
                    + T556 * incoming.cov[..., 4, 5] * incoming.cov[..., 5, 5]
                    + T555 * incoming.cov[..., 4, 4].square()
                )
                outgoing_cov[..., 4, 5] = (
                    T566 * incoming.cov[..., 5, 5].square()
                    + T556 * incoming.cov[..., 4, 5] * incoming.cov[..., 5, 5]
                    + T555 * incoming.cov[..., 4, 4].square()
                )
                outgoing_cov[..., 5, 4] = outgoing_cov[..., 4, 5]
            else:  # ParticleBeam
                outgoing_particles[..., 4] = outgoing_particles[..., 4] + (
                    T566.unsqueeze(-1) * incoming.particles[..., 5].square()
                    + T556.unsqueeze(-1)
                    * incoming.particles[..., 4]
                    * incoming.particles[..., 5]
                    + T555.unsqueeze(-1) * incoming.particles[..., 4].square()
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

        phi = self.phase.deg2rad()
        effective_voltage = self.voltage * -species.num_elementary_charges
        delta_energy = effective_voltage * phi.cos()
        # Comment from Ocelot: Pure pi-standing-wave case
        Ei = energy / species.mass_eV
        Ef = (energy + delta_energy) / species.mass_eV
        dE = delta_energy / species.mass_eV
        Ep = dE / self.length  # Derivative of the energy
        assert torch.all(Ei > 0), "Initial energy must be larger than 0"

        k = 2 * torch.pi * self.frequency / constants.speed_of_light

        if self.cavity_type == "standing_wave":
            alpha = math.sqrt(0.125) / phi.cos() * (Ef / Ei).log()
            beta0 = (1 - Ei.square().reciprocal()).sqrt()
            beta1 = (1 - Ef.square().reciprocal()).sqrt()

            r11 = alpha.cos() - math.sqrt(2.0) * phi.cos() * alpha.sin()

            # In Ocelot r12 is defined as below only if abs(Ep) > 10, and self.length
            # otherwise. This is implemented differently here to achieve results
            # closer to Bmad.
            r12 = (
                math.sqrt(8.0) * energy / effective_voltage * alpha.sin() * self.length
            )

            r21 = (
                -Ep
                / Ef
                * (phi.cos() / math.sqrt(2.0) + math.sqrt(0.125) / phi.cos())
                * alpha.sin()
            )

            r22 = Ei / Ef * (alpha.cos() + math.sqrt(2.0) * phi.cos() * alpha.sin())

            r56 = (
                -self.length / (Ef.square() * Ei * beta1) * (Ef + Ei) / (beta1 + beta0)
            )
            r55 = 1.0 + (
                k
                * self.length
                * beta0
                * effective_voltage
                / species.mass_eV
                * phi.sin()
                * (Ei * Ef * (beta0 * beta1 - 1) + 1)
                / (beta1 * Ef * dE.square())
            )
            r66 = Ei / Ef * beta0 / beta1
            r65 = k * phi.sin() * effective_voltage / (Ef * beta1 * species.mass_eV)

        elif self.cavity_type == "traveling_wave":
            # Reference paper: Rosenzweig and Serafini, PhysRevE, Vol.49, p.1599,(1994)
            f = Ei / dE * (dE / Ei).log1p()

            vector_shape = torch.broadcast_shapes(
                self.length.shape, f.shape, Ei.shape, Ef.shape
            )

            M_body = torch.eye(2, **factory_kwargs).expand(*vector_shape, 2, 2).clone()
            M_body[..., 0, 1] = self.length * f
            M_body[..., 1, 1] = Ei / Ef

            M_f_entry = (
                torch.eye(2, **factory_kwargs).expand(*vector_shape, 2, 2).clone()
            )
            M_f_entry[..., 1, 0] = -dE / (2 * self.length * Ei)

            M_f_exit = (
                torch.eye(2, **factory_kwargs).expand(*vector_shape, 2, 2).clone()
            )
            M_f_exit[..., 1, 0] = dE / (2 * self.length * Ef)

            M_combined = M_f_exit @ M_body @ M_f_entry

            r11 = M_combined[..., 0, 0]
            r12 = M_combined[..., 0, 1]
            r21 = M_combined[..., 1, 0]
            r22 = M_combined[..., 1, 1]
            r55 = self.length.new_ones(())
            r56 = self.length.new_zeros(())
            r66 = r22
            r65 = k * phi.sin() * effective_voltage / (Ef * species.mass_eV)
        else:
            raise ValueError(f"Invalid cavity type: {self.cavity_type}")

        # Make sure that all matrix elements have the same shape
        r11, r12, r21, r22, r55, r56, r65, r66 = torch.broadcast_tensors(
            r11, r12, r21, r22, r55, r56, r65, r66
        )

        R = torch.eye(7, **factory_kwargs).expand((*r11.shape, 7, 7)).clone()
        R[
            ...,
            (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5),
            (0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5),
        ] = torch.stack(
            [r11, r12, r21, r22, r11, r12, r21, r22, r55, r56, r65, r66], dim=-1
        )

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
        return super().defining_features + [
            "length",
            "voltage",
            "phase",
            "frequency",
            "cavity_type",
        ]
