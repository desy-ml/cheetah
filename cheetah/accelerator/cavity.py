import math
from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from scipy import constants

from cheetah.accelerator.drift import Drift
from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.track_methods import drift_matrix
from cheetah.utils import (
    UniqueNameGenerator,
    cache_transfer_map,
    compute_relativistic_factors,
)
from cheetah.utils.autograd import log1pdiv

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Cavity(Element):
    """
    Accelerating cavity in a particle accelerator.

    NOTE: When computing the gradient on cavities with exactly zero energy gain, it is
        possible that some gradient components are lost. This is unlikely to cause
        (therefore do not hesitate to have cavities with zero voltage in your lattice),
        but this note is here to make you aware of this possibility.

    :param length: Total element length in meters, including inactive end regions.
    :param voltage: Voltage of the cavity in volts. NOTE: This assumes the physical
        voltage. Positive voltage will accelerate electron-like particles.
        For particle with charge `n * e`, the energy gain on crest will be
        `n * voltage`.
    :param phase: Phase of the cavity in degrees.
    :param frequency: Frequency of the cavity in Hz.
    :param cavity_type: Type of the cavity. Traveling-wave transverse transport uses
        exact reference momenta, including at non-ultrarelativistic energies.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment. If `None` (default), a warning is raised for
        invalid names. Set to `True` to sanitise, or `False` to silence the warning.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element (e.g. control-system addresses or PVs). This information is *not* used
        in simulation and may contain any extra data the user wants to store along with
        the lattice. See :doc:`/getting_started/control_systems` for more information.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.
    :param active_length: Active RF length in meters. If provided, it must be positive
        and no greater than `length`. The remaining length is split into equal entrance
        and exit drifts. If `None`, the full `length` is active, preserving the existing
        model.
    """

    def __init__(
        self,
        length: torch.Tensor,
        voltage: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        frequency: torch.Tensor | None = None,
        cavity_type: Literal["standing_wave", "traveling_wave"] = "standing_wave",
        name: str | None = None,
        sanitize_name: bool | None = None,
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        active_length: torch.Tensor | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            name=name, sanitize_name=sanitize_name, metadata=metadata, **factory_kwargs
        )

        self.length = length
        if active_length is None:
            self.register_buffer("active_length", None)
        else:
            self.register_buffer_or_parameter("active_length", active_length)

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
    def is_active(self) -> bool:
        return (self.voltage != 0).any().item()

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    @cache_transfer_map
    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        matrix = self._cavity_rmatrix(energy, species)
        if self.active_length is None:
            return matrix
        end_length = (self.length - self.active_length) / 2.0
        outgoing_energy = (
            energy
            - self.voltage * self.phase.deg2rad().cos() * species.num_elementary_charges
        )
        return (
            drift_matrix(end_length, outgoing_energy, species)
            @ matrix
            @ drift_matrix(end_length, energy, species)
        )

    def track(self, incoming: Beam) -> Beam:
        """Track through the active RF region and any symmetric inactive end drifts."""
        if self.active_length is None:
            return self._track_active(
                incoming,
                self.length,
                self.first_order_transfer_map(incoming.energy, incoming.species),
            )
        matrix = self._cavity_rmatrix(incoming.energy, incoming.species)
        drift = Drift(length=(self.length - self.active_length) / 2.0)
        outgoing = self._track_active(drift.track(incoming), self.active_length, matrix)
        return drift.track(outgoing)

    def _track_active(
        self, incoming: Beam, length: torch.Tensor, tm: torch.Tensor
    ) -> Beam:
        """Track the active RF region using its existing nonlinear cavity model."""
        gamma0, igamma2, beta0 = compute_relativistic_factors(
            incoming.energy, incoming.species.mass_eV
        )

        phi = self.phase.deg2rad()

        if isinstance(incoming, ParameterBeam):
            outgoing_mu = (tm @ incoming.mu.unsqueeze(-1)).squeeze(-1)
            outgoing_cov = tm @ incoming.cov @ tm.mT
        else:  # ParticleBeam
            outgoing_particles = incoming.particles @ tm.mT
        delta_energy = (
            self.voltage * phi.cos() * incoming.species.num_elementary_charges * -1
        )

        T566 = 1.5 * length * igamma2 / beta0.pow(3)
        T556 = self.length.new_zeros(())
        T555 = self.length.new_zeros(())

        k = 2.0 * torch.pi * self.frequency / constants.speed_of_light
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
                (
                    -incoming.particles[..., 4] * beta0.unsqueeze(-1) * k.unsqueeze(-1)
                    + phi.unsqueeze(-1)
                ).cos()
                - phi.cos().unsqueeze(-1)
            )

        dgamma = self.voltage / incoming.species.mass_eV

        # TODO: Branch stays in for now because no proper limit exists and would
        # probably require a different model altogether.
        if (delta_energy > 0).any():
            T566 = (
                length
                * (beta0.pow(3) * gamma0.pow(3) - beta1.pow(3) * gamma1.pow(3))
                / (
                    2.0
                    * beta0
                    * beta1.pow(3)
                    * gamma0
                    * (gamma0 - gamma1)
                    * gamma1.pow(3)
                )
            )
            T556 = (
                beta0
                * k
                * length
                * dgamma
                * gamma0
                * (beta1.pow(3) * gamma1.pow(3) + beta0 * (gamma0 - gamma1.pow(3)))
                * phi.sin()
                / (beta1.pow(3) * gamma1.pow(3) * (gamma0 - gamma1).square())
            )
            T555 = (
                beta0.square()
                * k.square()
                * length
                * dgamma
                / 2.0
                * (
                    dgamma
                    * (
                        2.0 * gamma0 * gamma1.pow(3) * (beta0 * beta1.pow(3) - 1.0)
                        + gamma0.square()
                        + 3.0 * gamma1.square()
                        - 2.0
                    )
                    / (beta1.pow(3) * gamma1.pow(3) * (gamma0 - gamma1).pow(3))
                    * phi.sin().square()
                    - (gamma1 * gamma0 * (beta1 * beta0 - 1.0) + 1.0)
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
                s=incoming.s + length,
                species=incoming.species,
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
                s=incoming.s + length,
                species=incoming.species,
                device=outgoing_particles.device,
                dtype=outgoing_particles.dtype,
            )
            return outgoing

    def _cavity_rmatrix(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        """Produces an R-matrix for a cavity when it is on, i.e. voltage > 0.0."""
        assert torch.all(energy > 0), "Initial energy must be larger than 0"
        factory_kwargs = {"device": self.length.device, "dtype": self.length.dtype}
        length = self.length if self.active_length is None else self.active_length
        if self.active_length is not None and not torch.all(
            torch.isfinite(length)
            & (length > 0.0)
            & (length <= self.length)
            & torch.isfinite(self.length)
        ):
            raise ValueError(
                "active_length must be finite and satisfy 0 < active_length <= length"
            )

        phi = self.phase.deg2rad()
        effective_voltage = -self.voltage * species.num_elementary_charges
        delta_energy = effective_voltage * phi.cos()

        Ei = energy / species.mass_eV
        dE = delta_energy / species.mass_eV
        Ef = Ei + dE
        Ep = dE / length  # Derivative of the energy

        k = 2 * torch.pi * self.frequency / constants.speed_of_light

        if self.cavity_type == "standing_wave":
            alpha = (
                math.sqrt(0.125)
                * effective_voltage
                / energy
                * log1pdiv(delta_energy / energy)
            )
            beta0 = (1 - Ei.square().reciprocal()).sqrt()
            beta1 = (1 - Ef.square().reciprocal()).sqrt()

            r11 = alpha.cos() - math.sqrt(2.0) * phi.cos() * alpha.sin()
            r12 = (alpha / torch.pi).sinc() * log1pdiv(delta_energy / energy) * length
            r21 = -(
                effective_voltage
                / ((energy + delta_energy) * math.sqrt(2.0) * length)
                * (0.5 + phi.cos().square())
                * alpha.sin()
            )

            r22 = Ei / Ef * (alpha.cos() + math.sqrt(2.0) * phi.cos() * alpha.sin())

            # TODO: Branch stays in for now because no proper limit exists and would
            # probably require a different model altogether.
            r55 = 1.0 + (
                k
                * length
                * beta0
                * phi.tan()
                * (Ei * Ef * (beta0 * beta1 - 1) + 1)
                / (beta1 * Ef * dE)
            ).where(dE != 0.0, 0.0)
            r56 = -length / (Ef.square() * Ei * beta1) * (Ef + Ei) / (beta1 + beta0)
            r65 = k * phi.sin() * effective_voltage / (beta1 * (energy + delta_energy))
            r66 = Ei / Ef * beta0 / beta1

        elif self.cavity_type == "traveling_wave":
            # Reference paper: Rosenzweig and Serafini, PhysRevE, Vol.49, p.1599,(1994)
            assert torch.all(Ei > 1.0) and torch.all(
                Ef > 1.0
            ), "Traveling-wave entrance and exit energies must exceed the rest energy"
            Pi = (Ei.square() - 1.0).sqrt()
            Pf = (Ef.square() - 1.0).sqrt()
            log_ratio_slope = (1.0 + (Ei + Ef) / (Pi + Pf)) / (Ei + Pi)
            vector_shape = torch.broadcast_shapes(
                length.shape, dE.shape, Ei.shape, Ef.shape
            )

            M_body = torch.eye(2, **factory_kwargs).repeat((*vector_shape, 1, 1))
            # Stable form of `L * Pi * log((Ef + Pf) / (Ei + Pi)) / dE`
            M_body[..., 0, 1] = (
                length * Pi * log_ratio_slope * log1pdiv(dE * log_ratio_slope)
            )
            M_body[..., 1, 1] = Pi / Pf

            M_f_entry = torch.eye(2, **factory_kwargs).repeat((*vector_shape, 1, 1))
            M_f_entry[..., 1, 0] = -Ep / (2 * Pi)

            M_f_exit = torch.eye(2, **factory_kwargs).repeat((*vector_shape, 1, 1))
            M_f_exit[..., 1, 0] = Ep / (2 * Pf)

            M_combined = M_f_exit @ M_body @ M_f_entry

            r11 = M_combined[..., 0, 0]
            r12 = M_combined[..., 0, 1]
            r21 = M_combined[..., 1, 0]
            r22 = M_combined[..., 1, 1]
            r55 = self.length.new_ones(())
            r56 = self.length.new_zeros(())
            r65 = k * torch.sin(phi) * effective_voltage / (energy + delta_energy)
            # Preserve the existing longitudinal model
            r66 = Ei / Ef * (1.0 + 0.5 * (dE / Ei).log1p())
        else:
            raise ValueError(f"Invalid cavity type: {self.cavity_type}")

        # Make sure that all matrix elements have the same shape
        r11, r12, r21, r22, r55, r56, r65, r66 = torch.broadcast_tensors(
            r11, r12, r21, r22, r55, r56, r65, r66
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
        R[..., 4, 4] = r55
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
        return super().defining_features + [
            "length",
            "voltage",
            "phase",
            "frequency",
            "cavity_type",
            "active_length",
        ]
