from typing import Optional, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.constants import physical_constants
from torch import Size, nn

from cheetah.particles import Beam, ParameterBeam, ParticleBeam
from cheetah.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class Cavity(Element):
    """
    Accelerating cavity in a particle accelerator.

    :param length: Length in meters.
    :param voltage: Voltage of the cavity in volts.
    :param phase: Phase of the cavity in degrees.
    :param frequency: Frequency of the cavity in Hz.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[torch.Tensor, nn.Parameter],
        voltage: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        phase: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        frequency: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.length = torch.as_tensor(length, **factory_kwargs)
        self.voltage = (
            torch.as_tensor(voltage, **factory_kwargs)
            if voltage is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        self.phase = (
            torch.as_tensor(phase, **factory_kwargs)
            if phase is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        self.frequency = (
            torch.as_tensor(frequency, **factory_kwargs)
            if frequency is not None
            else torch.tensor(0.0, **factory_kwargs)
        )

    @property
    def is_active(self) -> bool:
        return torch.any(self.voltage != 0)

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        # There used to be a check for voltage > 0 here, where the cavity transfer map
        # was only computed for the elements with voltage > 0 and a basermatrix was
        # used otherwise. This was removed because it was causing issues with the
        # vectorisation, but I am not sure it is okay to remove.
        tm = self._cavity_rmatrix(energy)

        return tm

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the cavity. The input can be a `ParameterBeam` or a
        `ParticleBeam`. For a cavity, this does a little more than just the transfer map
        multiplication done by most elements.

        :param incoming: Beam of particles entering the element.
        :return: Beam of particles exiting the element.
        """
        if incoming is Beam.empty:
            return incoming
        elif isinstance(incoming, (ParameterBeam, ParticleBeam)):
            return self._track_beam(incoming)
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

    def _track_beam(self, incoming: Beam) -> Beam:
        device = self.length.device
        dtype = self.length.dtype

        beta0 = torch.full_like(self.length, 1.0)
        igamma2 = torch.full_like(self.length, 0.0)
        g0 = torch.full_like(self.length, 1e10)

        mask = incoming.energy != 0
        g0[mask] = incoming.energy[mask] / electron_mass_eV.to(
            device=device, dtype=dtype
        )
        igamma2[mask] = 1 / g0[mask] ** 2
        beta0[mask] = torch.sqrt(1 - igamma2[mask])

        phi = torch.deg2rad(self.phase)

        tm = self.transfer_map(incoming.energy)
        if isinstance(incoming, ParameterBeam):
            outgoing_mu = torch.matmul(tm, incoming._mu.unsqueeze(-1)).squeeze(-1)
            outgoing_cov = torch.matmul(
                tm, torch.matmul(incoming._cov, tm.transpose(-2, -1))
            )
        else:  # ParticleBeam
            outgoing_particles = torch.matmul(incoming.particles, tm.transpose(-2, -1))
        delta_energy = self.voltage * torch.cos(phi)

        T566 = 1.5 * self.length * igamma2 / beta0**3
        T556 = torch.full_like(self.length, 0.0)
        T555 = torch.full_like(self.length, 0.0)

        if torch.any(incoming.energy + delta_energy > 0):
            k = 2 * torch.pi * self.frequency / constants.speed_of_light
            outgoing_energy = incoming.energy + delta_energy
            g1 = outgoing_energy / electron_mass_eV
            beta1 = torch.sqrt(1 - 1 / g1**2)

            if isinstance(incoming, ParameterBeam):
                outgoing_mu[..., 5] = incoming._mu[..., 5] * incoming.energy * beta0 / (
                    outgoing_energy * beta1
                ) + self.voltage * beta0 / (outgoing_energy * beta1) * (
                    torch.cos(-incoming._mu[..., 4] * beta0 * k + phi) - torch.cos(phi)
                )
                outgoing_cov[..., 5, 5] = incoming._cov[..., 5, 5]
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

            dgamma = self.voltage / electron_mass_eV
            if torch.any(delta_energy > 0):
                T566 = (
                    self.length
                    * (beta0**3 * g0**3 - beta1**3 * g1**3)
                    / (2 * beta0 * beta1**3 * g0 * (g0 - g1) * g1**3)
                )
                T556 = (
                    beta0
                    * k
                    * self.length
                    * dgamma
                    * g0
                    * (beta1**3 * g1**3 + beta0 * (g0 - g1**3))
                    * torch.sin(phi)
                    / (beta1**3 * g1**3 * (g0 - g1) ** 2)
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
                            2 * g0 * g1**3 * (beta0 * beta1**3 - 1)
                            + g0**2
                            + 3 * g1**2
                            - 2
                        )
                        / (beta1**3 * g1**3 * (g0 - g1) ** 3)
                        * torch.sin(phi) ** 2
                        - (g1 * g0 * (beta1 * beta0 - 1) + 1)
                        / (beta1 * g1 * (g0 - g1) ** 2)
                        * torch.cos(phi)
                    )
                )

            if isinstance(incoming, ParameterBeam):
                outgoing_mu[..., 4] = outgoing_mu[..., 4] + (
                    T566 * incoming._mu[..., 5] ** 2
                    + T556 * incoming._mu[..., 4] * incoming._mu[..., 5]
                    + T555 * incoming._mu[..., 4] ** 2
                )
                outgoing_cov[..., 4, 4] = (
                    T566 * incoming._cov[..., 5, 5] ** 2
                    + T556 * incoming._cov[..., 4, 5] * incoming._cov[..., 5, 5]
                    + T555 * incoming._cov[..., 4, 4] ** 2
                )
                outgoing_cov[..., 4, 5] = (
                    T566 * incoming._cov[..., 5, 5] ** 2
                    + T556 * incoming._cov[..., 4, 5] * incoming._cov[..., 5, 5]
                    + T555 * incoming._cov[..., 4, 4] ** 2
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
                outgoing_mu,
                outgoing_cov,
                outgoing_energy,
                total_charge=incoming.total_charge,
                device=outgoing_mu.device,
                dtype=outgoing_mu.dtype,
            )
            return outgoing
        else:  # ParticleBeam
            outgoing = ParticleBeam(
                outgoing_particles,
                outgoing_energy,
                particle_charges=incoming.particle_charges,
                device=outgoing_particles.device,
                dtype=outgoing_particles.dtype,
            )
            return outgoing

    def _cavity_rmatrix(self, energy: torch.Tensor) -> torch.Tensor:
        """Produces an R-matrix for a cavity when it is on, i.e. voltage > 0.0."""
        device = self.length.device
        dtype = self.length.dtype

        phi = torch.deg2rad(self.phase)
        delta_energy = self.voltage * torch.cos(phi)
        # Comment from Ocelot: Pure pi-standing-wave case
        eta = torch.tensor(1.0, device=device, dtype=dtype)
        Ei = energy / electron_mass_eV
        Ef = (energy + delta_energy) / electron_mass_eV
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

        r56 = torch.tensor(0.0)
        beta0 = torch.tensor(1.0)
        beta1 = torch.tensor(1.0)

        k = 2 * torch.pi * self.frequency / torch.tensor(constants.speed_of_light)
        r55_cor = 0.0
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
                / electron_mass_eV
                * torch.sin(phi)
                * (g0 * g1 * (beta0 * beta1 - 1) + 1)
                / (beta1 * g1 * (g0 - g1) ** 2)
            )

        r66 = Ei / Ef * beta0 / beta1
        r65 = k * torch.sin(phi) * self.voltage / (Ef * beta1 * electron_mass_eV)

        R = torch.eye(7, device=device, dtype=dtype).repeat((*self.length.shape, 1, 1))
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

    def broadcast(self, shape: Size) -> Element:
        return self.__class__(
            length=self.length.repeat(shape),
            voltage=self.voltage.repeat(shape),
            phase=self.phase.repeat(shape),
            frequency=self.frequency.repeat(shape),
            name=self.name,
        )

    def split(self, resolution: torch.Tensor) -> list[Element]:
        # TODO: Implement splitting for cavity properly, for now just returns the
        # element itself
        return [self]

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle(
            (s, 0), self.length[0], height, color="gold", alpha=alpha, zorder=2
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
