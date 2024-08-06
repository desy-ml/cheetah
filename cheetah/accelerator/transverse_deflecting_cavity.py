from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from scipy.constants import c as c_light
from scipy.constants import physical_constants
from torch import Size, nn

from cheetah.particles import Beam, ParticleBeam
from cheetah.utils import UniqueNameGenerator, bmadx

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class TransverseDeflectingCavity(Element):
    """
    Transverse Deflecting Cavity Element.

    :param length: Length in meters.
    :param voltage: Voltage of the cavity in volts.
    :param phase: Phase of the cavity in radians.
    :param frequency: Frequency of the cavity in Hz.
    :param misalignment: Misalignment vector of the quadrupole in x- and y-directions.
    :param tilt: Tilt angle of the quadrupole in x-y plane [rad]. pi/4 for
        skew-quadrupole.
    :param num_steps: Number of drift-kick-drift steps to use for tracking through the
        element when tracking method is set to `"bmadx"`.
    :param tracking_method: Method to use for tracking through the element.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[torch.Tensor, nn.Parameter],
        voltage: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        phase: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        frequency: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        misalignment: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        tilt: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        num_steps: int = 1,
        tracking_method: Literal["bmadx"] = "bmadx",
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.register_buffer("length", torch.as_tensor(length, **factory_kwargs))
        self.register_buffer(
            "voltage",
            (
                torch.as_tensor(voltage, **factory_kwargs)
                if voltage is not None
                else torch.tensor(0.0, **factory_kwargs)
            ),
        )
        self.register_buffer(
            "phase",
            (
                torch.as_tensor(phase, **factory_kwargs)
                if phase is not None
                else torch.tensor(0.0, **factory_kwargs)
            ),
        )
        self.register_buffer(
            "frequency",
            (
                torch.as_tensor(frequency, **factory_kwargs)
                if frequency is not None
                else torch.tensor(0.0, **factory_kwargs)
            ),
        )
        self.register_buffer(
            "misalignment",
            (
                torch.as_tensor(misalignment, **factory_kwargs)
                if misalignment is not None
                else torch.zeros((*self.length.shape, 2), **factory_kwargs)
            ),
        )
        self.register_buffer(
            "tilt",
            (
                torch.as_tensor(tilt, **factory_kwargs)
                if tilt is not None
                else torch.zeros_like(self.length)
            ),
        )
        self.num_steps = num_steps
        self.tracking_method = tracking_method

    @property
    def is_active(self) -> bool:
        return torch.any(self.voltage != 0)

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the crab cavity.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        if self.tracking_method == "cheetah":
            raise NotImplementedError(
                "cheetah crab cavity tracking is not yet implemented"
            )
        elif self.tracking_method == "bmadx":
            assert isinstance(
                incoming, ParticleBeam
            ), "Bmad-X tracking is currently only supported for `ParticleBeam`."
            return self._track_bmadx(incoming)
        else:
            raise ValueError(
                f"Invalid tracking method {self.tracking_method}. "
                + "Supported methods are 'cheetah' and 'bmadx'."
            )

    def _track_bmadx(self, incoming: ParticleBeam) -> ParticleBeam:
        """
        Track particles through the TDC element
        using the Bmad-X tracking method.

        :param incoming: Beam entering the element. Currently only supports
            `ParticleBeam`.
        :return: Beam exiting the element.
        """
        # Compute Bmad coordinates and p0c
        mc2 = electron_mass_eV.to(
            device=incoming.particles.device, dtype=incoming.particles.dtype
        )

        x = incoming.particles[..., 0]
        px = incoming.particles[..., 1]
        y = incoming.particles[..., 2]
        py = incoming.particles[..., 3]
        tau = incoming.particles[..., 4]
        delta = incoming.particles[..., 5]

        z, pz, p0c = bmadx.cheetah_to_bmad_z_pz(tau, delta, incoming.energy, mc2)

        x_offset = self.misalignment[..., 0]
        y_offset = self.misalignment[..., 1]

        # Begin Bmad-X tracking
        x, px, y, py = bmadx.offset_particle_set(
            x_offset, y_offset, self.tilt, x, px, y, py
        )

        x, y, z = bmadx.track_a_drift(self.length / 2, x, px, y, py, z, pz, p0c, mc2)

        voltage = self.voltage / p0c
        k_rf = 2 * torch.pi * self.frequency / c_light
        phase = (
            2
            * torch.pi
            * (self.phase - (bmadx.particle_rf_time(z, pz, p0c, mc2) * self.frequency))
        )

        px = px + voltage * torch.sin(phase)

        beta = (1 + pz) * p0c / torch.sqrt(((1 + pz) * p0c) ** 2 + mc2**2)
        beta_old = beta
        E_old = (1 + pz) * p0c / beta_old
        E_new = E_old + voltage * torch.cos(phase) * k_rf * x * p0c
        pc = torch.sqrt(E_new**2 - mc2**2)
        beta = pc / E_new

        pz = (pc - p0c) / p0c
        z = z * beta / beta_old

        x, y, z = bmadx.track_a_drift(self.length / 2, x, px, y, py, z, pz, p0c, mc2)

        x, px, y, py = bmadx.offset_particle_unset(
            x_offset, y_offset, self.tilt, x, px, y, py
        )
        # End of Bmad-X tracking

        # Convert back to Cheetah coordinates
        tau, delta, ref_energy = bmadx.bmad_to_cheetah_z_pz(z, pz, p0c, mc2)

        outgoing_beam = ParticleBeam(
            torch.stack((x, px, y, py, tau, delta, torch.ones_like(x)), dim=-1),
            ref_energy,
            particle_charges=incoming.particle_charges,
            device=incoming.particles.device,
            dtype=incoming.particles.dtype,
        )
        return outgoing_beam

    def broadcast(self, shape: Size) -> Element:
        return self.__class__(
            length=self.length.repeat(shape),
            voltage=self.voltage.repeat(shape),
            phase=self.phase.repeat(shape),
            frequency=self.frequency.repeat(shape),
            misalignment=self.misalignment.repeat((*shape, 1)),
            tilt=self.tilt.repeat(shape),
            name=self.name,
            device=self.length.device,
            dtype=self.length.dtype,
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
        return super().defining_features + [
            "length", 
            "voltage", 
            "phase", 
            "frequency", 
            "misalignment", 
            "tilt"
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"voltage={repr(self.voltage)}, "
            + f"phase={repr(self.phase)}, "
            + f"frequency={repr(self.frequency)}, "
            + f"num_steps={repr(self.num_steps)}, "
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )
