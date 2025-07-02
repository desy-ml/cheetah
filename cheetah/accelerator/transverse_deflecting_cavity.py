from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from scipy.constants import speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParticleBeam
from cheetah.utils import UniqueNameGenerator, bmadx, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class TransverseDeflectingCavity(Element):
    """
    Transverse deflecting cavity element.

    :param length: Length in meters.
    :param voltage: Voltage of the cavity in volts. NOTE: This assumes the physical
        voltage. The sign is default for electron-like particles.
        For particles with a positive charge, the sign should be flipped.
    :param phase: Phase of the cavity in (radians / 2 pi).
    :param frequency: Frequency of the cavity in Hz.
    :param misalignment: Misalignment vector of the quadrupole in x- and y-directions.
    :param tilt: Tilt angle of the quadrupole in x-y plane [rad]. pi/4 for
        skew-quadrupole.
    :param num_steps: Number of drift-kick-drift steps to use for tracking through the
        element when tracking method is set to `"bmadx"`.
    :param tracking_method: Method to use for tracking through the element.
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
        misalignment: torch.Tensor | None = None,
        tilt: torch.Tensor | None = None,
        num_steps: int = 1,
        tracking_method: Literal["bmadx"] = "bmadx",
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [length, voltage, phase, frequency, misalignment, tilt], device, dtype
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
        self.register_buffer_or_parameter(
            "misalignment",
            torch.as_tensor(
                misalignment if misalignment is not None else (0.0, 0.0),
                **factory_kwargs,
            ),
        )
        self.register_buffer_or_parameter(
            "tilt", torch.as_tensor(tilt if tilt is not None else 0.0, **factory_kwargs)
        )

        self.num_steps = num_steps
        self.tracking_method = tracking_method

    @property
    def is_active(self) -> bool:
        return torch.any(self.voltage != 0).item()

    @property
    def is_skippable(self) -> bool:
        # TODO: Implement drrift-like `transfer_map` and set to `self.is_active`
        return False

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the transverse deflecting cavity.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        if self.tracking_method == "cheetah":
            raise NotImplementedError(
                "Cheetah transverse deflecting cavity tracking is not yet implemented."
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
        Track particles through the TDC element using the Bmad-X tracking method.

        :param incoming: Beam entering the element. Currently only supports
            `ParticleBeam`.
        :return: Beam exiting the element.
        """
        # Compute Bmad coordinates and p0c
        x = incoming.x
        px = incoming.px
        y = incoming.y
        py = incoming.py
        tau = incoming.tau
        delta = incoming.p
        mc2 = incoming.species.mass_eV

        z, pz, p0c = bmadx.cheetah_to_bmad_z_pz(tau, delta, incoming.energy, mc2)

        x_offset = self.misalignment[..., 0]
        y_offset = self.misalignment[..., 1]

        # Begin Bmad-X tracking
        x, px, y, py = bmadx.offset_particle_set(
            x_offset, y_offset, self.tilt, x, px, y, py
        )

        x, y, z = bmadx.track_a_drift(self.length / 2, x, px, y, py, z, pz, p0c, mc2)

        voltage = self.voltage * -1 * incoming.species.num_elementary_charges / p0c
        k_rf = 2 * torch.pi * self.frequency / speed_of_light
        # Phase that the particle sees
        phase = (
            2
            * torch.pi
            * (
                self.phase.unsqueeze(-1)
                - (
                    bmadx.particle_rf_time(z, pz, p0c, mc2)
                    * self.frequency.unsqueeze(-1)
                )
            )
        )

        # TODO: Assigning px to px is really bad practice and should be separated into
        # two separate variables
        px = px + voltage.unsqueeze(-1) * torch.sin(phase)

        beta_old = (
            (1 + pz)
            * p0c.unsqueeze(-1)
            / torch.sqrt(((1 + pz) * p0c.unsqueeze(-1)) ** 2 + mc2**2)
        )
        E_old = (1 + pz) * p0c.unsqueeze(-1) / beta_old
        E_new = E_old + voltage.unsqueeze(-1) * torch.cos(phase) * k_rf.unsqueeze(
            -1
        ) * x * p0c.unsqueeze(-1)
        pc = torch.sqrt(E_new**2 - mc2**2)
        beta = pc / E_new

        pz = (pc - p0c.unsqueeze(-1)) / p0c.unsqueeze(-1)
        z = z * beta / beta_old

        x, y, z = bmadx.track_a_drift(self.length / 2, x, px, y, py, z, pz, p0c, mc2)

        x, px, y, py = bmadx.offset_particle_unset(
            x_offset, y_offset, self.tilt, x, px, y, py
        )
        # End of Bmad-X tracking

        # Convert back to Cheetah coordinates
        tau, delta, ref_energy = bmadx.bmad_to_cheetah_z_pz(z, pz, p0c, mc2)

        outgoing_beam = ParticleBeam(
            particles=torch.stack(
                (x, px, y, py, tau, delta, torch.ones_like(x)), dim=-1
            ),
            energy=ref_energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            s=incoming.s + self.length,
            species=incoming.species,
        )
        return outgoing_beam

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="olive", alpha=alpha, zorder=2
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
            "tilt",
            "num_steps",
            "tracking_method",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"voltage={repr(self.voltage)}, "
            + f"phase={repr(self.phase)}, "
            + f"frequency={repr(self.frequency)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"tilt={repr(self.tilt)}, "
            + f"num_steps={repr(self.num_steps)}, "
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )
