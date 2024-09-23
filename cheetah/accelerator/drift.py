from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import torch
from scipy.constants import physical_constants
from torch import Size, nn

from ..particles import Beam, ParticleBeam
from ..utils import UniqueNameGenerator, bmadx
from ..utils.physics import compute_relativistic_factors
from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Drift(Element):
    """
    Drift section in a particle accelerator.

    Note: the transfer map now uses the linear approximation.
    Including the R_56 = L / (beta**2 * gamma **2)

    :param length: Length in meters.
    :param tracking_method: Method to use for tracking through the element.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[torch.Tensor, nn.Parameter],
        tracking_method: Literal["cheetah", "bmadx"] = "cheetah",
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.register_buffer("length", torch.as_tensor(length, **factory_kwargs))
        self.tracking_method = tracking_method

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        device = self.length.device
        dtype = self.length.dtype

        _, igamma2, beta = compute_relativistic_factors(energy)

        vector_shape = torch.broadcast_shapes(self.length.shape, igamma2.shape)

        tm = torch.eye(7, device=device, dtype=dtype).repeat((*vector_shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 2, 3] = self.length
        tm[..., 4, 5] = -self.length / beta**2 * igamma2

        return tm

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the dipole element.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        if self.tracking_method == "cheetah":
            return super().track(incoming)
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
        Track particles through the dipole element using the Bmad-X tracking method.

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

        z, pz, p0c = bmadx.cheetah_to_bmad_z_pz(
            tau, delta, incoming.energy, electron_mass_eV
        )

        # Begin Bmad-X tracking
        x, y, z = bmadx.track_a_drift(
            self.length, x, px, y, py, z, pz, p0c, electron_mass_eV
        )
        # End of Bmad-X tracking

        # Convert back to Cheetah coordinates
        tau, delta, ref_energy = bmadx.bmad_to_cheetah_z_pz(
            z, pz, p0c, electron_mass_eV
        )

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
            tracking_method=self.tracking_method,
            name=self.name,
            device=self.length.device,
            dtype=self.length.dtype,
        )

    @property
    def is_skippable(self) -> bool:
        return self.tracking_method == "cheetah"

    def split(self, resolution: torch.Tensor) -> list[Element]:
        num_splits = torch.ceil(torch.max(self.length) / resolution).int()
        return [
            Drift(
                self.length / num_splits,
                tracking_method=self.tracking_method,
                dtype=self.length.dtype,
                device=self.length.device,
            )
            for i in range(num_splits)
        ]

    def plot(self, ax: plt.Axes, s: float) -> None:
        pass

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "tracking_method"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )
