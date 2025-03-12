from typing import Literal

import matplotlib.pyplot as plt
import torch

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParticleBeam, Species
from cheetah.utils import UniqueNameGenerator, bmadx, compute_relativistic_factors

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Drift(Element):
    """
    Drift section in a particle accelerator.

    NOTE: The transfer map now uses the linear approximation.
    Including the R_56 = L / (beta**2 * gamma **2)

    :param length: Length in meters.
    :param tracking_method: Method to use for tracking through the element.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: torch.Tensor,
        tracking_method: Literal["cheetah", "bmadx"] = "cheetah",
        name: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        self.tracking_method = tracking_method

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        device = self.length.device
        dtype = self.length.dtype

        _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

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
            tau, delta, incoming.energy, incoming.species.mass_eV
        )

        # Begin Bmad-X tracking
        x, y, z = bmadx.track_a_drift(
            self.length, x, px, y, py, z, pz, p0c, incoming.species.mass_eV
        )
        # End of Bmad-X tracking

        # Convert back to Cheetah coordinates
        tau, delta, ref_energy = bmadx.bmad_to_cheetah_z_pz(
            z, pz, p0c, incoming.species.mass_eV
        )

        # Broadcast to align their shapes so that they can be stacked
        x, px, y, py, tau, delta = torch.broadcast_tensors(x, px, y, py, tau, delta)

        outgoing_beam = ParticleBeam(
            particles=torch.stack(
                [x, px, y, py, tau, delta, torch.ones_like(x)], dim=-1
            ),
            energy=ref_energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            species=incoming.species,
        )
        return outgoing_beam

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

    def plot(self, ax: plt.Axes, s: float, vector_idx: tuple | None = None) -> None:
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
