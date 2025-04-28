import matplotlib.pyplot as plt
import torch

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.track_methods import base_rmatrix, base_ttensor, misalignment_matrix
from cheetah.utils import verify_device_and_dtype


class Sextupole(Element):
    """
    A sextupole element in a particle accelerator.

    :param length: Length in meters.
    :param k2: Sextupole strength in 1/m^3.
    :param misalignment: Transverse misalignment in x and y directions in meters.
    :param tilt: Tilt angle of the quadrupole in x-y plane in radians.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: torch.Tensor,
        k2: torch.Tensor | None = None,
        misalignment: torch.Tensor | None = None,
        tilt: torch.Tensor | None = None,
        name: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [length, k2, misalignment, tilt], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        self.register_buffer_or_parameter(
            "k2", torch.as_tensor(k2 if k2 is not None else 0.0, **factory_kwargs)
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

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        R = base_rmatrix(
            length=self.length,
            k1=torch.zeros_like(self.length),
            hx=torch.zeros_like(self.length),
            species=species,
            tilt=self.tilt,
            energy=energy,
        )

        if torch.all(self.misalignment == 0):
            return R
        else:
            R_entry, R_exit = misalignment_matrix(self.misalignment)
            R = R_exit @ R @ R_entry
            return R

    def track(self, incoming: Beam) -> Beam:
        """
        Track the beam through the sextupole element.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        first_order_tm = self.transfer_map(incoming.energy, incoming.species)
        second_order_tm = base_ttensor(
            length=self.length,
            k1=torch.zeros_like(self.length),
            k2=self.k2,
            hx=torch.zeros_like(self.length),
            species=incoming.species,
            tilt=self.tilt,
            energy=incoming.energy,
        )

        if isinstance(incoming, ParameterBeam):
            # For ParameterBeam, only first-order effects are applied
            return super().track(incoming)
        elif isinstance(incoming, ParticleBeam):
            # Apply the transfer map to the incoming particles
            first_order_particles = incoming.particles @ first_order_tm.transpose(
                -2, -1
            )
            second_order_particles = torch.einsum(
                "...ijk,...j,...k->...i",
                second_order_tm.unsqueeze(-4),  # Add broadcast dimension for particles
                incoming.particles,
                incoming.particles,
            )
            outgoing_particles = second_order_particles + first_order_particles

            return ParticleBeam(
                particles=outgoing_particles,
                energy=incoming.energy,
                particle_charges=incoming.particle_charges,
                survival_probabilities=incoming.survival_probabilities,
                species=incoming.species,
            )
        else:
            raise TypeError(
                f"Unsupported beam type: {type(incoming)}. Expected ParameterBeam or "
                "ParticleBeam."
            )

    @property
    def is_skippable(self) -> bool:
        return False

    @property
    def is_active(self) -> bool:
        return torch.any(self.k2 != 0.0).item()

    def split(self, resolution: torch.Tensor) -> list[Element]:
        raise NotImplementedError

    def plot(self, ax: plt.Axes, s: float, vector_idx: tuple | None = None) -> None:
        raise NotImplementedError

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "k2", "misalignment", "tilt"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            f"k2={repr(self.k2)}, "
            f"misalignment={repr(self.misalignment)}, "
            f"tilt={repr(self.tilt)}, "
            f"name={repr(self.name)})"
        )
