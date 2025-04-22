import matplotlib.pyplot as plt
import torch

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.utils import compute_relativistic_factors, verify_device_and_dtype


class Sextupole(Element):
    """
    A sextupole element in a particle accelerator.

    :param length: Length in meters.
    :param k2: TODO
    :param misalignment: TODO
    :param tilt: TODO
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
        device, dtype = verify_device_and_dtype([length, k2], device, dtype)
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
        Track the beam through the sextupole element.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        first_order_tm = self.transfer_map(incoming.energy, incoming.species)

        second_order_tm = torch.eye(
            7, device=self.length.device, dtype=self.length.dtype
        ).repeat((*incoming.mu_x.shape, 1, 1))
        second_order_tm[..., 0, 1] = self.length
        second_order_tm[..., 2, 3] = self.length
        second_order_tm[..., 4, 5] = (
            -self.length
            / incoming.relativistic_beta**2
            * incoming.relativistic_gamma**2
        )
        second_order_tm[..., 1, 0] = self.k2 * self.length**3 / 6
        second_order_tm[..., 3, 2] = self.k2 * self.length**3 / 6
        second_order_tm[..., 5, 4] = -self.k2 * self.length**3 / 6
        second_order_tm[..., 1, 2] = self.k2 * self.length**3 / 6
        second_order_tm[..., 3, 4] = self.k2 * self.length**3 / 6
        second_order_tm[..., 5, 0] = -self.k2 * self.length**3 / 6
        second_order_tm[..., 5, 2] = -self.k2 * self.length**3 / 6
        second_order_tm[..., 1, 4] = -self.k2 * self.length**3 / 6
        second_order_tm[..., 3, 0] = -self.k2 * self.length**3 / 6
        second_order_tm[..., 5, 3] = -self.k2 * self.length**3 / 6
        second_order_tm[..., 1, 5] = -self.k2 * self.length**3 / 6
        second_order_tm[..., 3, 2] = -self.k2 * self.length**3 / 6
        second_order_tm[..., 5, 4] = -self.k2 * self.length**3 / 6

        # Apply the transfer map to the incoming beam
        particles = torch.einsum("ijk,ikl->ijl", first_order_tm, incoming.particles)
        particles = torch.einsum("ijk,ikl->ijl", second_order_tm, particles)
        return ParticleBeam(
            particles=particles,
            energy=incoming.energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            species=incoming.species,
        )

    @property
    def is_skippable(self) -> bool:
        return False

    @property
    def is_active(self) -> bool:
        return torch.any(self.k2 != 0.0)

    def split(self, resolution: torch.Tensor) -> list[Element]:
        raise NotImplementedError

    def plot(self, ax: plt.Axes, s: float, vector_idx: tuple | None = None) -> None:
        raise NotImplementedError

    def defining_features(self) -> list[str]:
        return super().defining_features() + ["length", "k2"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            f"k2={repr(self.k2)}, "
            f"name={repr(self.name)})"
        )
