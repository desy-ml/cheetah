from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParticleBeam, Species
from cheetah.utils import UniqueNameGenerator, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Aperture(Element):
    """
    Physical aperture.

    NOTE: The aperture currently only affects beams of type `ParticleBeam` and only has
        an effect when the aperture is active.

    :param x_max: half size horizontal offset in [m].
    :param y_max: half size vertical offset in [m].
    :param shape: Shape of the aperture. Can be "rectangular" or "elliptical".
    :param is_active: If the aperture actually blocks particles.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python
        variable name. This is needed if you want to use the `segment.element_name`
        syntax to access the element in a segment.
    """

    def __init__(
        self,
        x_max: torch.Tensor | None = None,
        y_max: torch.Tensor | None = None,
        shape: Literal["rectangular", "elliptical"] = "rectangular",
        is_active: bool = True,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype([x_max, y_max], device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.register_buffer_or_parameter(
            "x_max",
            torch.as_tensor(
                x_max if x_max is not None else float("inf"), **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "y_max",
            torch.as_tensor(
                y_max if y_max is not None else float("inf"), **factory_kwargs
            ),
        )

        self.shape = shape
        self.is_active = is_active

        self.lost_particles = None

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        device = self.x_max.device
        dtype = self.x_max.dtype

        return torch.eye(7, device=device, dtype=dtype).repeat((*energy.shape, 1, 1))

    def track(self, incoming: Beam) -> Beam:
        # Only apply aperture to particle beams and if the element is active
        if not (isinstance(incoming, ParticleBeam) and self.is_active):
            return incoming

        assert torch.all(self.x_max >= 0) and torch.all(self.y_max >= 0)
        assert self.shape in [
            "rectangular",
            "elliptical",
        ], f"Unknown aperture shape {self.shape}"

        if self.shape == "rectangular":
            survived_mask = torch.logical_and(
                torch.logical_and(
                    incoming.x > -self.x_max.unsqueeze(-1),
                    incoming.x < self.x_max.unsqueeze(-1),
                ),
                torch.logical_and(
                    incoming.y > -self.y_max.unsqueeze(-1),
                    incoming.y < self.y_max.unsqueeze(-1),
                ),
            )
        elif self.shape == "elliptical":
            survived_mask = (
                incoming.x**2 / self.x_max.unsqueeze(-1) ** 2
                + incoming.y**2 / self.y_max.unsqueeze(-1) ** 2
            ) <= 1.0

        return ParticleBeam(
            particles=incoming.particles,
            energy=incoming.energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities * survived_mask,
            s=incoming.s,
            species=incoming.species.clone(),
        )

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s

        alpha = 1 if self.is_active else 0.2
        height = 0.4

        dummy_length = 0.0

        patch = Rectangle(
            (plot_s, 0), dummy_length, height, color="tab:pink", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["x_max", "y_max", "shape", "is_active"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(x_max={repr(self.x_max)}, "
            + f"y_max={repr(self.y_max)}, "
            + f"shape={repr(self.shape)}, "
            + f"is_active={repr(self.is_active)}, "
            + f"name={repr(self.name)})"
        )
