import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Species
from cheetah.utils import (
    UniqueNameGenerator,
    cache_transfer_map,
    compute_relativistic_factors,
)

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class CombinedCorrector(Element):
    """
    Combined horizontal and vertical corrector magnet in a particle accelerator.

    NOTE: This is modeled as a drift section with thin-kicks in both the horizontal and
        vertical planes.

    :param length: Length in meters.
    :param horizontal_angle: Particle deflection angle in the horizontal plane in rad.
    :param vertical_angle: Particle deflection angle in the vertical plane in rad.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    """

    supported_tracking_methods = ["linear"]

    def __init__(
        self,
        length: torch.Tensor,
        horizontal_angle: torch.Tensor | None = None,
        vertical_angle: torch.Tensor | None = None,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = length

        self.register_buffer_or_parameter(
            "horizontal_angle",
            (
                horizontal_angle
                if horizontal_angle is not None
                else torch.tensor(0.0, **factory_kwargs)
            ),
        )
        self.register_buffer_or_parameter(
            "vertical_angle",
            (
                vertical_angle
                if vertical_angle is not None
                else torch.tensor(0.0, **factory_kwargs)
            ),
        )

    @cache_transfer_map
    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        factory_kwargs = {"device": self.length.device, "dtype": self.length.dtype}

        _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

        vector_shape = torch.broadcast_shapes(
            self.length.shape,
            igamma2.shape,
            self.horizontal_angle.shape,
            self.vertical_angle.shape,
        )

        tm = torch.eye(7, **factory_kwargs).repeat((*vector_shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 1, 6] = self.horizontal_angle
        tm[..., 2, 3] = self.length
        tm[..., 3, 6] = self.vertical_angle
        tm[..., 4, 5] = -self.length / beta**2 * igamma2

        return tm

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return (
            torch.any(self.horizontal_angle != 0).item()
            or torch.any(self.vertical_angle != 0).item()
        )

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2

        patch = Rectangle(
            (plot_s, -0.6),
            plot_length,
            0.6 * 2,
            color="chocolate",
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "horizontal_angle",
            "vertical_angle",
        ]
