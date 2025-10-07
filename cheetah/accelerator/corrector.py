from typing import Optional, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch import nn

from cheetah.particles.species import Species
from cheetah.utils import UniqueNameGenerator
from cheetah.utils.cache import cache_transfer_map
from cheetah.utils.physics import compute_relativistic_factors

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Corrector(Element):
    """
    Combined corrector magnet in a particle accelerator.

    Note: This is modeled as a drift section with a thin-kick in the horizontal plane
          followed by a thin-kick in the vertical plane.

    :param length: Length in meters.
    :param horizontal_angle: Particle deflection horizontal_angle in the horizontal
        plane in rad.
    :param vertical_angle: Particle deflection vertical_angle in the vertical plane in
        rad.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[torch.Tensor, nn.Parameter],
        horizontal_angle: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        vertical_angle: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        name: Optional[str] = None,
        sanitize_name: bool = False,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.register_buffer("length", torch.as_tensor(length, **factory_kwargs))
        self.register_buffer(
            "horizontal_angle",
            (
                torch.as_tensor(horizontal_angle, **factory_kwargs)
                if horizontal_angle is not None
                else torch.zeros_like(self.length)
            ),
        )
        self.register_buffer(
            "vertical_angle",
            (
                torch.as_tensor(vertical_angle, **factory_kwargs)
                if vertical_angle is not None
                else torch.zeros_like(self.length)
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
        tm[..., 2, 3] = self.length
        tm[..., 1, 6] = self.horizontal_angle
        tm[..., 3, 6] = self.vertical_angle
        tm[..., 4, 5] = -self.length / beta**2 * igamma2

        return tm

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return self.horizontal_angle != 0 or self.vertical_angle != 0

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length
        plot_angle_h = (
            self.horizontal_angle[vector_idx]
            if self.horizontal_angle.dim() > 0
            else self.horizontal_angle
        )
        plot_angle_v = (
            self.vertical_angle[vector_idx]
            if self.vertical_angle.dim() > 0
            else self.vertical_angle
        )

        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (torch.sign(plot_angle_h) if self.is_active else 1)

        patch = Rectangle(
            (plot_s, 0),
            plot_length,
            height,
            angle=plot_angle_v,
            color="tab:blue",
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
