from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch import nn

from cheetah.utils import UniqueNameGenerator

from ..utils.batching import get_batch_shape
from ..utils.physics import calculate_relativistic_factors
from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class HorizontalCorrector(Element):
    """
    Horizontal corrector magnet in a particle accelerator.
    Note: This is modeled as a drift section with
        a thin-kick in the horizontal plane.

    :param length: Length in meters.
    :param angle: Particle deflection angle in the horizontal plane in rad.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[torch.Tensor, nn.Parameter],
        angle: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.register_buffer("length", torch.as_tensor(length, **factory_kwargs))
        self.register_buffer(
            "angle",
            (
                torch.as_tensor(angle, **factory_kwargs)
                if angle is not None
                else torch.zeros_like(self.length)
            ),
        )

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        device = self.length.device
        dtype = self.length.dtype

        _, igamma2, beta = calculate_relativistic_factors(energy)

        batch_shape = get_batch_shape(self.length, self.angle, beta)
        tm = torch.eye(7, device=device, dtype=dtype).repeat((*batch_shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 1, 6] = self.angle
        tm[..., 2, 3] = self.length
        tm[..., 4, 5] = -self.length / beta**2 * igamma2

        return tm

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return any(self.angle != 0)

    def split(self, resolution: torch.Tensor) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = torch.min(resolution, remaining)
            element = HorizontalCorrector(
                length,
                self.angle * length / self.length,
                dtype=self.length.dtype,
                device=self.length.device,
            )
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.angle[0]) if self.is_active else 1)

        patch = Rectangle(
            (s, 0), self.length[0], height, color="tab:blue", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "angle"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"angle={repr(self.angle)}, "
            + f"name={repr(self.name)})"
        )
