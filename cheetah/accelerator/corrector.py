from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.constants import physical_constants
from torch import Size, nn

from cheetah.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class Corrector(Element):
    """
    Corrector magnet in a particle accelerator.
    Note: This is modeled as a drift section with
        a thin-kick in the horizontal plane followed by
        a thin-kick in the vertical plane.

    :param length: Length in meters.
    :param horizontal_angle: Particle deflection horizontal_angle in
        the horizontal plane in rad.
    :param vertical_angle: Particle deflection vertical_angle in
        the vertical plane in rad.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[torch.Tensor, nn.Parameter],
        horizontal_angle: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        vertical_angle: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.length = torch.as_tensor(length, **factory_kwargs)
        self.horizontal_angle = (
            torch.as_tensor(horizontal_angle, **factory_kwargs)
            if horizontal_angle is not None
            else torch.zeros_like(self.length)
        )
        self.vertical_angle = (
            torch.as_tensor(vertical_angle, **factory_kwargs)
            if vertical_angle is not None
            else torch.zeros_like(self.length)
        )

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        device = self.length.device
        dtype = self.length.dtype

        gamma = energy / electron_mass_eV.to(device=device, dtype=dtype)
        igamma2 = torch.zeros_like(gamma)  # TODO: Effect on gradients?
        igamma2[gamma != 0] = 1 / gamma[gamma != 0] ** 2
        beta = torch.sqrt(1 - igamma2)

        tm = torch.eye(7, device=device, dtype=dtype).repeat((*self.length.shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 2, 3] = self.length
        tm[..., 1, 6] = self.horizontal_angle
        tm[..., 3, 6] = self.vertical_angle
        tm[..., 4, 5] = -self.length / beta**2 * igamma2

        return tm

    def broadcast(self, shape: Size) -> Element:
        return self.__class__(
            length=self.length.repeat(shape),
            horizontal_angle=self.horizontal_angle,
            vertical_angle=self.vertical_angle,
            name=self.name,
        )

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return any(self.horizontal_angle != 0, self.vertical_angle != 0)

    def split(self, resolution: torch.Tensor) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = torch.min(resolution, remaining)
            element = Corrector(
                length,
                self.horizontal_angle * length / self.length,
                self.vertical_angle * length / self.length,
            )
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = (np.sign(self.horizontal_angle[0]) if self.is_active else 1) * (
            np.sign(self.vertical_angle[0]) if self.is_active else 1
        )

        patch = Rectangle(
            (s, 0), self.length[0], height, color="tab:blue", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "horizontal_angle",
            "vertical_angle",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"horizontal_angle={repr(self.horizontal_angle)}, "
            + f"vertical_angle={repr(self.vertical_angle)}, "
            + f"name={repr(self.name)})"
        )
