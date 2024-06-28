from typing import Optional, Union

import torch
from torch import Size, nn

from cheetah.utils import UniqueNameGenerator

from .corrector import Corrector
from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class HorizontalCorrector(Corrector):
    """
    Horizontal corrector magnet in a particle accelerator.
    Note: This is modeled as a drift section with
        a thin-kick in the horizontal plane.

    :param length: Length in meters.
    :param horizontal_angle: Particle deflection angle in the horizontal plane in rad.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[torch.Tensor, nn.Parameter],
        angle: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__(
            length=length,
            horizontal_angle=angle,
            name=name,
            device=device,
            dtype=dtype,
        )

    def broadcast(self, shape: Size) -> Element:
        return self.__class__(
            length=self.length.repeat(shape),
            angle=self.angle,
            name=self.name,
        )

    @property
    def angle(self) -> torch.Tensor:
        return self.horizontal_angle

    @angle.setter
    def angle(self, value: torch.Tensor) -> None:
        self.horizontal_angle = value

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "angle",
        ]
