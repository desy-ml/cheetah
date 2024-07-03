from typing import Optional, Union

import torch
from torch import Size, nn

from cheetah.utils import UniqueNameGenerator

from .corrector import Corrector
from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class VerticalCorrector(Corrector):
    """
    Vertical corrector magnet in a particle accelerator.
    Note: This is modeled as a drift section with
        a thin-kick in the vertical plane.

    :param length: Length in meters.
    :param vertical_angle: Particle deflection angle in the vertical plane in rad.
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
            vertical_angle=angle,
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
        return self.vertical_angle

    @angle.setter
    def angle(self, value: torch.Tensor) -> None:
        self.vertical_angle = value

    @property
    def defining_features(self) -> list[str]:
        features_with_both_angles = super().defining_features
        cleaned_features = [
            feature
            for feature in features_with_both_angles
            if feature not in ["horizontal_angle", "vertical_angle"]
        ]
        return cleaned_features + ["length", "angle"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"angle={repr(self.angle)}, "
            + f"name={repr(self.name)})"
        )
