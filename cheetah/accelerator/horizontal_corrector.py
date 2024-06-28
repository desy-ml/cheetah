from typing import Optional, Union

import torch
from torch import nn

from cheetah.utils import UniqueNameGenerator

from .corrector import Corrector

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
        horizontal_angle: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        # vertical_angle: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__(
            length=length,
            horizontal_angle=horizontal_angle,
            name=name,
            device=device,
            dtype=dtype,
        )
