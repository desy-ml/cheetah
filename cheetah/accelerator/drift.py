from typing import Optional, Union

import matplotlib.pyplot as plt
import torch
from scipy.constants import physical_constants
from torch import nn

from ..utils import UniqueNameGenerator
from ..utils.physics import calculate_relativistic_factors
from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class Drift(Element):
    """
    Drift section in a particle accelerator.

    Note: the transfer map now uses the linear approximation.
    Including the R_56 = L / (beta**2 * gamma **2)

    :param length: Length in meters.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[torch.Tensor, nn.Parameter],
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.register_buffer("length", torch.as_tensor(length, **factory_kwargs))

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        device = self.length.device
        dtype = self.length.dtype

        _, igamma2, beta = calculate_relativistic_factors(energy)

        tm = torch.eye(7, device=device, dtype=dtype).repeat(
            (*(self.length * igamma2).shape, 1, 1)
        )
        tm[..., 0, 1] = self.length
        tm[..., 2, 3] = self.length
        tm[..., 4, 5] = -self.length / beta**2 * igamma2

        return tm

    @property
    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: torch.Tensor) -> list[Element]:
        num_splits = torch.ceil(torch.max(self.length) / resolution).int()
        return [
            Drift(
                self.length / num_splits,
                dtype=self.length.dtype,
                device=self.length.device,
            )
            for i in range(num_splits)
        ]

    def plot(self, ax: plt.Axes, s: float) -> None:
        pass

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"name={repr(self.name)})"
        )
