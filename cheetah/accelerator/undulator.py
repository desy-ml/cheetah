from typing import Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from scipy.constants import physical_constants

from cheetah.accelerator.element import Element
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Undulator(Element):
    """
    Element representing an undulator in a particle accelerator.

    NOTE Currently behaves like a drift section but is plotted distinctively.

    :param length: Length in meters.
    :param is_active: Indicates if the undulator is active or not. Currently has no
        effect.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: torch.Tensor,
        is_active: bool = False,
        name: Optional[str] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.length = torch.as_tensor(length, **factory_kwargs)
        self.is_active = is_active

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        device = self.length.device
        dtype = self.length.dtype

        gamma = energy / electron_mass_eV
        igamma2 = torch.where(gamma != 0, 1 / gamma**2, torch.zeros_like(gamma))

        vector_shape = torch.broadcast_shapes(self.length.shape, igamma2.shape)

        tm = torch.eye(7, device=device, dtype=dtype).repeat((*vector_shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 2, 3] = self.length
        tm[..., 4, 5] = self.length * igamma2

        return tm

    @property
    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: torch.Tensor) -> list[Element]:
        # TODO: Implement splitting for undulator properly, for now just return self
        return [self]

    def plot(self, ax: plt.Axes, s: float, vector_idx: Optional[tuple] = None) -> None:
        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:purple", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"is_active={repr(self.is_active)}, "
            + f"name={repr(self.name)})"
        )
