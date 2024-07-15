from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch import nn

from ..track_methods import base_rmatrix, misalignment_matrix
from ..utils import UniqueNameGenerator
from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Quadrupole(Element):
    """
    Quadrupole magnet in a particle accelerator.

    :param length: Length in meters.
    :param k1: Strength of the quadrupole in 1/m^-2.
    :param misalignment: Misalignment vector of the quadrupole in x- and y-directions.
    :param tilt: Tilt angle of the quadrupole in x-y plane [rad]. pi/4 for
        skew-quadrupole.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[torch.Tensor, nn.Parameter],
        k1: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        misalignment: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        tilt: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.register_buffer("length", torch.as_tensor(length, **factory_kwargs))
        self.register_buffer(
            "k1",
            (
                torch.as_tensor(k1, **factory_kwargs)
                if k1 is not None
                else torch.zeros_like(self.length)
            ),
        )
        self.register_buffer(
            "misalignment",
            (
                torch.as_tensor(misalignment, **factory_kwargs)
                if misalignment is not None
                else torch.zeros((*self.length.shape, 2), **factory_kwargs)
            ),
        )
        self.register_buffer(
            "tilt",
            (
                torch.as_tensor(tilt, **factory_kwargs)
                if tilt is not None
                else torch.zeros_like(self.length)
            ),
        )

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        R = base_rmatrix(
            length=self.length,
            k1=self.k1,
            hx=torch.zeros_like(self.length),
            tilt=self.tilt,
            energy=energy,
        )

        if torch.all(self.misalignment == 0):
            return R
        else:
            R_entry, R_exit = misalignment_matrix(self.misalignment)
            R = torch.einsum("...ij,...jk,...kl->...il", R_exit, R, R_entry)
            return R

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return any(self.k1 != 0)

    def split(self, resolution: torch.Tensor) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Quadrupole(
                torch.min(resolution, remaining),
                self.k1,
                misalignment=self.misalignment,
                tilt=self.tilt,
                dtype=self.length.dtype,
                device=self.length.device,
            )
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.k1[0]) if self.is_active else 1)
        patch = Rectangle(
            (s, 0), self.length[0], height, color="tab:red", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "k1", "misalignment", "tilt"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"k1={repr(self.k1)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"tilt={repr(self.tilt)}, "
            + f"name={repr(self.name)})"
        )
