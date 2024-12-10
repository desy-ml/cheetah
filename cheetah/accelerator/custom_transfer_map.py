from typing import Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Beam
from cheetah.utils import UniqueNameGenerator, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class CustomTransferMap(Element):
    """
    This element can represent any custom transfer map.
    """

    def __init__(
        self,
        predefined_transfer_map: torch.Tensor,
        length: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        device=None,
        dtype=None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [predefined_transfer_map, length], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        assert isinstance(predefined_transfer_map, torch.Tensor)
        assert predefined_transfer_map.shape[-2:] == (7, 7)

        self.register_buffer("predefined_transfer_map", None)

        self.predefined_transfer_map = torch.as_tensor(
            predefined_transfer_map, **factory_kwargs
        )
        if length is not None:
            self.length = torch.as_tensor(length, **factory_kwargs)

    @classmethod
    def from_merging_elements(
        cls, elements: list[Element], incoming_beam: Beam
    ) -> "CustomTransferMap":
        """
        Combine the transfer maps of multiple successive elements into a single transfer
        map. This can be used to speed up tracking through a segment, if no changes
        are made to the elements in the segment or the energy of the beam being tracked
        through them.

        :param elements: List of consecutive elements to combine.
        :param incoming_beam: Beam entering the first element in the segment. NOTE: That
            this is required because the separate original transfer maps have to be
            computed before being combined and some of them may depend on the energy of
            the beam.
        """
        assert all(element.is_skippable for element in elements), (
            "Combining the elements in a Segment that is not skippable will result in"
            " incorrect tracking results."
        )

        device = elements[0].transfer_map(incoming_beam.energy).device
        dtype = elements[0].transfer_map(incoming_beam.energy).dtype

        tm = torch.eye(7, device=device, dtype=dtype).repeat(
            (*incoming_beam.energy.shape, 1, 1)
        )
        for element in elements:
            tm = torch.matmul(element.transfer_map(incoming_beam.energy), tm)
            incoming_beam = element.track(incoming_beam)

        combined_length = sum(element.length for element in elements)

        combined_name = "combined_" + "_".join(element.name for element in elements)

        return cls(
            tm, length=combined_length, device=device, dtype=dtype, name=combined_name
        )

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        return self.predefined_transfer_map

    @property
    def is_skippable(self) -> bool:
        return True

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + f"predefined_transfer_map={repr(self.predefined_transfer_map)}, "
            + f"length={repr(self.length)}, "
            + f"name={repr(self.name)})"
        )

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "predefined_transfer_map"]

    def split(self, resolution: torch.Tensor) -> list[Element]:
        return [self]

    def plot(self, ax: plt.Axes, s: float, vector_idx: Optional[tuple] = None) -> None:
        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        height = 0.4

        patch = Rectangle((plot_s, 0), plot_length, height, color="tab:olive", zorder=2)
        ax.add_patch(patch)
