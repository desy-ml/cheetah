import matplotlib.pyplot as plt
import torch

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, Species
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Marker(Element):
    """
    General Marker / Monitor element

    :param name: Unique identifier of the element.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        return torch.eye(7, device=energy.device, dtype=energy.dtype).repeat(
            (*energy.shape, 1, 1)
        )

    def track(self, incoming: Beam) -> Beam:
        # TODO: At some point Markers should be able to be active or inactive. Active
        # Markers would be able to record the beam tracked through them.
        return incoming

    @property
    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: torch.Tensor) -> list[Element]:
        return [self]

    def plot(self, ax: plt.Axes, s: float, vector_idx: tuple | None = None) -> None:
        # Do nothing on purpose. Maybe later we decide markers should be shown, but for
        # now they are invisible.
        pass

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={repr(self.name)})"
