from typing import Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class BAM(Element):
    """
    Beam Position Monitor (BPM) in a particle accelerator.

    :param is_active: If `True` the BPM is active and will record the beam's position.
        If `False` the BPM is inactive and will not record the beam's position.
    :param name: Unique identifier of the element.
    """

    def __init__(self, is_active: bool = False, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.is_active = is_active
        self.reading = None

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        return torch.eye(7, device=energy.device, dtype=energy.dtype).repeat(
            (*energy.shape, 1, 1)
        )

    def track(self, incoming: Beam) -> Beam:
        if isinstance(incoming, ParameterBeam):
            self.reading = torch.stack([incoming.mu_x, incoming.mu_y])
        elif isinstance(incoming, ParticleBeam):
            self.reading = torch.stack([incoming.mu_x, incoming.mu_y])
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

        return incoming.clone()

    def split(self, resolution: torch.Tensor) -> list[Element]:
        return [self]

    def plot(self, ax: plt.Axes, s: float, vector_idx: Optional[tuple] = None) -> None:
        plot_s = s[vector_idx] if s.dim() > 0 else s

        alpha = 1 if self.is_active else 0.2
        patch = Rectangle(
            (plot_s, -0.3), 0, 0.3 * 2, color="darkkhaki", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["is_active"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={repr(self.name)})"
