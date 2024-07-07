from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch import Size, nn

from cheetah.particles import Beam, ParticleBeam
from cheetah.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Aperture(Element):
    """
    Physical aperture.

    :param x_max: half size horizontal offset in [m]
    :param y_max: half size vertical offset in [m]
    :param shape: Shape of the aperture. Can be "rectangular" or "elliptical".
    :param is_active: If the aperture actually blocks particles.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        x_max: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        y_max: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        shape: Literal["rectangular", "elliptical"] = "rectangular",
        is_active: bool = True,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.register_buffer(
            "x_max",
            (
                torch.as_tensor(x_max, **factory_kwargs)
                if x_max is not None
                else torch.tensor(float("inf"), **factory_kwargs)
            ),
        )
        self.register_buffer(
            "y_max",
            (
                torch.as_tensor(y_max, **factory_kwargs)
                if y_max is not None
                else torch.tensor(float("inf"), **factory_kwargs)
            ),
        )
        self.shape = shape
        self.is_active = is_active

        self.lost_particles = None

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        device = self.x_max.device
        dtype = self.x_max.dtype

        return torch.eye(7, device=device, dtype=dtype).repeat((*energy.shape, 1, 1))

    def track(self, incoming: Beam) -> Beam:
        # Only apply aperture to particle beams and if the element is active
        if not (isinstance(incoming, ParticleBeam) and self.is_active):
            return incoming

        assert self.x_max >= 0 and self.y_max >= 0
        assert self.shape in [
            "rectangular",
            "elliptical",
        ], f"Unknown aperture shape {self.shape}"

        if self.shape == "rectangular":
            survived_mask = torch.logical_and(
                torch.logical_and(incoming.xs > -self.x_max, incoming.xs < self.x_max),
                torch.logical_and(incoming.ys > -self.y_max, incoming.ys < self.y_max),
            )
        elif self.shape == "elliptical":
            survived_mask = (
                incoming.xs**2 / self.x_max**2 + incoming.ys**2 / self.y_max**2
            ) <= 1.0
        outgoing_particles = incoming.particles[survived_mask]

        outgoing_particle_charges = incoming.particle_charges[survived_mask]

        self.lost_particles = incoming.particles[torch.logical_not(survived_mask)]

        self.lost_particle_charges = incoming.particle_charges[
            torch.logical_not(survived_mask)
        ]

        return (
            ParticleBeam(
                outgoing_particles,
                incoming.energy,
                particle_charges=outgoing_particle_charges,
                device=outgoing_particles.device,
                dtype=outgoing_particles.dtype,
            )
            if outgoing_particles.shape[0] > 0
            else ParticleBeam.empty
        )

    def broadcast(self, shape: Size) -> Element:
        new_aperture = self.__class__(
            x_max=self.x_max.repeat(shape),
            y_max=self.y_max.repeat(shape),
            shape=self.shape,
            is_active=self.is_active,
            name=self.name,
        )
        new_aperture.length = self.length.repeat(shape)
        return new_aperture

    def split(self, resolution: torch.Tensor) -> list[Element]:
        # TODO: Implement splitting for aperture properly, for now just return self
        return [self]

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.4

        dummy_length = 0.0

        patch = Rectangle(
            (s, 0), dummy_length, height, color="tab:pink", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "x_max",
            "y_max",
            "shape",
            "is_active",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(x_max={repr(self.x_max)}, "
            + f"y_max={repr(self.y_max)}, "
            + f"shape={repr(self.shape)}, "
            + f"is_active={repr(self.is_active)}, "
            + f"name={repr(self.name)})"
        )
