import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Species
from cheetah.utils import (
    UniqueNameGenerator,
    compute_relativistic_factors,
    verify_device_and_dtype,
)

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class VerticalCorrector(Element):
    """
    Verticle corrector magnet in a particle accelerator.
    Note: This is modeled as a drift section with
        a thin-kick in the vertical plane.

    :param length: Length in meters.
    :param angle: Particle deflection angle in the vertical plane in rad.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: torch.Tensor,
        angle: torch.Tensor | None = None,
        name: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype([length, angle], device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        self.register_buffer_or_parameter(
            "angle",
            torch.as_tensor(angle if angle is not None else 0.0, **factory_kwargs),
        )

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        device = self.length.device
        dtype = self.length.dtype

        _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

        vector_shape = torch.broadcast_shapes(
            self.length.shape, igamma2.shape, self.angle.shape
        )

        tm = torch.eye(7, device=device, dtype=dtype).repeat((*vector_shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 2, 3] = self.length
        tm[..., 3, 6] = self.angle
        tm[..., 4, 5] = -self.length / beta**2 * igamma2

        return tm

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return torch.any(self.angle != 0).item()

    def split(self, resolution: torch.Tensor) -> list[Element]:
        num_splits = torch.ceil(torch.max(self.length) / resolution).int()
        return [
            VerticalCorrector(
                self.length / num_splits,
                self.angle / num_splits,
                dtype=self.length.dtype,
                device=self.length.device,
            )
            for _ in range(num_splits)
        ]

    def plot(self, ax: plt.Axes, s: float, vector_idx: tuple | None = None) -> None:
        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length
        plot_angle = self.angle[vector_idx] if self.angle.dim() > 0 else self.angle

        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (torch.sign(plot_angle) if self.is_active else 1)

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:cyan", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "angle"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"angle={repr(self.angle)}, "
            + f"name={repr(self.name)})"
        )
