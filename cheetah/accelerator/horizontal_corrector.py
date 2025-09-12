import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Species
from cheetah.utils import UniqueNameGenerator, compute_relativistic_factors

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class HorizontalCorrector(Element):
    """
    Horizontal corrector magnet in a particle accelerator.

    NOTE: This is modeled as a drift section with a thin-kick in the horizontal plane.

    :param length: Length in meters.
    :param angle: Particle deflection angle in the horizontal plane in rad.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    """

    supported_tracking_methods = ["linear"]

    def __init__(
        self,
        length: torch.Tensor,
        angle: torch.Tensor | None = None,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = length

        self.register_buffer_or_parameter(
            "angle", angle if angle is not None else torch.tensor(0.0, **factory_kwargs)
        )

    def _compute_first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        factory_kwargs = {"device": self.length.device, "dtype": self.length.dtype}

        _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

        vector_shape = torch.broadcast_shapes(
            self.length.shape, igamma2.shape, self.angle.shape
        )

        tm = torch.eye(7, **factory_kwargs).repeat((*vector_shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 1, 6] = self.angle
        tm[..., 2, 3] = self.length
        tm[..., 4, 5] = -self.length / beta.square() * igamma2

        return tm

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return (self.angle != 0).any().item()

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length
        plot_angle = self.angle[vector_idx] if self.angle.dim() > 0 else self.angle

        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (plot_angle.sign() if self.is_active else 1)

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:blue", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "angle"]
