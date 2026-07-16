import math

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Species
from cheetah.utils import (
    UniqueNameGenerator,
    cache_transfer_map,
    compute_relativistic_factors,
)

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Undulator(Element):
    """
    Element representing an undulator in a particle accelerator.

    Implements a linear paraxial undulator field approximation taken from Ocelot, see
    S.Tomin, Varenna, 2017.

    :param length: Length in meters.
    :param period: Undulator period in meters.
    :param Kx: Vertical undulator parameter.
    :param Ky: Horizontal undulator parameter.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element (e.g. control-system addresses or PVs). This information is *not* used
        in simulation and may contain any extra data the user wants to store along with
        the lattice. See :doc:`/examples/including_metadata` for more information.
    """

    supported_tracking_methods = ["linear"]

    def __init__(
        self,
        length: torch.Tensor,
        period: torch.Tensor | None = None,
        Kx: torch.Tensor | None = None,
        Ky: torch.Tensor | None = None,
        name: str | None = None,
        sanitize_name: bool = False,
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            name=name, sanitize_name=sanitize_name, metadata=metadata, **factory_kwargs
        )

        self.length = length

        self.register_buffer_or_parameter(
            "period",
            period if period is not None else torch.tensor(0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "Kx", Kx if Kx is not None else torch.tensor(0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "Ky", Ky if Ky is not None else torch.tensor(0.0, **factory_kwargs)
        )

    @property
    def is_active(self) -> bool:
        return ((self.Kx != 0) | (self.Ky != 0)).any().item()

    @cache_transfer_map
    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        factory_kwargs = {"device": self.length.device, "dtype": self.length.dtype}

        _, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

        vector_shape = torch.broadcast_shapes(
            self.length.shape,
            igamma2.shape,
            self.Kx.shape,
            self.Ky.shape,
            self.period.shape,
        )

        tm = torch.eye(7, **factory_kwargs).repeat((*vector_shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 2, 3] = self.length

        K_sq = self.Kx**2 + self.Ky**2

        # Longitudinal: R56 = -L/(gamma²·beta²) * (1 + 0.5·K²·beta²)
        tm[..., 4, 5] = (
            -self.length / beta.square() * igamma2 * (1 + 0.5 * K_sq * beta.square())
        )

        gamma = 1 / torch.sqrt(1 - beta.square() + 1e-30)  # one-to-one with beta

        # Transverse focusing from vertical field (Kx > 0) -> y-plane oscillations
        nonzero_Kx = self.Kx.abs() > 1e-15
        omega_x = math.sqrt(2) * math.pi * self.Kx / (self.period * gamma * beta)
        cos_omega_x = (omega_x * self.length).cos()
        sin_omega_x = (omega_x * self.length).sin()
        omega_x_safe = omega_x.clamp(min=1e-30)

        tm[..., 2, 2] = torch.where(nonzero_Kx, cos_omega_x, tm[..., 2, 2])
        tm[..., 2, 3] = torch.where(
            nonzero_Kx, sin_omega_x / omega_x_safe, tm[..., 2, 3]
        )
        tm[..., 3, 2] = torch.where(nonzero_Kx, -sin_omega_x * omega_x, tm[..., 3, 2])
        tm[..., 3, 3] = torch.where(nonzero_Kx, cos_omega_x, tm[..., 3, 3])

        # Transverse focusing from horizontal field (Ky > 0) -> x-plane oscillations
        nonzero_Ky = self.Ky.abs() > 1e-15
        omega_y = math.sqrt(2) * math.pi * self.Ky / (self.period * gamma * beta)
        cos_omega_y = (omega_y * self.length).cos()
        sin_omega_y = (omega_y * self.length).sin()
        omega_y_safe = omega_y.clamp(min=1e-30)

        tm[..., 0, 0] = torch.where(nonzero_Ky, cos_omega_y, tm[..., 0, 0])
        tm[..., 0, 1] = torch.where(
            nonzero_Ky, sin_omega_y / omega_y_safe, tm[..., 0, 1]
        )
        tm[..., 1, 0] = torch.where(nonzero_Ky, -sin_omega_y * omega_y, tm[..., 1, 0])
        tm[..., 1, 1] = torch.where(nonzero_Ky, cos_omega_y, tm[..., 1, 1])

        return tm

    @property
    def is_skippable(self) -> bool:
        return True

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

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
        return super().defining_features + ["length", "Kx", "Ky", "period"]
