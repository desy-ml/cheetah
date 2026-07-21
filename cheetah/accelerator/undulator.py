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
    :param kx: Vertical undulator parameter.
    :param ky: Horizontal undulator parameter.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment. If `None` (default), a warning is raised for
        invalid names. Set to `True` to sanitise, or `False` to silence the warning.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element (e.g. control-system addresses or PVs). This information is *not* used
        in simulation and may contain any extra data the user wants to store along with
        the lattice. See :doc:`/examples/including_metadata` for more information.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.
    """

    supported_tracking_methods = ["linear"]

    def __init__(
        self,
        length: torch.Tensor,
        period: torch.Tensor | None = None,
        kx: torch.Tensor | None = None,
        ky: torch.Tensor | None = None,
        name: str | None = None,
        sanitize_name: bool | None = None,
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
            period if period is not None else torch.tensor(1.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "kx", kx if kx is not None else torch.tensor(0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "ky", ky if ky is not None else torch.tensor(0.0, **factory_kwargs)
        )

    @property
    def is_active(self) -> bool:
        return torch.logical_or(self.kx != 0.0, self.ky != 0.0).any().item()

    @cache_transfer_map
    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        factory_kwargs = {"device": self.length.device, "dtype": self.length.dtype}

        gamma, igamma2, beta = compute_relativistic_factors(energy, species.mass_eV)

        vector_shape = torch.broadcast_shapes(
            self.length.shape,
            igamma2.shape,
            self.kx.shape,
            self.ky.shape,
            self.period.shape,
        )

        tm = torch.eye(7, **factory_kwargs).repeat((*vector_shape, 1, 1))
        tm[..., 4, 5] = (
            -self.length
            * igamma2
            * (beta.square().reciprocal() + 0.5 * (self.kx.square() + self.ky.square()))
        )

        spatial_frequency = torch.where(
            self.period > 0.0,
            math.sqrt(2) * torch.pi / (self.period * gamma * beta),
            torch.tensor(0.0),
        )

        # Transverse focusing from vertical field (Kx > 0.0)
        omega_x = spatial_frequency * self.kx
        cos_omega_x = (omega_x * self.length).cos()

        tm[..., 2, 2] = cos_omega_x
        tm[..., 2, 3] = (omega_x * self.length / torch.pi).sinc() * self.length
        tm[..., 3, 2] = -(omega_x * self.length).sin() * omega_x
        tm[..., 3, 3] = cos_omega_x

        # Transverse focusing from horizontal field (Ky > 0.0)
        omega_y = spatial_frequency * self.ky
        cos_omega_y = (omega_y * self.length).cos()

        tm[..., 0, 0] = cos_omega_y
        tm[..., 0, 1] = (omega_y * self.length / torch.pi).sinc() * self.length
        tm[..., 1, 0] = -(omega_y * self.length).sin() * omega_y
        tm[..., 1, 1] = cos_omega_y

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
        return super().defining_features + ["length", "period", "kx", "ky"]
