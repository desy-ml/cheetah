import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Species
from cheetah.track_methods import misalignment_matrix
from cheetah.utils import UniqueNameGenerator, compute_relativistic_factors

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Solenoid(Element):
    """
    Solenoid magnet.

    Implemented according to A.W.Chao P74

    :param length: Length in meters.
    :param k: Normalised strength of the solenoid magnet B0/(2*Brho). B0 is the field
        inside the solenoid, Brho is the momentum of central trajectory.
    :param misalignment: Misalignment vector of the solenoid magnet in x- and
        y-directions.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    """

    supported_tracking_methods = ["linear"]

    def __init__(
        self,
        length: torch.Tensor,
        k: torch.Tensor | None = None,
        misalignment: torch.Tensor | None = None,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        self.register_buffer_or_parameter(
            "k", k if k is not None else torch.tensor(0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "misalignment",
            (
                misalignment
                if misalignment is not None
                else torch.tensor((0.0, 0.0), **factory_kwargs)
            ),
        )

    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        factory_kwargs = {"device": self.length.device, "dtype": self.length.dtype}

        gamma, _, _ = compute_relativistic_factors(energy, species.mass_eV)
        length, k, gamma = torch.broadcast_tensors(self.length, self.k, gamma)

        c = (length * k).cos()
        s = (length * k).sin()

        s_k = torch.where(k == 0.0, length, s / k)

        r56 = torch.where(
            gamma != 0, length / (1 - gamma.square()), length.new_zeros(())
        )

        R = torch.eye(7, **factory_kwargs).expand((*length.shape, 7, 7)).clone()
        R[
            ...,
            (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4),
            (0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 5),
        ] = torch.stack(
            [
                c.square(),
                c * s_k,
                s * c,
                s * s_k,
                -k * s * c,
                c.square(),
                -k * s.square(),
                s * c,
                -s * c,
                -s * s_k,
                c.square(),
                c * s_k,
                k * s.square(),
                -s * c,
                -k * s * c,
                c.square(),
                r56,
            ],
            dim=-1,
        )

        R_entry, R_exit = misalignment_matrix(self.misalignment)
        R = torch.einsum("...ij,...jk,...kl->...il", R_exit, R, R_entry)

        return R

    @property
    def is_active(self) -> torch.Tensor:
        return self.k.any()

    @property
    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: torch.Tensor) -> list[Element]:
        num_splits = (self.length.abs().max() / resolution).ceil().int()
        split_length = self.length / num_splits
        device = self.length.device
        dtype = self.length.dtype
        return [
            Solenoid(
                length=split_length,
                k=self.k,
                misalignment=self.misalignment,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_splits)
        ]

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2
        height = 0.8

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:orange", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "k", "misalignment"]
