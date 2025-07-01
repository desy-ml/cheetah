import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Species
from cheetah.track_methods import misalignment_matrix
from cheetah.utils import (
    UniqueNameGenerator,
    compute_relativistic_factors,
    verify_device_and_dtype,
)

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
    :param sanitize_name: Whether to sanitise the name to be a valid Python
        variable name. This is needed if you want to use the `segment.element_name`
        syntax to access the element in a segment.
    """

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
        device, dtype = verify_device_and_dtype(
            [length, k, misalignment], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        self.register_buffer_or_parameter(
            "k", torch.as_tensor(k if k is not None else 0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "misalignment",
            torch.as_tensor(
                misalignment if misalignment is not None else (0.0, 0.0),
                **factory_kwargs,
            ),
        )

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        device = self.length.device
        dtype = self.length.dtype

        gamma, _, _ = compute_relativistic_factors(energy, species.mass_eV)
        c = torch.cos(self.length * self.k)
        s = torch.sin(self.length * self.k)

        s_k = torch.where(self.k == 0.0, self.length, s / self.k)

        vector_shape = torch.broadcast_shapes(
            self.length.shape, self.k.shape, energy.shape
        )

        r56 = torch.where(
            gamma != 0, self.length / (1 - gamma**2), torch.zeros_like(self.length)
        )

        R = torch.eye(7, device=device, dtype=dtype).repeat((*vector_shape, 1, 1))
        R[..., 0, 0] = c**2
        R[..., 0, 1] = c * s_k
        R[..., 0, 2] = s * c
        R[..., 0, 3] = s * s_k
        R[..., 1, 0] = -self.k * s * c
        R[..., 1, 1] = c**2
        R[..., 1, 2] = -self.k * s**2
        R[..., 1, 3] = s * c
        R[..., 2, 0] = -s * c
        R[..., 2, 1] = -s * s_k
        R[..., 2, 2] = c**2
        R[..., 2, 3] = c * s_k
        R[..., 3, 0] = self.k * s**2
        R[..., 3, 1] = -s * c
        R[..., 3, 2] = -self.k * s * c
        R[..., 3, 3] = c**2
        R[..., 4, 5] = r56

        R = R.real

        if torch.all(self.misalignment == 0):
            return R
        else:
            R_entry, R_exit = misalignment_matrix(self.misalignment)
            R = torch.einsum("...ij,...jk,...kl->...il", R_exit, R, R_entry)
            return R

    @property
    def is_active(self) -> bool:
        return torch.any(self.k != 0).item()

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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"k={repr(self.k)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"name={repr(self.name)})"
        )
