from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, Species
from cheetah.track_methods import base_ttensor, drift_matrix, misalignment_matrix
from cheetah.utils import squash_index_for_unavailable_dims


class Sextupole(Element):
    """
    A sextupole element in a particle accelerator.

    :param length: Length in meters.
    :param k2: Sextupole strength in 1/m^3.
    :param misalignment: Transverse misalignment in x and y directions in meters.
    :param tilt: Tilt angle of the quadrupole in x-y plane in radians.
    :param tracking_method: Method to use for tracking through the element.
        Note: By default, the sextupole is created with linear tracking method so it
        will not have second order effects.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    """

    supported_tracking_methods = ["linear", "second_order"]

    def __init__(
        self,
        length: torch.Tensor,
        k2: torch.Tensor | None = None,
        misalignment: torch.Tensor | None = None,
        tilt: torch.Tensor | None = None,
        tracking_method: Literal["linear", "second_order"] = "second_order",
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        self.register_buffer_or_parameter(
            "k2", k2 if k2 is not None else torch.tensor(0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "misalignment",
            (
                misalignment
                if misalignment is not None
                else torch.tensor((0.0, 0.0), **factory_kwargs)
            ),
        )
        self.register_buffer_or_parameter(
            "tilt", tilt if tilt is not None else torch.tensor(0.0, **factory_kwargs)
        )

        self.tracking_method = tracking_method

    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        return drift_matrix(length=self.length, species=species, energy=energy)

    def second_order_transfer_map(self, energy, species):
        T = base_ttensor(
            length=self.length,
            k1=self.length.new_zeros(()),
            k2=self.k2,
            hx=self.length.new_zeros(()),
            species=species,
            tilt=self.tilt,
            energy=energy,
        )

        # Fill the first-order transfer map into the second-order transfer map
        T[..., :, 6, :] = drift_matrix(
            length=self.length, species=species, energy=energy
        )

        # Apply misalignments to the entire second-order transfer map
        if self.misalignment.any():
            R_entry, R_exit = misalignment_matrix(self.misalignment)
            T = torch.einsum(
                "...ij,...jkl,...kn,...lm->...inm", R_exit, T, R_entry, R_entry
            )

        return T

    def track(self, incoming: Beam) -> Beam:
        return (
            self._track_second_order(incoming)
            if self.tracking_method == "second_order"
            else self._track_first_order(incoming)
        )

    @property
    def is_skippable(self) -> bool:
        return False

    @property
    def is_active(self) -> torch.Tensor:
        return self.k2.any()

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_k2 = (
            self.k2[squash_index_for_unavailable_dims(vector_idx, self.k2.shape)]
            if len(self.k2.shape) > 0
            else self.k2
        )

        plot_s = (
            s[squash_index_for_unavailable_dims(vector_idx, s.shape)]
            if len(s.shape) > 0
            else s
        )
        plot_length = (
            self.length[squash_index_for_unavailable_dims(vector_idx, s.shape)]
            if len(self.length.shape) > 0
            else self.length
        )

        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (plot_k2.sign() if self.is_active else 1)
        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:orange", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "k2", "misalignment", "tilt"]
