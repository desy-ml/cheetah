import math

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, Species
from cheetah.track_methods import drift_matrix
from cheetah.utils import UniqueNameGenerator, cache_transfer_map

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class PhaseShifter(Element):
    """
    Phase shifter element for FEL undulator lines.

    Applies an explicit ponderomotive phase shift `phase_shift` (in radians, modulo
    :math:`2\\pi`) to the FEL slice macroparticles while acting as a drift of length
    `length` on the charged-particle beam and radiation field.

    :param phase_shift: Ponderomotive phase shift in radians.
    :param length: Physical length of the phase shifter in meters. Defaults to `0.0`.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.
    """

    def __init__(
        self,
        phase_shift: torch.Tensor | None = None,
        length: torch.Tensor | None = None,
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

        self.length = (
            length if length is not None else torch.tensor(0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "phase_shift",
            (
                phase_shift
                if phase_shift is not None
                else torch.tensor(0.0, **factory_kwargs)
            ),
        )

    @property
    def is_active(self) -> bool:
        return (self.phase_shift != 0.0).any().item()

    @property
    def is_skippable(self) -> bool:
        return False

    @cache_transfer_map
    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        """Return a drift transfer map of `self.length`."""
        return drift_matrix(length=self.length, energy=energy, species=species)

    def track(self, incoming: Beam) -> Beam:
        """Track beam and apply ponderomotive phase shift to FEL macroparticles."""
        outgoing = self._track_first_order(incoming)
        if hasattr(outgoing, "_fel_particles") and outgoing._fel_particles is not None:
            x, px, y, py, gammas, theta = outgoing._fel_particles
            theta = torch.remainder(theta + self.phase_shift[..., None], 2.0 * math.pi)
            outgoing._fel_particles = (x, px, y, py, gammas, theta)
        return outgoing

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        """Plot a schematic representation of the phase shifter."""
        ax = ax or plt.subplot(111)
        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length
        patch = Rectangle(
            (plot_s, -0.2), plot_length, 0.4, color="tab:cyan", alpha=0.8, zorder=2
        )
        ax.add_patch(patch)
        return ax

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "phase_shift"]
