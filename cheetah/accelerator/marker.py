import matplotlib.pyplot as plt
import torch

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, Species
from cheetah.utils import UniqueNameGenerator, cache_transfer_map

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Marker(Element):
    """
    General Marker / Monitor element

    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment. If `None` (default), a warning is raised for
        invalid names. Set to `True` to sanitise, or `False` to silence the warning.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element (e.g. control-system addresses or PVs). This information is *not* used
        in simulation and may contain any extra data the user wants to store along with
        the lattice. See :doc:`/examples/including_metadata` for more information.
    """

    def __init__(
        self,
        name: str | None = None,
        sanitize_name: bool | None = None,
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            name=name,
            sanitize_name=sanitize_name,
            metadata=metadata,
            device=device,
            dtype=dtype,
        )

    @cache_transfer_map
    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        return torch.eye(7, device=energy.device, dtype=energy.dtype).repeat(
            (*energy.shape, 1, 1)
        )

    def track(self, incoming: Beam) -> Beam:
        return incoming.clone()

    @property
    def is_skippable(self) -> bool:
        return True

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)
        # TODO: Implement a better visualisation for markers. At the moment they are
        # invisible.
        return ax
