import torch
from matplotlib import pyplot as plt

from cheetah.accelerator.element import Element
from cheetah.accelerator.segment import Segment
from cheetah.particles.beam import Beam
from cheetah.particles.species import Species
from cheetah.utils.names import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Superimposed(Element):
    """
    A segment that represents a superimposed structure in an accelerator, i.e. where one
    element is placed over another at the centre of the base element.

    :param base_element: The base element at the centre of which the superimposed
        element is placed.
    :param superimposed_element: Element to be placed at the centre of the base element.
        NOTE: The `superimposed_element` must have a length of zero.
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

    def __init__(
        self,
        base_element: Element,
        superimposed_element: Element,
        name: str | None = None,
        sanitize_name: bool | None = None,
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            name=name, sanitize_name=sanitize_name, metadata=metadata, **factory_kwargs
        )

        self.base_element = base_element
        self.superimposed_element = superimposed_element

        if not torch.allclose(
            superimposed_element.length, torch.zeros_like(superimposed_element.length)
        ):
            raise ValueError("The superimposed element must have zero length.")

        base_element_halves = base_element.split(base_element.length / 2.0)
        if len(base_element_halves) != 2:
            raise ValueError(
                f"The base element of type {base_element.__class__.__name__} could not "
                "be split into two halves."
            )

        # Add useful names for element halves such that they can be accessed in the
        # flattened segment. These are derived from the name of this `Superimposed`
        # element rather than from `base_element.name`, because the latter may clash
        # with `self.name` during serialisation.
        half_1 = base_element_halves[0].clone()
        half_2 = base_element_halves[1].clone()
        half_1.name = f"{self.name}_1"
        half_2.name = f"{self.name}_2"

        if isinstance(superimposed_element, Segment):
            super_elements = superimposed_element.elements
        else:
            super_elements = [superimposed_element]

        self._segment = Segment(
            elements=[half_1, *super_elements, half_2],
            name=f"{self.name}_segment",
            sanitize_name=False,
        ).flattened()

    def flattened(self, skip_superimposed: bool = False) -> "Segment | Superimposed":
        if skip_superimposed:
            return self

        return self._segment.flattened()

    @property
    def is_skippable(self) -> bool:
        return self._segment.is_skippable

    @property
    def length(self) -> torch.Tensor:
        return self._segment.length

    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        return self._segment.first_order_transfer_map(energy, species)

    def track(self, incoming: Beam) -> Beam:
        return self._segment.track(incoming)

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        return self._segment.plot(s, vector_idx=vector_idx, ax=ax)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["base_element", "superimposed_element"]
