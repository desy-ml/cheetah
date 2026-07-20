import torch
from matplotlib import pyplot as plt

from cheetah.accelerator.element import Element
from cheetah.accelerator.segment import Segment
from cheetah.particles.beam import Beam
from cheetah.particles.species import Species
from cheetah.utils.unique_name_generator import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Superimposed(Element):
    """
    A segment that represents a superimposed structure in an accelerator, i.e. where one
    element is placed over another at the center of the base element.

    NOTE: Changing either `base_element` or `superimposed_element` after initialisation
        will lead to unexpected behaviour. If you need to change either of these
        elements, please create a new instance of `Superimposed`.

    :param base_element: The base element at the center of which the superimposed
        element is placed.
    :param superimposed_element: Element to be placed at the center of the base element.
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

        assert superimposed_element.length == torch.tensor(
            0.0
        ), "The superimposed element must have zero length."

        self.base_element = base_element
        self.superimposed_element = superimposed_element

        base_element_halves = base_element.split(base_element.length / 2.0)

        # check to make sure the base element was split into two halves
        if len(base_element_halves) != 2:
            raise ValueError("The base element could not be split into two halves.")

        # add useful names for element halves such that
        # it can be accessed in the flattened segment
        base_element_halves[0].name = f"{base_element.name}#1"
        base_element_halves[1].name = f"{base_element.name}#2"

        # if the base element has the same name as the
        # superimposed element, prepend an underscore to the base
        # element's name to avoid naming conflicts in the flattened segment
        if self.base_element.name == name:
            self.base_element.name = "_" + self.base_element.name

        self._segment = Segment(
            elements=[
                base_element_halves[0],
                superimposed_element,
                base_element_halves[1],
            ],
            name=f"{self.name}_segment",
            sanitize_name=sanitize_name,
        )

    def flattened(self) -> "Segment":
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
