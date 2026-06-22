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

    :param base_element: The base element over which other elements are superimposed.
    :param superimposed_element: Element or list of elements to be
        superimposed at the center of the base element.
        If a single Element is provided
    :param name: The name of the segment. If None, a default name is generated.
    :param sanitize_name: Whether to sanitize the name to ensure it is valid.
    """

    def __init__(
        self,
        base_element: Element,
        superimposed_element: Element | list[Element],
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.base_element = base_element.to(**factory_kwargs)

        if (
            isinstance(superimposed_element, Element)
            and type(superimposed_element) is not Element
        ):
            # convert to a list of one element for uniform handling
            self.superimposed_element = [superimposed_element.to(**factory_kwargs)]
        elif isinstance(superimposed_element, list) and all(
            isinstance(ele, Element) and type(ele) is not Element
            for ele in superimposed_element
        ):
            superimposed_element = [
                ele.to(**factory_kwargs) for ele in superimposed_element
            ]
            self.superimposed_element = superimposed_element
        else:
            raise TypeError(
                "Superimposed_element must be an Element subclass"
                " or a list of Element subclasses, got "
                f"{superimposed_element.__class__.__name__}"
            )

        for superimposed_ele in self.superimposed_element:
            if superimposed_ele.length is not None and not torch.all(
                superimposed_ele.length == 0
            ):
                raise ValueError(
                    f"Superimposed elements must have zero length, "
                    f"but {superimposed_ele.name} has length {superimposed_ele.length}"
                )

        self._length = self.base_element.length.clone()

        # Split base and insert superimposed element(s)
        self.update_subelements()

    @property
    def length(self) -> torch.Tensor:
        return self.base_element.length

    def update_subelements(self):
        """
        Update the subelements of the superimposed element.

        Called whenever base_element or superimposed_element is modified.
        """
        resolution = self.base_element.length.abs().max() / 2.0
        halves = self.base_element.split(resolution=resolution)

        self._segment = Segment(
            elements=[halves[0], *self.superimposed_element, halves[1]],
            name=f"{self.name}_segment",
            sanitize_name=False,
        )

    @property
    def subelements(self) -> list[Element]:
        """
        The two halves of the base element with the superimposed
        element(s) in between.
        """

        if not hasattr(self, "_segment") or self._length != self.base_element.length:
            # if base element length has been updated redraw the _segment
            self._length = self.base_element.length.clone()
            self.update_subelements()
        return self._segment.elements

    def track(self, incoming: Beam) -> Beam:
        if self.is_skippable:
            return self._track_first_order(incoming)
        else:
            todos = []
            continuous_skippable_elements = []
            for element in self.subelements:
                if element.is_skippable:
                    # Collect skippable elements until a non-skippable element is found
                    continuous_skippable_elements.append(element)
                else:
                    # If a non-skippable element is found, merge the skippable elements
                    # and append them before the non-skippable element
                    if len(continuous_skippable_elements) > 0:
                        todos.append(Segment(elements=continuous_skippable_elements))
                        continuous_skippable_elements = []

                    todos.append(element)

            # If there are still skippable elements at the end of the segment append
            # them as well
            if len(continuous_skippable_elements) > 0:
                todos.append(Segment(elements=continuous_skippable_elements))

            for todo in todos:
                incoming = todo.track(incoming)

            return incoming

    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        tm = torch.eye(7, device=energy.device, dtype=energy.dtype).repeat(
            (*energy.shape, 1, 1)
        )
        for element in self.subelements:
            tm = element.first_order_transfer_map(energy, species) @ tm
        return tm

    @property
    def is_skippable(self) -> bool:
        return all([element.is_skippable for element in self.subelements])

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        return self.base_element.plot(s, vector_idx=vector_idx, ax=ax)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["base_element", "superimposed_element"]
