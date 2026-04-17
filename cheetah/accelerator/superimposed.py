import torch
from matplotlib import pyplot as plt

from cheetah.accelerator import Segment
from cheetah.accelerator.element import Element
from cheetah.particles.beam import Beam
from cheetah.particles.species import Species
from cheetah.utils.unique_name_generator import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class SuperimposedElement(Element):
    """
    A segment that represents a superimposed structure in an accelerator,
    ie. where one element is placed over another at the center of the base element.

    """

    def __init__(
        self,
        base_element: Element,
        superimposed_element: Segment,
        name=None,
        sanitize_name=False,
    ):
        """
        Parameters
        ----------
        base_element : Element
            The base element over which other elements are superimposed.
        superimposed_element : Element
            Segment of elements to be superimposed at the center of base element
        name : str, optional
            The name of the segment. If None, a default name is generated.
        sanitize_name : bool, optional
            Whether to sanitize the name to ensure it is valid.
        """
        # Call the parent constructor with the composed elements
        super().__init__(name=name, sanitize_name=sanitize_name)

        self.base_element = base_element
        if isinstance(superimposed_element, Segment):
            self.superimposed_element = superimposed_element
        elif isinstance(superimposed_element, Element):
            self.superimposed_element = Segment(
                elements=[superimposed_element],
                name=f"{superimposed_element.name}_segment",
            )
        else:
            raise TypeError(
                f"superimposed_element must be a Segment or Element subclass, "
                f"got {type(superimposed_element).__name__}"
            )

    # self._buffers["length"] = self.base_element._buffers["length"]

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if name == "length" and hasattr(self, "base_element"):
            base = getattr(self, "base_element", None)
            if base is not None:
                base.length = self.length

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
    def subelements(self) -> list[Element]:
        half_length = self.base_element.length / 2
        base = self.base_element

        kwargs = {
            feature: getattr(base, feature)
            for feature in base.defining_features
            if feature != "length" and feature != "name"
        }

        first = type(base)(
            length=half_length,
            name=f"{base.name}#0",
            sanitize_name=False,
            dtype=base.length.dtype,
            device=base.length.device,
            **kwargs,
        )
        second = type(base)(
            length=half_length,
            name=f"{base.name}#1",
            sanitize_name=False,
            dtype=base.length.dtype,
            device=base.length.device,
            **kwargs,
        )

        if isinstance(self.superimposed_element, Segment):
            return [first, *self.superimposed_element.elements, second]

        raise TypeError(
            f"superimposed_element must be a Segment or Element subclass, "
            f"got {type(self.superimposed_element).__name__}"
        )

    @property
    def is_skippable(self) -> bool:
        return all([el.is_skippable for el in self.subelements])

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        return self.base_element.plot(s, vector_idx=vector_idx, ax=ax)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["base_element", "superimposed_element"]
