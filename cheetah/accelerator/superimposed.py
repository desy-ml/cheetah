from matplotlib import pyplot as plt
import torch
from cheetah.accelerator import Segment
from cheetah.accelerator.element import Element
from cheetah.particles.beam import Beam
from cheetah.particles.species import Species


class SuperimposedElement(Element):
    """
    A segment that represents a superimposed structure in an accelerator, 
    ie. where one element is placed over another at the center of the base element.
    
    """
    def __init__(
        self, 
        base_element: Element,
        superimposed_element: Element,
        name = None, 
        sanitize_name = False
    ):

        """
        Parameters
        ----------
        base_element : Element
            The base element over which other elements are superimposed.
        superimposed_element : Element
            Element to be superimposed at the center of the base element.
        name : str, optional
            The name of the segment. If None, a default name is generated.
        sanitize_name : bool, optional
            Whether to sanitize the name to ensure it is valid.
        """
        # Call the parent constructor with the composed elements
        super().__init__(name=name, sanitize_name=sanitize_name)

        self.base_element = base_element
        self.superimposed_element = superimposed_element

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
        if self.is_skippable:
            tm = torch.eye(7, device=energy.device, dtype=energy.dtype)
            for element in self.subelements:
                tm = element.first_order_transfer_map(energy, species) @ tm
            return tm
        else:
            return None

    @property
    def subelements(self) -> list[Element]:
        base_split = self.base_element.split(self.base_element.length / 2)
        base_split[0].name = f"{self.base_element.name}#0"
        base_split[1].name = f"{self.base_element.name}#1"

        elements = [base_split[0], self.superimposed_element, base_split[1]]

        return elements

    @property
    def is_skippable(self) -> bool:
        return all([el.is_skippable for el in self.subelements])

    @property
    def length(self) -> float:
        return self.base_element.length
    
    def split(self, resolution):
        raise NotImplementedError("Splitting a SuperimposedElement is not supported yet.")
    
    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        return self.base_element.plot(s, vector_idx=vector_idx, ax=ax)
    
    @property
    def defining_features(self) -> list[str]:
        return ["base_element", "superimposed_element"]
