from copy import deepcopy
from functools import reduce
from pathlib import Path
from typing import Any, Iterator, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Polygon
from torch import nn

from cheetah import latticejson
from cheetah.accelerator.aperture import Aperture
from cheetah.accelerator.bpm import BPM
from cheetah.accelerator.custom_transfer_map import CustomTransferMap
from cheetah.accelerator.dipole import Dipole
from cheetah.accelerator.drift import Drift
from cheetah.accelerator.element import Element
from cheetah.accelerator.marker import Marker
from cheetah.accelerator.screen import Screen
from cheetah.converters import bmad, elegant, nxtables
from cheetah.particles import Beam, Species
from cheetah.utils import (
    UniqueNameGenerator,
    merge_element_names,
    squash_index_for_unavailable_dims,
)

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

DEFAULT_ELEMENT_COLORS: dict[str, str] = {
    "Dipole": "tab:green",
    "RBend": "tab:green",
    "Quadrupole": "tab:red",
    "Sextupole": "tab:orange",
    "Cavity": "gold",
    "TransverseDeflectingCavity": "olive",
    "Solenoid": "tab:orange",
    "Undulator": "tab:purple",
    "HorizontalCorrector": "tab:blue",
    "VerticalCorrector": "tab:cyan",
    "CombinedCorrector": "chocolate",
    "CustomTransferMap": "tab:olive",
    "Screen": "tab:green",
    "BPM": "darkkhaki",
    "Aperture": "tab:pink",
    "SpaceChargeKick": "orange",
    "Marker": "tab:grey",
}


class Segment(Element):
    """
    Segment of a particle accelerator consisting of several elements.

    :param cell: List of Cheetah elements that describe an accelerator (section).
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment. If `None` (default), a warning is raised for
        invalid names. Set to `True` to sanitise, or `False` to silence the warning.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element (e.g. control-system addresses or PVs). This information is *not* used
        in simulation and may contain any extra data the user wants to store along with
        the lattice. See :doc:`/getting_started/control_systems` for more information.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.
    """

    def __init__(
        self,
        elements: list[Element],
        name: str | None = None,
        sanitize_name: bool | None = None,
        metadata: dict | None = None,
    ) -> None:
        super().__init__(name=name, sanitize_name=sanitize_name, metadata=metadata)

        # Segment inherits `length` as a buffer from `Element`. Since `length` is
        # overwritten as a standard Python property, this is misleading when calling
        # `Segment.buffers()`. We therefore manually remove `length` from the list of
        # buffers.
        del self._buffers["length"]

        self.register_module("elements", nn.ModuleList(elements))

        for element in self.elements:
            # Make elements accessible via .name attribute. If multiple elements have
            # the same name, they are accessible via a list.
            if element.name in self.__dict__:
                if isinstance(self.__dict__[element.name], list):
                    self.__dict__[element.name].append(element)
                else:  # Is instance of cheetah.Element
                    self.__dict__[element.name] = [self.__dict__[element.name], element]
            else:
                self.__dict__[element.name] = element

    @property
    def element_names(self) -> list[str]:
        """
        Ordered list of names of the elements in the segment.

        :return: Ordered list of names of the elements in the segment.
        """
        return [element.name for element in self.elements]

    def element_index(self, element_name: str) -> int:
        """
        Get the index of the first occurrence of an element in the segment by its name.

        :param element_name: Name of the element to get the index for.
        :return: Index of the first element in the segment with the passed name.
        """
        try:
            return self.element_names.index(element_name)
        except ValueError:
            raise ValueError(f"Element '{element_name}' not found in segment.")

    def subcell(
        self,
        start: str | None = None,
        end: str | None = None,
        include_start: bool = True,
        include_end: bool = True,
    ) -> "Segment":
        """
        Extract a subcell from this segment.

        If either `start` or `end` is `None`, the subcell starts or ends at the same
        element as the original segment. If `start` or `end` is not part of the segment,
        a `ValueError` is raised.

        :param start: Name of the element at the start of the subcell. If `None` is
            passed, the subcell starts at the same element as the original segment.
        :param end: Name of the element at the end of the subcell. If `None` is
            passed, the subcell ends at the same element as the original segment.
        :param include_start: If `True`, `start` is included in the subcell, otherwise
            not.
        :param include_end: If `True`, `end` is included in the subcell, otherwise not.
        :return: Subcell of elements from `start` to `end`.
        """
        is_start_in_segment = start is None or start in self.__dict__
        if not is_start_in_segment:
            raise ValueError(f"Element {start} is not part of the segment.")
        is_end_in_segment = end is None or end in self.__dict__
        if not is_end_in_segment:
            raise ValueError(f"Element {end} is not part of the segment.")

        subcell = []
        is_in_subcell = start is None
        for element in self.elements:
            if element.name == start:
                is_in_subcell = True
                if include_start:
                    subcell.append(element)
                continue

            if element.name == end:
                if include_end and is_in_subcell:
                    subcell.append(element)
                break

            if is_in_subcell:
                subcell.append(element)

        return self.__class__(subcell, metadata=deepcopy(self.metadata))

    def flattened(self) -> "Segment":
        """
        Return a flattened version of the segment, i.e. one where all subsegments are
        resolved and their elements entered into a top-level segment.
        """
        flattened_elements = []
        for element in self.elements:
            if isinstance(element, Segment):
                flattened_elements += element.flattened().elements
            elif hasattr(element, "flattened"):
                flattened_elements += element.flattened().elements
            else:
                flattened_elements.append(element)

        return self.__class__(
            elements=flattened_elements,
            name=self.name,
            sanitize_name=False,
            metadata=deepcopy(self.metadata),
        )

    def reversed(self) -> "Segment":
        """
        Return a reversed version of the segment, i.e. one where the order of the
        elements is reversed.
        """
        reversed_elements = list(
            reversed(
                [
                    element.reversed() if isinstance(element, Segment) else element
                    for element in self.elements
                ]
            )
        )

        return self.__class__(
            elements=reversed_elements,
            name=f"{self.name}_reversed",
            sanitize_name=False,
            metadata=deepcopy(self.metadata),
        )

    def transfer_maps_merged(
        self, incoming_beam: Beam, except_for: list[str] | None = None
    ) -> "Segment":
        """
        Return a segment where the transfer maps of skipable elements are merged into
        elements of type `CustomTransferMap`. This can be used to speed up tracking
        through the segment.

        :param incoming_beam: Beam that is incoming to the segment. NOTE: This beam is
            needed to determine the energy of the beam when entering each element, as
            the transfer maps of merged elements might depend on the beam energy.
        :param except_for: List of names of elements that should not be merged despite
            being skippable. Usually these are the elements that are changed from one
            tracking to another.
        :return: Segment with merged transfer maps.
        """
        if except_for is None:
            except_for = []

        merged_elements = []  # Elements for new merged segment
        skippable_elements = []  # Keep track of elements that are not yet merged
        tracked_beam = incoming_beam
        for element in self.elements:
            if element.is_skippable and element.name not in except_for:
                skippable_elements.append(element)
            else:
                if len(skippable_elements) == 1:
                    merged_elements.append(skippable_elements[0])
                    tracked_beam = skippable_elements[0].track(tracked_beam)
                elif len(skippable_elements) > 1:  # i.e. we need to merge some elements
                    merged_elements.append(
                        CustomTransferMap.from_merging_elements(
                            skippable_elements, incoming_beam=tracked_beam
                        )
                    )
                    tracked_beam = merged_elements[-1].track(tracked_beam)
                skippable_elements = []

                merged_elements.append(element)
                tracked_beam = element.track(tracked_beam)

        if len(skippable_elements) > 0:
            merged_elements.append(
                CustomTransferMap.from_merging_elements(
                    skippable_elements, incoming_beam=tracked_beam
                )
            )

        return self.__class__(
            elements=merged_elements,
            name=self.name,
            sanitize_name=False,
            metadata=deepcopy(self.metadata),
        )

    def without_inactive_markers(
        self, except_for: list[str] | None = None
    ) -> "Segment":
        """
        Return a segment where all inactive markers are removed. This can be used to
        speed up tracking through the segment.

        NOTE: `is_active` has not yet been implemented for Markers. Therefore, this
        function currently removes all markers.

        :param except_for: List of names of elements that should not be removed despite
            being inactive.
        :return: Segment without inactive markers.
        """
        # TODO: Add check for is_active once that has been implemented for Markers
        if except_for is None:
            except_for = []

        return self.__class__(
            elements=[
                element
                for element in self.elements
                if not isinstance(element, Marker) or element.name in except_for
            ],
            name=self.name,
            sanitize_name=False,
            metadata=deepcopy(self.metadata),
        )

    def without_inactive_zero_length_elements(
        self, except_for: list[str] | None = None
    ) -> "Segment":
        """
        Return a segment where all inactive zero length elements are removed. This can
        be used to speed up tracking through the segment.

        NOTE: If `is_active` is not implemented for an element, it is assumed to be
        inactive and will be removed.

        :param except_for: List of names of elements that should not be removed despite
            being inactive and having a zero length.
        :return: Segment without inactive zero length elements.
        """
        if except_for is None:
            except_for = []

        return self.__class__(
            elements=[
                element
                for element in self.elements
                if (element.length != 0.0).any()
                or (hasattr(element, "is_active") and element.is_active)
                or element.name in except_for
            ],
            name=self.name,
            sanitize_name=False,
            metadata=deepcopy(self.metadata),
        )

    def inactive_elements_as_drifts(
        self, except_for: list[str] | None = None
    ) -> "Segment":
        """
        Return a segment where all inactive elements (that have a length) are replaced
        by drifts. This can be used to speed up tracking through the segment and is a
        valid thing to as inactive elements should basically be no different from drift
        sections.

        :param except_for: List of names of elements that should not be replaced by
            drifts despite being inactive. Usually these are the elements that are
            currently inactive but will be activated later.
        :return: Segment with inactive elements replaced by drifts.
        """
        if except_for is None:
            except_for = []

        return self.__class__(
            elements=[
                (
                    element
                    if (hasattr(element, "is_active") and element.is_active)
                    or (element.length == 0.0).all()
                    or element.name in except_for
                    else Drift(
                        element.length,
                        name=element.name,
                        device=element.length.device,
                        dtype=element.length.dtype,
                        sanitize_name=False,
                        metadata=deepcopy(element.metadata),
                    )
                )
                for element in self.elements
            ],
            name=self.name,
            sanitize_name=False,
            metadata=deepcopy(self.metadata),
        )

    def with_consecutive_elements_merged(
        self, except_for: list[str] | None = None
    ) -> "Segment":
        """
        Return a segment where consecutive elements of the same type that can be merged
        are combined into single elements.

        :param except_for: List of names of elements that should not be merged despite
            being mergeable.
        :return: Segment with consecutive mergeable elements merged.
        """
        if except_for is None:
            except_for = []

        merged_elements = []
        current = self.elements[0]
        for next_element in self.elements[1:]:
            if current.name not in except_for:
                if type(current) is Segment:
                    current = current.with_consecutive_elements_merged(
                        except_for=except_for
                    )
                elif (
                    type(current) is type(next_element)
                    and next_element.name not in except_for
                ):
                    merged = current.merge(next_element)
                    if merged is not None:
                        current = merged
                        continue  # Don't do merged_elements.append(current) and advance

            merged_elements.append(current)
            current = next_element

        merged_elements.append(current)

        return self.__class__(
            elements=merged_elements,
            name=self.name,
            sanitize_name=False,
            metadata=deepcopy(self.metadata),
        )

    @classmethod
    def from_lattice_json(
        cls,
        filepath: str,
        sanitize_names: bool | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Segment":
        """
        Load a Cheetah model from a JSON file.

        :param filepath: Path of the file to load the lattice from.
        :param sanitize_names: Whether to sanitise the names of the elements to be valid
            Python variable names. This is needed if you want to use the
            `segment.element_name` syntax to access the element in a segment. If `None`
            (default), a warning is raised for invalid names. Set to `True` to sanitise,
            or `False` to silence the warning.
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Loaded Cheetah `Segment`.
        """
        return latticejson.load_cheetah_model(
            filepath, sanitize_names=sanitize_names, device=device, dtype=dtype
        )

    def to_lattice_json(
        self,
        filepath: str,
        title: str | None = None,
        info: str = "This is a placeholder lattice description",
    ) -> None:
        """
        Save a Cheetah model to a JSON file.

        :param filename: Name/path of the file to save the lattice to.
        :param title: Title of the lattice. If not provided, defaults to the name of the
            `Segment` object. If that also does not have a name, defaults to "Unnamed
            Lattice".
        :param info: Information about the lattice. Defaults to "This is a placeholder
            lattice description".
        """
        latticejson.save_cheetah_model(self, filepath, title, info)

    @classmethod
    def from_ocelot(
        cls,
        cell,
        name: str | None = None,
        sanitize_names: bool | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "Segment":
        """
        Translate an Ocelot cell to a Cheetah `Segment`.

        NOTE Objects not supported by Cheetah are translated to drift sections. Screen
        objects are created only from `ocelot.Monitor` objects when the string "BSC" is
        contained in their `id` attribute. Their screen properties are always set to
        default values and most likely need adjusting afterwards. BPM objects are only
        created from `ocelot.Monitor` objects when their id has a substring "BPM".

        :param cell: Ocelot cell, i.e. a list of Ocelot elements to be converted.
        :param name: Unique identifier for the entire segment.
        :param sanitize_names: Whether to sanitise the names of the elements to be valid
            Python variable names. This is needed if you want to use the
            `segment.element_name` syntax to access the element in a segment. If `None`
            (default), a warning is raised for invalid names. Set to `True` to sanitise,
            or `False` to silence the warning.
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Cheetah segment closely resembling the Ocelot cell.
        """
        from cheetah.converters import ocelot

        converted = [
            ocelot.convert_element(
                element,
                sanitize_name=sanitize_names,
                device=device,
                dtype=dtype,
            )
            for element in cell
        ]
        return cls(converted, name=name, sanitize_name=sanitize_names, **kwargs)

    @classmethod
    def from_bmad(
        cls,
        bmad_lattice_file_path: str,
        environment_variables: dict | None = None,
        sanitize_names: bool | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Segment":
        """
        Read a Cheetah segment from a Bmad lattice file.

        NOTE: This function was designed at the example of the LCLS lattice. While this
        lattice is extensive, this function might not properly convert all features of
        a Bmad lattice. If you find that this function does not work for your lattice,
        please open an issue on GitHub.

        :param bmad_lattice_file_path: Path to the Bmad lattice file.
        :param environment_variables: Dictionary of environment variables to use when
            parsing the lattice file.
        :param sanitize_names: Whether to sanitise the names of the elements to be valid
            Python variable names. This is needed if you want to use the
            `segment.element_name` syntax to access the element in a segment. If `None`
            (default), a warning is raised for invalid names. Set to `True` to sanitise,
            or `False` to silence the warning.
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Cheetah `Segment` representing the Bmad lattice.
        """
        bmad_lattice_file_path = Path(bmad_lattice_file_path)
        return bmad.convert_lattice(
            bmad_lattice_file_path, environment_variables, sanitize_names, device, dtype
        )

    @classmethod
    def from_elegant(
        cls,
        elegant_lattice_file_path: str,
        name: str,
        sanitize_names: bool | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Segment":
        """
        Read a Cheetah segment from an Elegant lattice file.

        :param elegant_lattice_file_path: Path to the Elegant lattice file.
        :param name: Name of the root element.
        :param sanitize_names: Whether to sanitise the names of the elements to be valid
            Python variable names. This is needed if you want to use the
            `segment.element_name` syntax to access the element in a segment. If `None`
            (default), a warning is raised for invalid names. Set to `True` to sanitise,
            or `False` to silence the warning.
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Cheetah `Segment` representing the Elegant lattice.
        """
        elegant_lattice_file_path = Path(elegant_lattice_file_path)
        return elegant.convert_lattice(
            elegant_lattice_file_path, name, sanitize_names, device, dtype
        )

    @classmethod
    def from_nx_tables(cls, filepath: Path | str) -> "Element":
        """
        Read an NX Tables CSV-like file generated for the ARES lattice into a Cheetah
        `Segment`.

        NOTE: This format is specific to the ARES accelerator at DESY.

        :param filepath: Path to the NX Tables file.
        :return: Converted Cheetah `Segment`.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        return nxtables.convert_lattice(filepath)

    @property
    def is_skippable(self) -> bool:
        return all(element.is_skippable for element in self.elements)

    @property
    def length(self) -> torch.Tensor:
        lengths = [element.length for element in self.elements]
        return reduce(torch.add, lengths)

    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        if self.is_skippable:
            tm = torch.eye(7, device=energy.device, dtype=energy.dtype)
            for element in self.elements:
                tm = element.first_order_transfer_map(energy, species) @ tm
            return tm
        else:
            return None

    def track(self, incoming: Beam) -> Beam:
        if self.is_skippable:
            return super()._track_first_order(incoming)
        else:
            todos = []
            continuous_skippable_elements = []
            for element in self.elements:
                if element.is_skippable:
                    # Collect skippable elements until a non-skippable element is found
                    continuous_skippable_elements.append(element)
                else:
                    # If a non-skippable element is found, merge the skippable elements
                    # and append them before the non-skippable element
                    if len(continuous_skippable_elements) > 0:
                        todos.append(
                            self.__class__(elements=continuous_skippable_elements)
                        )
                        continuous_skippable_elements = []

                    todos.append(element)

            # If there are still skippable elements at the end of the segment append
            # them as well
            if len(continuous_skippable_elements) > 0:
                todos.append(self.__class__(elements=continuous_skippable_elements))

            for todo in todos:
                incoming = todo.track(incoming)

            return incoming

    def clone(self) -> "Segment":
        return self.__class__(
            elements=[element.clone() for element in self.elements],
            name=self.name,
            metadata=deepcopy(self.metadata),
            sanitize_name=False,
        )

    def split(self, resolution: torch.Tensor) -> list[Element]:
        return [
            split_element
            for element in self.elements
            for split_element in element.split(resolution)
        ]

    def merge(self, other: "Segment") -> "Segment | None":
        return self.__class__(
            elements=self.elements + other.elements,
            name=merge_element_names(self.name, other.name),
            sanitize_name=False,
            metadata={**other.metadata, **self.metadata},
        )

    def partition_at(
        self, element_name: str, mode: Literal["before", "after", "both"] = "both"
    ) -> tuple[Element, ...]:
        """
        Partition the segment into multiple subcells around a named element. If the
        segment is split neither before nor after the named element, a single-element
        tuple containing the original segment is returned.

        :param element_name: Name of the element at which the segment is split.
        :param mode: Mode of partitioning. If "before", the segment is split before the
            named element. If "after", the segment is split after the named element. If
            "both", the segment is split both before and after the named element.
        :return: Segment partition. May contain 1, 2, or 3 subcells depending on whether
            the segment is split before and/or after the named element.
        """
        index = self.element_index(element_name)
        pre_cell = (
            self.__class__(self.elements[: index + 1], metadata=deepcopy(self.metadata))
            if mode == "after"
            else self.__class__(self.elements[:index], metadata=deepcopy(self.metadata))
        )
        post_cell = (
            self.__class__(self.elements[index:], metadata=deepcopy(self.metadata))
            if mode == "before"
            else self.__class__(
                self.elements[index + 1 :], metadata=deepcopy(self.metadata)
            )
        )
        return (
            (pre_cell, self.elements[index], post_cell)  # Two splits: before and after
            if mode == "both"
            else (pre_cell, post_cell)  # One split: before or after
        )

    def beam_along_segment_generator(
        self, incoming: Beam, resolution: float | None = None
    ) -> Iterator[Beam]:
        """
        Generator for beam objects along the segment either at the end of each element
        or at a given resolution.

        :param incoming: Beam that is entering the segment from upstream for which the
            trajectory is computed.
        :param resolution: Requested resolution of trajectory. Note that not all
            elements can be split at arbitrary resolutions, which can lead to deviations
            from the requested resolution. If `None` is passed, samples are taken at the
            end of each element.
        :return: Generator that yields the beam objects along the segment.
        """
        # If a resolution is passed, run this method for the split Segment
        if resolution is not None:
            yield from self.__class__(
                elements=self.split(resolution), name=f"{self.name}_split"
            ).beam_along_segment_generator(incoming)
        else:
            yield incoming
            for element in self.elements:
                outgoing = element.track(incoming)
                yield outgoing
                incoming = outgoing

    def get_beam_attrs_along_segment(
        self,
        attr_names: tuple[str, ...] | str,
        incoming: Beam,
        resolution: float | None = None,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """
        Get metrics along the segment either at the end of each element or at a given
        resolution.

        :param attr_names: Metrics to compute. Can be a single metric or a tuple of
            metrics. Supported metrics are any property of beam class of `incoming`.
        :param incoming: Beam that is entering the segment from upstream for which the
            trajectory is computed.
        :param resolution: Requested resolution of trajectory. Note that not all
            elements can be split at arbitrary resolutions, which can lead to deviations
            from the requested resolution. If `None` is passed, samples are taken at the
            end of each element.
        :return: Tuple of tensors containing the requested metrics along the segment.
        """
        attr_name_tuple = attr_names if isinstance(attr_names, tuple) else (attr_names,)

        results = zip(
            *(
                tuple(getattr(beam, attr_name) for attr_name in attr_name_tuple)
                for beam in self.beam_along_segment_generator(
                    incoming, resolution=resolution
                )
            )
        )
        broadcasted_results = tuple(
            torch.stack(
                torch.broadcast_tensors(*attr_tensor),
                dim=-(incoming.UNVECTORIZED_NUM_ATTR_DIMS.get(attr_name, 0) + 1),
            )
            for attr_tensor, attr_name in zip(results, attr_name_tuple)
        )

        return (
            broadcasted_results
            if isinstance(attr_names, tuple)
            else broadcasted_results[0]
        )

    def set_attrs_on_every_element(
        self,
        filter_type: type[Element] | tuple[type[Element]] | None = None,
        is_recursive: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Set attributes on every element of a specific type in the segment.

        :param filter_type: Type of the elements to set the attributes for.
        :param is_recursive: If `True`, the this method is applied to nested `Segment`s
            as well. If `False`, only the elements directly in the top-level `Segment`
            are considered.
        :param kwargs: Attributes to set and their values.
        """
        for element in self.elements:
            if filter_type is None or isinstance(element, filter_type):
                for key, value in kwargs.items():
                    setattr(element, key, value)
            elif is_recursive and isinstance(element, Segment):
                element.set_attrs_on_every_element(
                    filter_type=filter_type, is_recursive=True, **kwargs
                )

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        element_lengths = [element.length for element in self.elements]
        broadcast_element_lengths = torch.broadcast_tensors(*element_lengths)
        stacked_element_lengths = torch.stack(broadcast_element_lengths, dim=-1)
        element_end_s_positions = torch.cumsum(stacked_element_lengths, dim=-1)
        s_positions = torch.cat(
            (
                torch.zeros_like(element_end_s_positions[..., :1]),
                element_end_s_positions,
            ),
            dim=-1,
        )

        # The element lengths might not capture the entire vector shape with the
        # incoming beam used in plotting functions that might use this function. The
        # following tries to extract the correct vector index for just the s positions
        # of the elements, while preserving the ability to use this element plotting
        # function without an incoming beam.
        plot_ss = (
            s_positions[
                squash_index_for_unavailable_dims(vector_idx, s_positions.shape)
            ]
            if s_positions.dim() > 1
            else s_positions
        ).detach()

        ax.plot([0, plot_ss[-1]], [0, 0], "--", color="black")

        for element, s in zip(self.elements, plot_ss[:-1]):
            element.plot(s, vector_idx, ax)

        ax.set_ylim(-1, 1)
        ax.set_xlabel("s (m)")
        ax.set_yticks([])

        return ax

    def plot_mean_and_std(
        self,
        incoming: Beam,
        resolution: float | None = None,
        vector_idx: tuple | None = None,
        axx: plt.Axes | None = None,
        axy: plt.Axes | None = None,
    ) -> tuple[plt.Axes, plt.Axes]:
        """
        Plot the mean (i.e. beam position) and standard deviation (i.e. beam size) of
        the beam along the segment view in x- and y-direction.

        :param incoming: Entering beam for which the position and size are shown
        :param resolution: Minimum resolution of the tracking of the beam position and
            beam size in the plot.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        :param axx: Axes to plot the particle traces into viewed in x-direction. Must
            be provided together with `axy`, or both must be `None` to create new axes.
        :param axy: Axes to plot the particle traces into viewed in y-direction. Must
            be provided together with `axx`, or both must be `None` to create new axes.
        :return: Both axes with the plotted mean and standard deviation of the beam
            along the segment.
        """
        if axx is None and axy is None:
            _, (axx, axy) = plt.subplots(2, 1, sharex=True)
        elif axx is None or axy is None:
            raise ValueError("Either provide both axx and axy, or neither.")

        reference_segment = self.clone()  # Prevent side effects when plotting

        ss, x_means, x_stds, y_means, y_stds = (
            reference_segment.get_beam_attrs_along_segment(
                ("s", "mu_x", "sigma_x", "mu_y", "sigma_y"),
                incoming,
                resolution=resolution,
            )
        )
        ss, x_means, x_stds, y_means, y_stds = torch.broadcast_tensors(
            ss, x_means, x_stds, y_means, y_stds
        )

        plot_ss, plot_x_means, plot_x_stds, plot_y_means, plot_y_stds = (
            (metric[vector_idx] if metric.dim() > 1 else metric).detach()
            for metric in (ss, x_means, x_stds, y_means, y_stds)
        )

        axx.plot(plot_ss, plot_x_means)
        axx.fill_between(
            plot_ss, plot_x_means - plot_x_stds, plot_x_means + plot_x_stds, alpha=0.4
        )

        axy.plot(plot_ss, plot_y_means)
        axy.fill_between(
            plot_ss, plot_y_means - plot_y_stds, plot_y_means + plot_y_stds, alpha=0.4
        )

        axx.set_xlabel("s (m)")
        axx.set_ylabel("x (m)")
        axy.set_xlabel("s (m)")
        axy.set_ylabel("y (m)")

        return axx, axy

    def plot_overview(
        self,
        incoming: Beam,
        resolution: float | None = None,
        vector_idx: tuple | None = None,
        fig: matplotlib.figure.Figure | None = None,
    ) -> matplotlib.figure.Figure:
        """
        Plot an overview of the segment with the lattice along with the beam position
        and size.

        :param incoming: Entering beam for which the position and size are shown.
        :param resolution: Minimum resolution of the tracking of the beam position and
            beam size in the plot.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        :param fig: Figure to plot the overview into.
        :return: Figure with the plotted overview.
        """
        if fig is None:
            fig = plt.figure()
        gs = fig.add_gridspec(3, hspace=0, height_ratios=[2, 2, 1])
        axs = gs.subplots(sharex=True)

        axs[0].set_title("Beam Position and Size")
        self.plot_mean_and_std(
            axx=axs[0],
            axy=axs[1],
            incoming=incoming,
            resolution=resolution,
            vector_idx=vector_idx,
        )

        _ = self.plot(ax=axs[2], s=0.0, vector_idx=vector_idx)

        return fig

    def plot_beam_attrs(
        self,
        incoming: Beam,
        attr_names: tuple[str, ...] | str,
        resolution: float | None = None,
        vector_idx: tuple | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot beam attributes along the segment.

        :param incoming: Beam that is entering the segment from upstream for which the
            trajectory is computed.
        :param attr_names: Metrics to compute. Can be a single metric or a tuple of
            metrics. Supported metrics are any property of beam class of `incoming`.
        :param resolution: Requested resolution of trajectory. Note that not all
            elements can be split at arbitrary resolutions, which can lead to deviations
            from the requested resolution. If `None` is passed, samples are taken at the
            end of each element.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        :param ax: Axes to plot into.
        :return: Axes with the plotted beam attributes.
        """
        attr_names_with_s = ("s",) + (
            attr_names if isinstance(attr_names, tuple) else (attr_names,)
        )
        beam_attrs = self.get_beam_attrs_along_segment(
            attr_names_with_s, incoming, resolution=resolution
        )

        ax = ax or plt.subplot(111)

        s = beam_attrs[0]
        for attr, attr_name in zip(beam_attrs[1:], attr_names_with_s[1:]):
            ax.plot(
                (s[vector_idx] if s.dim() > 1 else s).detach(),
                (attr[vector_idx] if attr.dim() > 1 else attr).detach(),
                label=attr_name,
            )

        ax.legend()

        return ax

    def plot_beam_attrs_over_lattice(
        self,
        incoming: Beam,
        attr_names: tuple[str, ...] | str,
        resolution: float | None = None,
        vector_idx: tuple | None = None,
        fig: matplotlib.figure.Figure | None = None,
    ) -> matplotlib.figure.Figure:
        """
        Plot beam attributes in a plot over a plot of the lattice.

        :param incoming: Beam that is entering the segment from upstream for which the
            trajectory is computed.
        :param attr_names: Metrics to compute. Can be a single metric or a tuple of
            metrics. Supported metrics are any property of beam class of `incoming`.
        :param resolution: Minimum resolution of the tracking of the beam position and
            beam size in the plot.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        :param fig: Figure to plot into. Can be used, for example to pass a figure with
            a desired custom `figsize`. If `None`, a new figure with default size is
            created.
        :return: Figure with the plotted beam attributes over the lattice.
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 4))
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
        axs = gs.subplots(sharex=True)

        self.plot_beam_attrs(
            incoming=incoming,
            attr_names=attr_names,
            resolution=resolution,
            vector_idx=vector_idx,
            ax=axs[0],
        )
        self.plot(s=0.0, ax=axs[1], vector_idx=vector_idx)

        return fig

    def plot_twiss(
        self, incoming: Beam, vector_idx: tuple | None = None, ax: Any | None = None
    ) -> plt.Axes:
        """
        Plot Twiss parameters along the segment.

        :param incoming: Beam that is entering the segment from upstream for which the
            trajectory is computed.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        :param ax: Axes to plot into.
        :return: Axes with the plotted Twiss parameters.
        """
        ax = self.plot_beam_attrs(
            incoming,
            ("beta_x", "beta_y"),
            resolution=None,
            vector_idx=vector_idx,
            ax=ax,
        )

        beta_x_line = ax.get_lines()[0]
        beta_y_line = ax.get_lines()[1]

        beta_x_line.set_label(r"$\beta_x$")
        beta_x_line.set_color("tab:red")
        beta_y_line.set_label(r"$\beta_y$")
        beta_y_line.set_color("tab:green")

        ax.set_title("Twiss Parameters")
        ax.set_xlabel("s (m)")
        ax.set_ylabel(r"$\beta$ (m)")
        ax.legend()

        return ax

    def plot_twiss_over_lattice(
        self,
        incoming: Beam,
        vector_idx: tuple | None = None,
        fig: matplotlib.figure.Figure | None = None,
    ) -> matplotlib.figure.Figure:
        """
        Plot Twiss parameters in a plot over a plot of the lattice.

        :param incoming: Beam that is entering the segment from upstream for which the
            trajectory is computed.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        :param fig: Figure to plot into. Can be used, for example to pass a figure with
            a desired custom `figsize`. If `None`, a new figure with default size is
            created.
        :return: Figure with the plotted Twiss parameters over the lattice.
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 4))
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
        axs = gs.subplots(sharex=True)

        self.plot_twiss(incoming, vector_idx=vector_idx, ax=axs[0])
        self.plot(s=0.0, vector_idx=vector_idx, ax=axs[1])

        return fig

    def plot_floor_space(
        self,
        vector_idx: tuple | None = None,
        ax: plt.Axes | None = None,
        start_position: tuple[float, float] = (0.0, 0.0),
        start_angle: float = 0.0,
        element_width: float | dict | None = None,
        element_color: dict | None = None,
        draw_trajectory: bool = True,
        trajectory_kwargs: dict | None = None,
        show_labels: bool = False,
        show_legend: bool = False,
        style: Literal["shapes", "sprites"] = "shapes",
    ) -> plt.Axes:
        """
        Plot a 2D floor space representation of the accelerator segment.

        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        :param ax: Matplotlib axes to plot into. If `None`, a new subplot is created.
        :param start_position: Tuple of `(z, x)` coordinates in meters where the segment
            begins in the laboratory floor plane.
        :param start_angle: Initial angle of the beam trajectory in radians relative
            to the z-axis. A value of 0.0 means the beam travels in the positive
            z-direction.
        :param element_width: Transverse width of the drawn elements in meters. Can be a
            float applied to all elements, or a dictionary mapping element names, types,
            or `"*"` to custom widths. If `None`, a sensible default based on total
            segment length is used.
        :param element_color: Dictionary mapping element names, types, or `"*"` to
            matplotlib color specifications. If `None`, default element colors are used.
        :param draw_trajectory: Whether to draw the centerline trajectory of the beam.
        :param trajectory_kwargs: Keyword arguments passed to `ax.plot` for the
            trajectory line.
        :param show_labels: Whether to annotate elements with their names.
        :param show_legend: Whether to display a legend indicating element types.
        :param style: Rendering style to use. Currently only `"shapes"` is supported.
        :return: Matplotlib axes containing the floor space plot.
        """
        if style == "sprites":
            raise NotImplementedError(
                "Sprite-based rendering with pre-rendered assets is not yet "
                "implemented. Please use style='shapes'."
            )
        elif style != "shapes":
            raise ValueError(
                f"Unknown style '{style}'. Supported styles are: 'shapes'."
            )

        if ax is None:
            _, ax = plt.subplots()

        flattened_elements = self.flattened().elements
        if not flattened_elements:
            return ax

        def _extract_scalar(val: Any) -> float:
            if val is None:
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    return val.item()
                if vector_idx is not None:
                    try:
                        return val[vector_idx].item()
                    except (IndexError, TypeError):
                        pass
                return val.flatten()[0].item()
            return float(val)

        total_length = sum(
            _extract_scalar(e.length)
            for e in flattened_elements
            if hasattr(e, "length")
        )
        default_width = max(0.3, total_length * 0.015) if total_length > 0.0 else 0.3

        def _get_element_width(element: Element) -> float:
            if isinstance(element_width, (int, float)):
                return float(element_width)
            if isinstance(element_width, dict):
                if element.name in element_width:
                    return float(element_width[element.name])
                if element.__class__ in element_width:
                    return float(element_width[element.__class__])
                if element.__class__.__name__ in element_width:
                    return float(element_width[element.__class__.__name__])
                if "*" in element_width:
                    return float(element_width["*"])
            return default_width

        def _get_element_color(element: Element) -> str:
            if element_color is not None:
                if element.name in element_color:
                    return element_color[element.name]
                if element.__class__ in element_color:
                    return element_color[element.__class__]
                if element.__class__.__name__ in element_color:
                    return element_color[element.__class__.__name__]
                if "*" in element_color:
                    return element_color["*"]
            return DEFAULT_ELEMENT_COLORS.get(element.__class__.__name__, "tab:blue")

        z = float(start_position[0])
        x = float(start_position[1])
        theta = float(start_angle)
        trajectory_points: list[tuple[float, float]] = [(z, x)]
        plotted_element_types: dict[str, str] = {}

        for element in flattened_elements:
            length = (
                _extract_scalar(element.length) if hasattr(element, "length") else 0.0
            )
            w = _get_element_width(element)
            color = _get_element_color(element)
            elem_type = element.__class__.__name__

            if isinstance(element, Drift):
                z_exit = z + length * np.cos(theta)
                x_exit = x + length * np.sin(theta)
                z, x = float(z_exit), float(x_exit)
                trajectory_points.append((z, x))
                continue

            if isinstance(element, Dipole):
                angle = (
                    _extract_scalar(element.angle) if hasattr(element, "angle") else 0.0
                )
                if abs(angle) > 1e-12:
                    plotted_element_types[elem_type] = color
                    omega = -angle / length if length > 0.0 else 0.0
                    if length > 0.0 and abs(omega) > 1e-12:
                        num_points = max(5, int(abs(angle) / 0.05))
                        s_vals = np.linspace(0.0, length, num_points)
                        theta_s = theta + omega * s_vals
                        zs = z + (np.sin(theta_s) - np.sin(theta)) / omega
                        xs = x - (np.cos(theta_s) - np.cos(theta)) / omega
                        normals = np.column_stack([-np.sin(theta_s), np.cos(theta_s)])
                        inner = np.column_stack([zs, xs]) - (w / 2.0) * normals
                        outer = np.column_stack([zs, xs]) + (w / 2.0) * normals
                        verts = np.vstack([inner, outer[::-1]])
                        patch = Polygon(
                            verts,
                            closed=True,
                            facecolor=color,
                            edgecolor="black",
                            linewidth=0.5,
                            alpha=0.85,
                            zorder=2,
                        )
                        ax.add_patch(patch)
                        for pt in zip(zs[1:], xs[1:]):
                            trajectory_points.append((float(pt[0]), float(pt[1])))
                        if show_labels and element.name:
                            mid_idx = num_points // 2
                            label_pos = (
                                np.array([zs[mid_idx], xs[mid_idx]])
                                + (w * 0.8) * normals[mid_idx]
                            )
                            rot = np.degrees(theta_s[mid_idx])
                            if rot > 90.0:
                                rot -= 180.0
                            elif rot < -90.0:
                                rot += 180.0
                            ax.text(
                                label_pos[0],
                                label_pos[1],
                                element.name,
                                fontsize=7,
                                ha="center",
                                va="center",
                                rotation=rot,
                                zorder=4,
                            )
                        z, x = float(zs[-1]), float(xs[-1])
                        theta = float(theta_s[-1])
                    else:
                        # Thin dipole kick
                        theta -= angle
                    continue

            if length > 0.0:
                plotted_element_types[elem_type] = color
                z_exit = z + length * np.cos(theta)
                x_exit = x + length * np.sin(theta)
                normal = np.array([-np.sin(theta), np.cos(theta)])
                p0 = np.array([z, x])
                p1 = np.array([z_exit, x_exit])
                verts = np.array(
                    [
                        p0 - (w / 2.0) * normal,
                        p1 - (w / 2.0) * normal,
                        p1 + (w / 2.0) * normal,
                        p0 + (w / 2.0) * normal,
                    ]
                )
                patch = Polygon(
                    verts,
                    closed=True,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.85,
                    zorder=2,
                )
                ax.add_patch(patch)
                if show_labels and element.name:
                    center = (p0 + p1) / 2.0
                    label_pos = center + (w * 0.8) * normal
                    rot = np.degrees(theta)
                    if rot > 90.0:
                        rot -= 180.0
                    elif rot < -90.0:
                        rot += 180.0
                    ax.text(
                        label_pos[0],
                        label_pos[1],
                        element.name,
                        fontsize=7,
                        ha="center",
                        va="center",
                        rotation=rot,
                        zorder=4,
                    )
                z, x = float(z_exit), float(x_exit)
                trajectory_points.append((z, x))
                continue

            # Lengthless or zero-length elements
            plotted_element_types[elem_type] = color
            normal = np.array([-np.sin(theta), np.cos(theta)])
            if isinstance(element, Screen):
                p_start = np.array([z, x]) - (w / 2.0) * normal
                p_end = np.array([z, x]) + (w / 2.0) * normal
                ax.plot(
                    [p_start[0], p_end[0]],
                    [p_start[1], p_end[1]],
                    color=color,
                    linewidth=2.0,
                    zorder=3,
                )
            elif isinstance(element, BPM):
                ax.scatter(
                    [z],
                    [x],
                    marker="D",
                    s=40.0,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
            elif isinstance(element, Aperture):
                p_center = np.array([z, x])
                # Two jaw markers on either side
                p1_outer = p_center - (w / 2.0) * normal
                p1_inner = p_center - (w * 0.2) * normal
                p2_inner = p_center + (w * 0.2) * normal
                p2_outer = p_center + (w / 2.0) * normal
                ax.plot(
                    [p1_outer[0], p1_inner[0]],
                    [p1_outer[1], p1_inner[1]],
                    color=color,
                    linewidth=2.5,
                    zorder=3,
                )
                ax.plot(
                    [p2_inner[0], p2_outer[0]],
                    [p2_inner[1], p2_outer[1]],
                    color=color,
                    linewidth=2.5,
                    zorder=3,
                )
            elif isinstance(element, Marker):
                p_start = np.array([z, x]) - (w * 0.4) * normal
                p_end = np.array([z, x]) + (w * 0.4) * normal
                ax.plot(
                    [p_start[0], p_end[0]],
                    [p_start[1], p_end[1]],
                    color=color,
                    linestyle=":",
                    linewidth=1.0,
                    zorder=3,
                )
            else:
                p_start = np.array([z, x]) - (w / 2.0) * normal
                p_end = np.array([z, x]) + (w / 2.0) * normal
                ax.plot(
                    [p_start[0], p_end[0]],
                    [p_start[1], p_end[1]],
                    color=color,
                    linewidth=1.5,
                    zorder=3,
                )

            if show_labels and element.name:
                label_pos = np.array([z, x]) + (w * 0.8) * normal
                rot = np.degrees(theta)
                if rot > 90.0:
                    rot -= 180.0
                elif rot < -90.0:
                    rot += 180.0
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    element.name,
                    fontsize=7,
                    ha="center",
                    va="center",
                    rotation=rot,
                    zorder=4,
                )

        if draw_trajectory and len(trajectory_points) > 1:
            traj_opts = {
                "color": "black",
                "linestyle": "--",
                "linewidth": 1.0,
                "zorder": 1,
            }
            if trajectory_kwargs is not None:
                traj_opts.update(trajectory_kwargs)
            tz, tx = zip(*trajectory_points)
            ax.plot(tz, tx, **traj_opts)

        if show_legend and plotted_element_types:
            handles = [
                Patch(
                    facecolor=plotted_element_types[name],
                    edgecolor="black",
                    linewidth=0.5,
                    label=name,
                )
                for name in sorted(plotted_element_types.keys())
            ]
            ax.legend(handles=handles)

        ax.set_aspect("equal")
        ax.set_xlabel("z (m)")
        ax.set_ylabel("x (m)")
        ax.autoscale_view()

        return ax

    plot_floor_plan = plot_floor_space

    def to_mesh(
        self,
        cuteness: float | dict = 1.0,
        asset_version: str = "v1.2.0",
        show_download_progress: bool = True,
    ) -> "tuple[trimesh.Scene | None, np.ndarray]":  # noqa: F821 # type: ignore
        # Import only here because most people will not need it
        import trimesh

        scene = trimesh.Scene()
        input_transform = trimesh.transformations.identity_matrix()
        for element in self.elements:
            element_mesh, element_output_transform = element.to_mesh(
                cuteness=cuteness,
                asset_version=asset_version,
                show_download_progress=show_download_progress,
            )

            if element_mesh is not None:
                element_mesh.apply_transform(input_transform)
            input_transform = input_transform @ element_output_transform

            scene.add_geometry(element_mesh)

        segment_output_transform = input_transform

        return scene, segment_output_transform

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["elements"]

    def __repr__(self) -> str:
        num_elements = len(self.elements)
        if num_elements <= 5:
            elements_repr = repr(self.elements)
        else:
            element_repr_list = [
                f"({i}): {repr(self.elements[i])}"
                for i in [0, 1, num_elements - 2, num_elements - 1]
            ]
            element_repr_list.insert(2, " ⋮")

            # Using `format` since Python<=3.11 does not permit backslashes in nested
            # f-strings: https://stackoverflow.com/questions/67680296/syntaxerror-f-string-expression-part-cannot-include-a-backslash   # noqa: E501
            # TODO: Switch to f-string when Python>=3.12 is the minimum requirement
            elements_repr = "ModuleList(\n  {0}\n)".format(
                "\n  ".join(element_repr_list)
            )

        return (
            f"{self.__class__.__name__}(elements={elements_repr}, "
            + f"name={repr(self.name)})"
        )
