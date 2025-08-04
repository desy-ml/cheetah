from functools import reduce
from pathlib import Path
from typing import Any, Iterator

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

from cheetah.accelerator.custom_transfer_map import CustomTransferMap
from cheetah.accelerator.drift import Drift
from cheetah.accelerator.element import Element
from cheetah.accelerator.marker import Marker
from cheetah.converters import bmad, elegant, nxtables
from cheetah.latticejson import load_cheetah_model, save_cheetah_model
from cheetah.particles import Beam, Species
from cheetah.utils import UniqueNameGenerator, squash_index_for_unavailable_dims

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Segment(Element):
    """
    Segment of a particle accelerator consisting of several elements.

    :param cell: List of Cheetah elements that describe an accelerator (section).
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python
        variable name. This is needed if you want to use the `segment.element_name`
        syntax to access the element in a segment.
    """

    def __init__(
        self,
        elements: list[Element],
        name: str | None = None,
        sanitize_name: bool = False,
    ) -> None:
        super().__init__(name=name, sanitize_name=sanitize_name)

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

        return self.__class__(subcell)

    def flattened(self) -> "Segment":
        """
        Return a flattened version of the segment, i.e. one where all subsegments are
        resolved and their elements entered into a top-level segment.
        """
        flattened_elements = []
        for element in self.elements:
            if isinstance(element, Segment):
                flattened_elements += element.flattened().elements
            else:
                flattened_elements.append(element)

        return Segment(elements=flattened_elements, name=self.name)

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

        return Segment(
            elements=reversed_elements,
            name=f"{self.name}_reversed",
            sanitize_name=self.sanitize_name,
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

        return Segment(elements=merged_elements, name=self.name)

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

        return Segment(
            elements=[
                element
                for element in self.elements
                if not isinstance(element, Marker) or element.name in except_for
            ],
            name=self.name,
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

        return Segment(
            elements=[
                element
                for element in self.elements
                if torch.any(element.length > 0.0)
                or (hasattr(element, "is_active") and element.is_active)
                or element.name in except_for
            ],
            name=self.name,
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

        return Segment(
            elements=[
                (
                    element
                    if (hasattr(element, "is_active") and element.is_active)
                    or torch.all(element.length == 0.0)
                    or element.name in except_for
                    else Drift(
                        element.length,
                        name=element.name,
                        device=element.length.device,
                        dtype=element.length.dtype,
                    )
                )
                for element in self.elements
            ],
            name=self.name,
        )

    @classmethod
    def from_lattice_json(
        cls,
        filepath: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Segment":
        """
        Load a Cheetah model from a JSON file.

        :param filepath: Path of the file to load the lattice from.
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Loaded Cheetah `Segment`.
        """
        return load_cheetah_model(filepath, device=device, dtype=dtype)

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
        save_cheetah_model(self, filepath, title, info)

    @classmethod
    def from_ocelot(
        cls,
        cell,
        name: str | None = None,
        sanitize_names: bool = False,
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
            `segment.element_name` syntax to access the element in a segment.
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Cheetah segment closely resembling the Ocelot cell.
        """
        from cheetah.converters import ocelot

        converted = [
            ocelot.convert_element_to_cheetah(
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
        sanitize_names: bool = False,
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
            `segment.element_name` syntax to access the element in a segment.
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Cheetah `Segment` representing the Bmad lattice.
        """
        bmad_lattice_file_path = Path(bmad_lattice_file_path)
        return bmad.convert_lattice_to_cheetah(
            bmad_lattice_file_path, environment_variables, sanitize_names, device, dtype
        )

    @classmethod
    def from_elegant(
        cls,
        elegant_lattice_file_path: str,
        name: str,
        sanitize_names: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Segment":
        """
        Read a Cheetah segment from an elegant lattice file.

        :param bmad_lattice_file_path: Path to the Bmad lattice file.
        :param name: Name of the root element
        :param sanitize_names: Whether to sanitise the names of the elements to be valid
            Python variable names. This is needed if you want to use the
            `segment.element_name` syntax to access the element in a segment.
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Cheetah `Segment` representing the elegant lattice.
        """

        elegant_lattice_file_path = Path(elegant_lattice_file_path)
        return elegant.convert_lattice_to_cheetah(
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

        return nxtables.convert_lattice_to_cheetah(filepath)

    @property
    def is_skippable(self) -> bool:
        return all(element.is_skippable for element in self.elements)

    @property
    def length(self) -> torch.Tensor:
        lengths = [element.length for element in self.elements]
        return reduce(torch.add, lengths)

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        if self.is_skippable:
            tm = torch.eye(7, device=energy.device, dtype=energy.dtype)
            for element in self.elements:
                tm = element.transfer_map(energy, species) @ tm
            return tm
        else:
            return None

    def track(self, incoming: Beam) -> Beam:
        if self.is_skippable:
            return super().track(incoming)
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

    def clone(self) -> "Segment":
        return Segment(
            elements=[element.clone() for element in self.elements], name=self.name
        )

    def split(self, resolution: torch.Tensor) -> list[Element]:
        return [
            split_element
            for element in self.elements
            for split_element in element.split(resolution)
        ]

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
                dim=-(incoming.UNVECTORIZED_NUM_ATTR_DIMS[attr_name] + 1),
            )
            for attr_tensor, attr_name in zip(results, attr_name_tuple)
        )

        return (
            broadcasted_results
            if isinstance(attr_names, tuple)
            else broadcasted_results[0]
        )

    def set_attrs_on_every_element_of_type(
        self,
        element_type: type[Element] | tuple[type[Element]],
        is_recursive: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Set attributes on every element of a specific type in the segment.

        :param element_type: Type of the elements to set the attributes for.
        :param is_recursive: If `True`, the this method is applied to nested `Segment`s
            as well. If `False`, only the elements directly in the top-level `Segment`
            are considered.
        :param kwargs: Attributes to set and their values.
        """
        for element in self.elements:
            if isinstance(element, element_type):
                for key, value in kwargs.items():
                    setattr(element, key, value)
            elif is_recursive and isinstance(element, Segment):
                element.set_attrs_on_every_element_of_type(
                    element_type, is_recursive=True, **kwargs
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
        axx: plt.Axes,
        axy: plt.Axes,
        incoming: Beam,
        resolution: float | None = None,
        vector_idx: tuple | None = None,
    ) -> None:
        """
        Plot the mean (i.e. beam position) and standard deviation (i.e. beam size) of
        the beam along the segment view in x- and y-direction.

        :param axx: Axes to plot the particle traces into viewed in x-direction.
        :param axy: Axes to plot the particle traces into viewed in y-direction.
        :param incoming: Entering beam for which the position and size are shown
        :param resolution: Minimum resolution of the tracking of the beam position and
            beam size in the plot.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        """
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
        axx.set_xlabel("s (m)")
        axy.set_ylabel("y (m)")

    def plot_overview(
        self,
        incoming: Beam,
        fig: matplotlib.figure.Figure | None = None,
        resolution: float | None = None,
        vector_idx: tuple | None = None,
    ) -> None:
        """
        Plot an overview of the segment with the lattice along with the beam position
        and size.

        :param incoming: Entering beam for which the position and size are shown.
        :param fig: Figure to plot the overview into.
        :param resolution: Minimum resolution of the tracking of the beam position and
            beam size in the plot.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
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

        self.plot(ax=axs[2], s=0.0, vector_idx=vector_idx)

        plt.tight_layout()

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
        figsize=(8, 4),
        resolution: float | None = None,
        vector_idx: tuple | None = None,
    ) -> None:
        """
        Plot beam attributes in a plot over a plot of the lattice.

        :param incoming: Beam that is entering the segment from upstream for which the
            trajectory is computed.
        :param attr_names: Metrics to compute. Can be a single metric or a tuple of
            metrics. Supported metrics are any property of beam class of `incoming`.
        :param figsize: Size of the figure.
        :param resolution: Minimum resolution of the tracking of the beam position and
            beam size in the plot.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
        axs = gs.subplots(sharex=True)

        self.plot_beam_attrs(
            incoming=incoming,
            attr_names=attr_names,
            resolution=resolution,
            vector_idx=vector_idx,
            ax=axs[0],
        )
        self.plot(s=0.0, ax=axs[1])

        plt.tight_layout()

    def plot_twiss(
        self, incoming: Beam, vector_idx: tuple | None = None, ax: Any | None = None
    ) -> plt.Axes:
        """Plot twiss parameters along the segment."""
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

    def plot_twiss_over_lattice(self, incoming: Beam, figsize=(8, 4)) -> None:
        """Plot twiss parameters in a plot over a plot of the lattice."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
        axs = gs.subplots(sharex=True)

        self.plot_twiss(incoming, ax=axs[0])
        self.plot(s=0.0, ax=axs[1])

        plt.tight_layout()

    def to_mesh(
        self, cuteness: float | dict = 1.0, show_download_progress: bool = True
    ) -> "tuple[trimesh.Trimesh | None, np.ndarray]":  # noqa: F821 # type: ignore
        # Import only here because most people will not need it
        import trimesh

        meshes = []
        input_transform = trimesh.transformations.identity_matrix()
        for element in self.elements:
            element_mesh, element_output_transform = element.to_mesh(
                cuteness=cuteness, show_download_progress=show_download_progress
            )

            if element_mesh is not None:
                element_mesh.apply_transform(input_transform)
            input_transform = input_transform @ element_output_transform

            meshes.append(element_mesh)

        # Using `trimesh.util.concatenate` rather than adding to `Scene` to preserve
        # materials. Otherwise you might find that everything becomes glossy. (But
        # doesn't always work.)
        segment_mesh = trimesh.util.concatenate(meshes)
        segment_output_transform = input_transform

        return segment_mesh, segment_output_transform

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["elements"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(elements={repr(self.elements)}, "
            + f"name={repr(self.name)})"
        )
