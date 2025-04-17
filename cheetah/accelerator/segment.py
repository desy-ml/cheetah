from functools import reduce
from pathlib import Path
from typing import Any

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
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Segment(Element):
    """
    Segment of a particle accelerator consisting of several elements.

    :param cell: List of Cheetah elements that describe an accelerator (section).
    :param name: Unique identifier of the element.
    """

    def __init__(self, elements: list[Element], name: str | None = None) -> None:
        super().__init__(name=name)

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
        warnings: bool = True,
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
        :param warnings: Whether to print warnings when objects are not supported by
            Cheetah or converted with potentially unexpected behavior.
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Cheetah segment closely resembling the Ocelot cell.
        """
        from cheetah.converters import ocelot

        converted = [
            ocelot.convert_element_to_cheetah(
                element, warnings=warnings, device=device, dtype=dtype
            )
            for element in cell
        ]
        return cls(converted, name=name, **kwargs)

    @classmethod
    def from_bmad(
        cls,
        bmad_lattice_file_path: str,
        environment_variables: dict | None = None,
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
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Cheetah `Segment` representing the Bmad lattice.
        """
        bmad_lattice_file_path = Path(bmad_lattice_file_path)
        return bmad.convert_lattice_to_cheetah(
            bmad_lattice_file_path, environment_variables, device, dtype
        )

    @classmethod
    def from_elegant(
        cls,
        elegant_lattice_file_path: str,
        name: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Segment":
        """
        Read a Cheetah segment from an elegant lattice file.

        :param bmad_lattice_file_path: Path to the Bmad lattice file.
        :param name: Name of the root element
        :param device: Device to place the lattice elements on.
        :param dtype: Data type to use for the lattice elements.
        :return: Cheetah `Segment` representing the elegant lattice.
        """

        elegant_lattice_file_path = Path(elegant_lattice_file_path)
        return elegant.convert_lattice_to_cheetah(
            elegant_lattice_file_path, name, device, dtype
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
                tm = torch.matmul(element.transfer_map(energy, species), tm)
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

    def plot(self, ax: plt.Axes, s: float, vector_idx: tuple | None = None) -> None:
        element_lengths = [element.length for element in self.elements]
        element_ss = [torch.tensor(0.0)] + [
            sum(element_lengths[: i + 1]) for i, _ in enumerate(element_lengths)
        ]
        element_ss = [s + element_s for element_s in element_ss]
        broadcast_ss = torch.broadcast_tensors(*element_ss)
        stacked_ss = torch.stack(broadcast_ss)
        dimension_reordered_ss = stacked_ss.movedim(0, -1)  # Place vector dims first

        plot_ss = (
            dimension_reordered_ss[vector_idx]
            if stacked_ss.dim() > 1
            else dimension_reordered_ss
        ).detach()

        ax.plot([0, plot_ss[-1]], [0, 0], "--", color="black")

        for element, s in zip(self.elements, plot_ss[:-1]):
            element.plot(ax, s, vector_idx)

        ax.set_ylim(-1, 1)
        ax.set_xlabel("s (m)")
        ax.set_yticks([])

    def plot_mean_and_std(
        self,
        axx: plt.Axes,
        axy: plt.Axes,
        incoming: Beam,
        resolution: float = 0.01,
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
        reference_segment = self.clone()
        splits = reference_segment.split(resolution=torch.tensor(resolution))

        split_lengths = [split.length for split in splits]
        ss = [torch.tensor(0.0)] + [
            sum(split_lengths[: i + 1]) for i, _ in enumerate(split_lengths)
        ]
        broadcast_ss = torch.broadcast_tensors(*ss)
        stacked_ss = torch.stack(broadcast_ss)
        dimension_reordered_ss = stacked_ss.movedim(0, -1)  # Place vector dims first

        references = [incoming]
        for split in splits:
            sample = split(references[-1])
            references.append(sample)

        x_means = [reference_beam.mu_x for reference_beam in references]
        broadcast_x_means = torch.broadcast_tensors(*x_means)
        stacked_x_means = torch.stack(broadcast_x_means)
        dimension_reordered_x_means = stacked_x_means.movedim(
            0, -1
        )  # Place vector dims first
        x_stds = [reference_beam.sigma_x for reference_beam in references]
        broadcast_x_stds = torch.broadcast_tensors(*x_stds)
        stacked_x_stds = torch.stack(broadcast_x_stds)
        dimension_reordered_x_stds = stacked_x_stds.movedim(
            0, -1
        )  # Place vector dims first

        y_means = [reference_beam.mu_y for reference_beam in references]
        broadcast_y_means = torch.broadcast_tensors(*y_means)
        stacked_y_means = torch.stack(broadcast_y_means)
        dimension_reordered_y_means = stacked_y_means.movedim(
            0, -1
        )  # Place vector dims first
        y_stds = [reference_beam.sigma_y for reference_beam in references]
        broadcast_y_stds = torch.broadcast_tensors(*y_stds)
        stacked_y_stds = torch.stack(broadcast_y_stds)
        dimension_reordered_y_stds = stacked_y_stds.movedim(
            0, -1
        )  # Place vector dims first

        plot_ss = (
            dimension_reordered_ss[vector_idx]
            if stacked_ss.dim() > 1
            else dimension_reordered_ss
        ).detach()
        plot_x_means = (
            dimension_reordered_x_means[vector_idx]
            if stacked_x_means.dim() > 2
            else dimension_reordered_x_means
        ).detach()
        plot_x_stds = (
            dimension_reordered_x_stds[vector_idx]
            if stacked_x_stds.dim() > 2
            else dimension_reordered_x_stds
        ).detach()
        plot_y_means = (
            dimension_reordered_y_means[vector_idx]
            if stacked_y_means.dim() > 2
            else dimension_reordered_y_means
        ).detach()
        plot_y_stds = (
            dimension_reordered_y_stds[vector_idx]
            if stacked_y_stds.dim() > 2
            else dimension_reordered_y_stds
        ).detach()

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
        resolution: float = 0.01,
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

    def plot_twiss(
        self, incoming: Beam, ax: Any | None = None, vector_idx: tuple | None = None
    ) -> None:
        """Plot twiss parameters along the segment."""
        longitudinal_beams = [incoming]
        s_positions = [torch.tensor(0.0)]
        for element in self.flattened().elements:
            if torch.all(element.length == 0):
                continue

            outgoing = element.track(longitudinal_beams[-1])

            longitudinal_beams.append(outgoing)
            s_positions.append(s_positions[-1] + element.length)

        beta_x = [beam.beta_x for beam in longitudinal_beams]
        beta_y = [beam.beta_y for beam in longitudinal_beams]

        # Extraction of the correct vector element for plotting
        broadcast_s_positions = torch.broadcast_tensors(*s_positions)
        stacked_s_positions = torch.stack(broadcast_s_positions)
        dimension_reordered_s_positions = stacked_s_positions.movedim(0, -1)
        plot_s_positions = (
            dimension_reordered_s_positions[vector_idx]
            if stacked_s_positions.dim() > 1
            else dimension_reordered_s_positions
        ).detach()

        broadcast_beta_x = torch.broadcast_tensors(*beta_x)
        stacked_beta_x = torch.stack(broadcast_beta_x)
        dimension_reordered_beta_x = stacked_beta_x.movedim(0, -1)
        plot_beta_x = (
            dimension_reordered_beta_x[vector_idx]
            if stacked_beta_x.dim() > 2
            else dimension_reordered_beta_x
        ).detach()

        broadcast_beta_y = torch.broadcast_tensors(*beta_y)
        stacked_beta_y = torch.stack(broadcast_beta_y)
        dimension_reordered_beta_y = stacked_beta_y.movedim(0, -1)
        plot_beta_y = (
            dimension_reordered_beta_y[vector_idx]
            if stacked_beta_y.dim() > 2
            else dimension_reordered_beta_y
        ).detach()

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.set_title("Twiss Parameters")
        ax.set_xlabel("s (m)")
        ax.set_ylabel(r"$\beta$ (m)")

        ax.plot(plot_s_positions, plot_beta_x, label=r"$\beta_x$", c="tab:red")
        ax.plot(plot_s_positions, plot_beta_y, label=r"$\beta_y$", c="tab:green")

        ax.legend()
        plt.tight_layout()

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["elements"]

    def plot_twiss_over_lattice(self, incoming: Beam, figsize=(8, 4)) -> None:
        """Plot twiss parameters in a plot over a plot of the lattice."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
        axs = gs.subplots(sharex=True)

        self.plot_twiss(incoming, ax=axs[0])
        self.plot(axs[1], 0)

        plt.tight_layout()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(elements={repr(self.elements)}, "
            + f"name={repr(self.name)})"
        )
