from copy import deepcopy
from typing import Any, Iterator

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch import nn

from cheetah.accelerator.element import Element
from cheetah.accelerator.segment import Segment
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class BranchBeamMapping(dict):
    """
    Mapping of branch names to tracked `Beam` objects.

    Supports dictionary key access (e.g. `mapping["branch_a"]`), attribute access
    (e.g. `mapping.branch_a`), and integer index access (e.g. `mapping[0]`).
    """

    def __getitem__(self, key: int | str) -> Beam:
        """
        Get tracked beam by branch name or index.

        :param key: Branch name as string or branch index as integer.
        :return: Tracked `Beam` for the specified branch.
        """
        if isinstance(key, int):
            branch_names = list(self.keys())
            return super().__getitem__(branch_names[key])
        return super().__getitem__(key)

    def __getattr__(self, name: str) -> Beam:
        """
        Allow accessing branch beams as attributes.

        :param name: Branch name.
        :return: Tracked `Beam` for the specified branch.
        """
        if name in self:
            return self[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set branch beam as dictionary item."""
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete branch beam by name."""
        if name in self:
            del self[name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'."
            )


class Branch(Element):
    """
    Branch element that splits a beamline into multiple downstream beamlines.

    Allows representing forks in an accelerator lattice where beamlines branch into
    multiple alternative or simultaneous downstream beamlines (e.g. switchyards,
    diagnostic branches, or undulator lines).

    :param branches: Dictionary or sequence of beamline branches. Each branch can be a
        Cheetah `Segment` or a list of `Element`s (which will automatically be wrapped
        in a `Segment`). Alternatively, branches can be provided as keyword arguments.
    :param active_branch: Name or index of the active branch. If `None` (default), all
        branches are tracked simultaneously, returning a `BranchBeamMapping`. If set,
        only the specified branch is tracked, returning a single `Beam`.
    :param split_fractions: Optional relative split fractions/weights for the branches,
        either as a dictionary mapping branch names to fractions, or a list or
        tensor of fractions for each branch. If `None` (default), each branch
        receives a full clone of the incoming beam.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment. If `None` (default), a warning is raised for
        invalid names. Set to `True` to sanitise, or `False` to silence the warning.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.
    :param kwargs: Additional named branches passed as keyword arguments.
    """

    def __init__(
        self,
        branches: (
            dict[str, Segment | list[Element]]
            | list[Segment | list[Element]]
            | tuple[Segment | list[Element], ...]
            | None
        ) = None,
        active_branch: str | int | None = None,
        split_fractions: (
            dict[str, torch.Tensor | float]
            | list[torch.Tensor | float]
            | torch.Tensor
            | None
        ) = None,
        name: str | None = None,
        sanitize_name: bool | None = None,
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Segment | list[Element],
    ) -> None:
        super().__init__(
            name=name,
            sanitize_name=sanitize_name,
            metadata=metadata,
            device=device,
            dtype=dtype,
        )

        collected_branches: dict[str, Segment] = {}
        if branches is not None:
            if isinstance(branches, dict):
                for k, v in branches.items():
                    collected_branches[str(k)] = (
                        v if isinstance(v, Segment) else Segment(list(v))
                    )
            elif isinstance(branches, (list, tuple)):
                for i, v in enumerate(branches):
                    seg = v if isinstance(v, Segment) else Segment(list(v))
                    branch_key = (
                        seg.name
                        if seg.name is not None
                        and not seg.name.startswith("unnamed_element")
                        else f"branch_{i}"
                    )
                    if branch_key in collected_branches:
                        branch_key = f"branch_{i}"
                    collected_branches[branch_key] = seg
            else:
                raise TypeError(
                    f"Expected branches to be a dict, list, or tuple, got "
                    f"{type(branches).__name__}."
                )

        if kwargs:
            for k, v in kwargs.items():
                collected_branches[k] = (
                    v if isinstance(v, Segment) else Segment(list(v))
                )

        if not collected_branches:
            raise ValueError("At least one branch must be provided to `Branch`.")

        self.register_module("branches", nn.ModuleDict(collected_branches))
        self.active_branch = active_branch

        if split_fractions is not None:
            self._init_split_fractions(split_fractions, device=device, dtype=dtype)

        for branch_name, branch_segment in self.branches.items():
            if (
                branch_name.isidentifier()
                and branch_name not in self.__dict__
                and not hasattr(type(self), branch_name)
            ):
                self.__dict__[branch_name] = branch_segment

    def _init_split_fractions(
        self,
        split_fractions: (
            dict[str, torch.Tensor | float] | list[torch.Tensor | float] | torch.Tensor
        ),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialise and register split fractions as PyTorch tensors or parameters."""
        branch_names = self.branch_names
        fractions_dict: dict[str, torch.Tensor] = {}

        if isinstance(split_fractions, dict):
            for name in branch_names:
                if name in split_fractions:
                    val = split_fractions[name]
                    fractions_dict[name] = (
                        val
                        if isinstance(val, torch.Tensor)
                        else torch.tensor(float(val), device=device, dtype=dtype)
                    )
                else:
                    fractions_dict[name] = torch.tensor(1.0, device=device, dtype=dtype)
        elif isinstance(split_fractions, (list, tuple, torch.Tensor)):
            if len(split_fractions) != len(branch_names):
                raise ValueError(
                    f"Number of split fractions ({len(split_fractions)}) does not "
                    f"match number of branches ({len(branch_names)})."
                )
            for name, val in zip(branch_names, split_fractions):
                fractions_dict[name] = (
                    val
                    if isinstance(val, torch.Tensor)
                    else torch.tensor(float(val), device=device, dtype=dtype)
                )
        else:
            raise TypeError(
                f"Unsupported type for split_fractions: "
                f"{type(split_fractions).__name__}."
            )

        self._split_fraction_keys = list(fractions_dict.keys())
        for name, tensor_val in fractions_dict.items():
            attr_name = f"_split_frac_{name}"
            if isinstance(tensor_val, nn.Parameter):
                self.register_parameter(attr_name, tensor_val)
            else:
                self.register_buffer(attr_name, tensor_val)

    @property
    def split_fractions(self) -> dict[str, torch.Tensor] | None:
        """Split fractions for the branches."""
        if not hasattr(self, "_split_fraction_keys"):
            return None
        return {
            name: getattr(self, f"_split_frac_{name}")
            for name in self._split_fraction_keys
        }

    @split_fractions.setter
    def split_fractions(
        self,
        fractions: (
            dict[str, torch.Tensor | float]
            | list[torch.Tensor | float]
            | torch.Tensor
            | None
        ),
    ) -> None:
        if fractions is None:
            if hasattr(self, "_split_fraction_keys"):
                for name in self._split_fraction_keys:
                    attr_name = f"_split_frac_{name}"
                    if attr_name in self._buffers:
                        del self._buffers[attr_name]
                    elif attr_name in self._parameters:
                        del self._parameters[attr_name]
                del self._split_fraction_keys
        else:
            self._init_split_fractions(fractions)

    @property
    def branch_names(self) -> list[str]:
        """Ordered list of branch names in the branch element."""
        return list(self.branches.keys())

    def __getitem__(self, key: int | str) -> Segment:
        """
        Get branch by name or integer index.

        :param key: Branch name or branch index.
        :return: Corresponding branch `Segment`.
        """
        if isinstance(key, int):
            return self.branches[self.branch_names[key]]
        return self.branches[key]

    def __len__(self) -> int:
        """Number of branches in this branch element."""
        return len(self.branches)

    def __iter__(self) -> Iterator[Segment]:
        """Iterate over the branch segments."""
        return iter(self.branches.values())

    def __contains__(self, key: str) -> bool:
        """Check if branch name exists in branch element."""
        return key in self.branches

    def keys(self):
        """Branch names."""
        return self.branches.keys()

    def values(self):
        """Branch segments."""
        return self.branches.values()

    def items(self):
        """Pairs of branch names and branch segments."""
        return self.branches.items()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if (
                "branches" in self.__dict__.get("_modules", {})
                and name in self._modules["branches"]
            ):
                return self._modules["branches"][name]
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'."
            )

    def _apply_split_fraction(self, beam: Beam, branch_name: str) -> Beam:
        """
        Scale the beam charge for a specific branch if split fractions are defined.

        :param beam: Incoming beam to be scaled.
        :param branch_name: Name of the target branch.
        :return: Scaled beam.
        """
        if not hasattr(self, "_split_fraction_keys"):
            return beam.clone()

        attr_name = f"_split_frac_{branch_name}"
        if not hasattr(self, attr_name):
            return beam.clone()

        fraction = getattr(self, attr_name)

        if isinstance(beam, ParameterBeam):
            return ParameterBeam(
                mu=beam.mu.clone(),
                cov=beam.cov.clone(),
                energy=beam.energy.clone(),
                total_charge=beam.total_charge * fraction,
                s=beam.s.clone(),
                species=beam.species.clone(),
            )
        elif isinstance(beam, ParticleBeam):
            return ParticleBeam(
                particles=beam.particles.clone(),
                energy=beam.energy.clone(),
                particle_charges=beam.particle_charges * fraction,
                survival_probabilities=beam.survival_probabilities.clone(),
                s=beam.s.clone(),
                species=beam.species.clone(),
            )
        else:
            raise TypeError(f"Unsupported beam type: {type(beam).__name__}.")

    def track(self, incoming: Beam) -> Beam | BranchBeamMapping:
        """
        Track particles through the branch element.

        If `active_branch` is set, tracks exclusively through that branch and returns a
        single `Beam`. Otherwise, tracks through all branches and returns a
        `BranchBeamMapping` containing the tracked beam for each branch.

        :param incoming: Beam entering the branch element.
        :return: Tracked `Beam` if `active_branch` is set, otherwise a
            `BranchBeamMapping` of branch names to tracked beams.
        """
        if self.active_branch is not None:
            resolved_name = (
                self.branch_names[self.active_branch]
                if isinstance(self.active_branch, int)
                else str(self.active_branch)
            )
            if resolved_name not in self.branches:
                raise KeyError(
                    f"Active branch '{resolved_name}' not found in branches: "
                    f"{self.branch_names}."
                )
            branch_beam = self._apply_split_fraction(incoming, resolved_name)
            return self.branches[resolved_name].track(branch_beam)

        results = BranchBeamMapping()
        for name, branch_segment in self.branches.items():
            branch_beam = self._apply_split_fraction(incoming, name)
            results[name] = branch_segment.track(branch_beam)

        return results

    @property
    def is_skippable(self) -> bool:
        """
        Whether the element can be skipped during tracking. Branch elements cannot be
        skipped as they split the beamline into multiple paths.
        """
        return False

    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor | None:
        """
        First order transfer map is not defined for a multi-branch element.

        :param energy: Reference energy of the incoming beam.
        :param species: Species of the particles in the incoming beam.
        :return: None.
        """
        return None

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        """
        Plot a representation of this branch element into a `matplotlib` Axes at
        position `s`.

        :param s: Position of the object along s in meters.
        :param vector_idx: Index of the vector dimension to plot.
        :param ax: Matplotlib Axes to plot onto.
        :return: Matplotlib Axes with the element plotted.
        """
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if hasattr(s, "dim") and s.dim() > 0 else s
        patch = Rectangle(
            (plot_s, -0.4),
            0.0,
            0.8,
            color="purple",
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(patch)
        return ax

    def clone(self) -> "Branch":
        """
        Create a copy of the branch element which does not share underlying memory.

        :return: Cloned `Branch` element.
        """
        cloned_branches = {
            name: branch.clone() for name, branch in self.branches.items()
        }
        cloned_fractions = None
        if hasattr(self, "_split_fraction_keys"):
            cloned_fractions = {
                name: getattr(self, f"_split_frac_{name}").clone()
                for name in self._split_fraction_keys
            }

        return self.__class__(
            branches=cloned_branches,
            active_branch=self.active_branch,
            split_fractions=cloned_fractions,
            name=self.name,
            metadata=deepcopy(self.metadata),
            sanitize_name=False,
        )

    def flattened(self) -> "Branch":
        """
        Return a version of the branch element where each branch segment is flattened.

        :return: Branch with flattened sub-beamlines.
        """
        flattened_branches = {
            name: branch.flattened() for name, branch in self.branches.items()
        }
        cloned_fractions = None
        if hasattr(self, "_split_fraction_keys"):
            cloned_fractions = {
                name: getattr(self, f"_split_frac_{name}").clone()
                for name in self._split_fraction_keys
            }

        return self.__class__(
            branches=flattened_branches,
            active_branch=self.active_branch,
            split_fractions=cloned_fractions,
            name=self.name,
            metadata=deepcopy(self.metadata),
            sanitize_name=False,
        )

    @property
    def defining_features(self) -> list[str]:
        """List of features that define the element."""
        return super().defining_features + ["length"]

    def __repr__(self) -> str:
        """String representation of the Branch element."""
        branch_reprs = ", ".join(
            f"{name}={len(seg.elements)} elements"
            for name, seg in self.branches.items()
        )
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"branches={{{branch_reprs}}}, "
            f"active_branch={repr(self.active_branch)})"
        )
