import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from torch import nn

from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.utils import DirtyNameWarning, NoVisualizationWarning, UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Element(ABC, nn.Module):
    """
    Base class for elements of particle accelerators.

    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python
        variable name. This is needed if you want to use the `segment.element_name`
        syntax to access the element in a segment.
    """

    def __init__(
        self,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.name = name if name is not None else generate_unique_name()
        if not self.is_name_sanitized():
            if sanitize_name:
                self.sanitize_name()
            else:
                warnings.warn(
                    f"Dirty element name {self.name} is not a valid Python variable "
                    "name. You will not be able to use the `segment.element_name` "
                    "syntax to access this element. Set `sanitize_name=True` to change "
                    "the name to a valid one, if you want to use this syntax.",
                    category=DirtyNameWarning,
                    stacklevel=2,
                )

        self.register_buffer("length", torch.tensor(0.0, device=device, dtype=dtype))

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        r"""
        Generates the element's transfer map that describes how the beam and its
        particles are transformed when traveling through the element.
        The state vector consists of 6 values with a physical meaning.
        They represent a particle in the phase space with

        - x: Position in x direction (m) relative to the reference particle
        - px: Horinzontal momentum normalized over the reference momentum
            (dimensionless) :math:`px = P_x / P_0`
        - y: Position in y direction (m) relative to the reference particle
        - py: Vertical momentum normalized over the reference momentum
            (dimensionless) :math:`py = P_y / P_0`
        - tau: Position in longitudinal direction (m) with the zero value set to the
        reference position (usually the center of the pulse)
        - p: Relative energy deviation from the reference particle (dimensionless)
        :math:`p = \frac{\Delta E}{p_0 C}`

        As well as a seventh value used to add constants to some of the previous values
        if necessary. Through this seventh state, the addition of constants can be
        represented using a matrix multiplication, i.e. the augmented matrix as in an
        affine transformation.

        :param energy: Reference energy of the beam. Read from the fed-in Cheetah beam.
        :param species: Species of the particles in the beam
        :return: A 7x7 Matrix for further calculations.
        """
        raise NotImplementedError

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the element. The input can be a `ParameterBeam` or a
        `ParticleBeam`.

        :param incoming: Beam of particles entering the element.
        :return: Beam of particles exiting the element.
        """
        if isinstance(incoming, ParameterBeam):
            tm = self.transfer_map(incoming.energy, incoming.species)
            new_mu = (tm @ incoming.mu.unsqueeze(-1)).squeeze(-1)
            new_cov = tm @ incoming.cov @ tm.transpose(-2, -1)
            new_s = incoming.s + self.length
            return ParameterBeam(
                new_mu,
                new_cov,
                incoming.energy,
                total_charge=incoming.total_charge,
                s=new_s,
                species=incoming.species.clone(),
            )
        elif isinstance(incoming, ParticleBeam):
            tm = self.transfer_map(incoming.energy, incoming.species)
            new_particles = incoming.particles @ tm.transpose(-2, -1)
            new_s = incoming.s + self.length
            return ParticleBeam(
                new_particles,
                incoming.energy,
                particle_charges=incoming.particle_charges,
                survival_probabilities=incoming.survival_probabilities,
                s=new_s,
                species=incoming.species.clone(),
            )
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

    def forward(self, incoming: Beam) -> Beam:
        """Forward function required by `torch.nn.Module`. Simply calls `track`."""
        return self.track(incoming)

    @property
    @abstractmethod
    def is_skippable(self) -> bool:
        """
        Whether the element can be skipped during tracking. If `True`, the element's
        transfer map is combined with the transfer maps of surrounding skipable
        elements.
        """
        raise NotImplementedError

    def register_buffer_or_parameter(
        self, name: str, value: torch.Tensor | nn.Parameter
    ) -> None:
        """
        Register a buffer or parameter with the given name and value. Automatically
        selects the correct method from `register_buffer` or `register_parameter` based
        on the type of `value`.

        :param name: Name of the buffer or parameter.
        :param value: Value of the buffer or parameter.
        :param default: Default value of the buffer.
        """
        if isinstance(value, nn.Parameter):
            self.register_parameter(name, value)
        else:
            self.register_buffer(name, value)

    @property
    @abstractmethod
    def defining_features(self) -> list[str]:
        """
        List of features that define the element. Used to compare elements for equality
        and to save them.

        NOTE: When overriding this property, make sure to call the super method and
        extend the list it returns.
        """
        return ["name"]

    @property
    def defining_tensors(self) -> list[str]:
        """Subset of defining features that are of type `torch.Tensor`."""
        return [
            feature
            for feature in self.defining_features
            if isinstance(getattr(self, feature), torch.Tensor)
        ]

    def clone(self) -> "Element":
        """Create a copy of the element which does not share the underlying memory."""
        return self.__class__(
            **{
                feature: (
                    getattr(self, feature).clone()
                    if isinstance(getattr(self, feature), torch.Tensor)
                    else deepcopy(getattr(self, feature))
                )
                for feature in self.defining_features
            }
        )

    def split(self, resolution: torch.Tensor) -> list["Element"]:
        """
        Split the element into slices no longer than `resolution`. Some elements may not
        be splittable, in which case a list containing only the element itself is
        returned.

        :param resolution: Length of the longest allowed split in meters.
        :return: Ordered sequence of sliced elements.
        """
        return [self]

    def is_name_sanitized(self) -> bool:
        """
        Check if a name is sanitised, i.e. it contains only alphanumeric characters and
        underscores.

        A clean name can be used as a Python variable name, which is a requirement
        when using the `segment.element_name` syntax to access the element in a segment.
        """
        return all(c.isalnum() or c == "_" for c in self.name)

    def sanitize_name(self) -> None:
        """
        Sanitise the element's name to be a valid Python variable name.

        Replaces characters that are not alphanumeric or underscores with underscores.
        """
        self.name = "".join(
            c if c.isalnum() or c == "_" else "_" for c in self.name
        ).strip("_")

    @abstractmethod
    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        """
        Plot a representation of this element into a `matplotlib` Axes at position `s`.

        :param s: Position of the object along s in meters.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        :param ax: Axes to plot the representation into.
        """
        raise NotImplementedError

    def to_mesh(
        self, cuteness: float | dict = 1.0, show_download_progress: bool = True
    ) -> "tuple[trimesh.Trimesh | None, np.ndarray]":  # noqa: F821 # type: ignore
        """
        Return a 3D mesh representation of the element at position `s`.

        :param cuteness: Scaling factor for the mesh. This can be used to adjust the
            size of the mesh for better visualisation. A value of 1.0 means no
            scaling, while values less than 1.0 will make the mesh smaller and values
            greater than 1.0 will make it larger. May be float applied to all elements,
            or a dictionary mapping element names and types to their respective
            scaling factors. Names have precedence over types. The `"*"` key can be used
            to specify a default scaling factor.
        :param show_download_progress: If `True`, show a progress bar during the
            download of the mesh if it is not cached.
        :return: Tuple of a 3D mesh representation of the element, oriented with the
            beam's inbound point in the origin and the s-axis pointing along the
            longitudinal direction of the element, and the transformation matrix that
            would have to be applied to a downstream mesh to align it with this mesh's
            output.
        """
        # Import only here because most people will not need it
        import trimesh

        from cheetah.utils import cache

        snake_case_class_name = "".join(
            "_" + c.lower() if c.isupper() else c for c in self.__class__.__name__
        ).lstrip("_")
        mesh = cache.load_3d_asset(
            f"{snake_case_class_name}.glb",
            show_download_progress=show_download_progress,
        )

        if mesh is None:
            warnings.warn(
                f"Could not load 3D mesh for element {self.name} of type "
                f"{self.__class__.__name__}. The element will not be visualised.",
                category=NoVisualizationWarning,
                stacklevel=2,
            )
            output_transform = trimesh.transformations.translation_matrix(
                [0.0, 0.0, self.length.item()]
            )
            return None, output_transform

        # NOTE: Scaling must be done before translation to ensure the mesh is
        # positioned correctly after scaling.

        # Scale element to the correct length (only if the mesh has a length)
        if abs(self.length.item()) > 0.0:
            _, _, mesh_length = mesh.extents
            scale_factor_for_correct_length = self.length.item() / mesh_length
            mesh.apply_scale(scale_factor_for_correct_length)

        # Apply scaling to make the mesh look cuter
        scale_factor_for_cuteness = 1.0
        if isinstance(cuteness, float):
            scale_factor_for_cuteness = cuteness
        elif isinstance(cuteness, dict):
            # Use the name of the element to find the correct scaling factor
            if self.name in cuteness:
                scale_factor_for_cuteness = cuteness[self.name]
            elif self.__class__ in cuteness:
                scale_factor_for_cuteness = cuteness[self.__class__]
            elif "*" in cuteness:
                scale_factor_for_cuteness = cuteness["*"]
        mesh.apply_scale(scale_factor_for_cuteness)

        # Compute transformation matrix needed for next mesh to align to output
        output_transform = trimesh.transformations.translation_matrix(
            [0.0, 0.0, self.length.item()]
        )

        return mesh, output_transform

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={repr(self.name)})"
