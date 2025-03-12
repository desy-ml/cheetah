from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from torch import nn

from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Element(ABC, nn.Module):
    """
    Base class for elements of particle accelerators.

    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        name: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.name = name if name is not None else generate_unique_name()
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
            mu = torch.matmul(tm, incoming.mu.unsqueeze(-1)).squeeze(-1)
            cov = torch.matmul(tm, torch.matmul(incoming.cov, tm.transpose(-2, -1)))
            return ParameterBeam(
                mu,
                cov,
                incoming.energy,
                total_charge=incoming.total_charge,
                species=incoming.species.clone(),
            )
        elif isinstance(incoming, ParticleBeam):
            tm = self.transfer_map(incoming.energy, incoming.species)
            new_particles = torch.matmul(incoming.particles, tm.transpose(-2, -1))
            return ParticleBeam(
                new_particles,
                incoming.energy,
                particle_charges=incoming.particle_charges,
                survival_probabilities=incoming.survival_probabilities,
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

    @abstractmethod
    def split(self, resolution: torch.Tensor) -> list["Element"]:
        """
        Split the element into slices no longer than `resolution`. Some elements may not
        be splittable, in which case a list containing only the element itself is
        returned.

        :param resolution: Length of the longest allowed split in meters.
        :return: Ordered sequence of sliced elements.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self, ax: plt.Axes, s: float, vector_idx: tuple | None = None) -> None:
        """
        Plot a representation of this element into a `matplotlib` Axes at position `s`.

        :param ax: Axes to plot the representation into.
        :param s: Position of the object along s in meters.
        :param vector_idx: Index of the vector dimension to plot. If the model has more
            than one vector dimension, this can be used to select a specific one. In the
            case of present vector dimension but no index provided, the first one is
            used by default.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={repr(self.name)})"
