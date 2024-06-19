from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn

from cheetah.particles import Beam, ParameterBeam, ParticleBeam
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Element(ABC, nn.Module):
    """
    Base class for elements of particle accelerators.

    :param name: Unique identifier of the element.
    """

    length: torch.Tensor = torch.zeros((1))

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()

        self.name = name if name is not None else generate_unique_name()

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        r"""
        Generates the element's transfer map that describes how the beam and its
        particles are transformed when traveling through the element.
        The state vector consists of 6 values with a physical meaning:
        (in the trace space notation)

        - x: Position in x direction
        - xp: Angle in x direction
        - y: Position in y direction
        - yp: Angle in y direction
        - s: Position in longitudinal direction, the zero value is set to the
        reference position (usually the center of the pulse)
        - p: Relative energy deviation from the reference particle
           :math:`p = \frac{\Delta E}{p_0 C}`
        As well as a seventh value used to add constants to some of the prior values if
        necessary. Through this seventh state, the addition of constants can be
        represented using a matrix multiplication.

        :param energy: Reference energy of the Beam. Read from the fed-in Cheetah Beam.
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
        if incoming is Beam.empty:
            return incoming
        elif isinstance(incoming, ParameterBeam):
            tm = self.transfer_map(incoming.energy)
            mu = torch.matmul(tm, incoming._mu.unsqueeze(-1)).squeeze(-1)
            cov = torch.matmul(tm, torch.matmul(incoming._cov, tm.transpose(-2, -1)))
            return ParameterBeam(
                mu,
                cov,
                incoming.energy,
                total_charge=incoming.total_charge,
                device=mu.device,
                dtype=mu.dtype,
            )
        elif isinstance(incoming, ParticleBeam):
            tm = self.transfer_map(incoming.energy)
            new_particles = torch.matmul(incoming.particles, tm.transpose(-2, -1))
            return ParticleBeam(
                new_particles,
                incoming.energy,
                particle_charges=incoming.particle_charges,
                device=new_particles.device,
                dtype=new_particles.dtype,
            )
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

    def forward(self, incoming: Beam) -> Beam:
        """Forward function required by `torch.nn.Module`. Simply calls `track`."""
        return self.track(incoming)

    def broadcast(self, shape: torch.Size) -> "Element":
        """Broadcast the element to higher batch dimensions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_skippable(self) -> bool:
        """
        Whether the element can be skipped during tracking. If `True`, the element's
        transfer map is combined with the transfer maps of surrounding skipable
        elements.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def defining_features(self) -> list[str]:
        """
        List of features that define the element. Used to compare elements for equality
        and to save them.

        NOTE: When overriding this property, make sure to call the super method and
        extend the list it returns.
        """
        return []

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
    def plot(self, ax: plt.Axes, s: float) -> None:
        """
        Plot a representation of this element into a `matplotlib` Axes at position `s`.

        :param ax: Axes to plot the representation into.
        :param s: Position of the object along s in meters.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={repr(self.name)})"
