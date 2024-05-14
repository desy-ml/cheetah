from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.constants import physical_constants
from torch import Size, nn
from torch.distributions import MultivariateNormal

from cheetah.converters.dontbmad import convert_bmad_lattice
from cheetah.converters.nxtables import read_nx_tables
from cheetah.latticejson import load_cheetah_model, save_cheetah_model
from cheetah.particles import Beam, ParameterBeam, ParticleBeam
from cheetah.track_methods import base_rmatrix, misalignment_matrix, rotation_matrix
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = torch.tensor(
    constants.electron_mass
    * constants.speed_of_light**2
    / constants.elementary_charge  # electron mass
)
electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class BPM(Element):
    """
    Beam Position Monitor (BPM) in a particle accelerator.

    :param is_active: If `True` the BPM is active and will record the beam's position.
        If `False` the BPM is inactive and will not record the beam's position.
    :param name: Unique identifier of the element.
    """

    def __init__(self, is_active: bool = False, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.is_active = is_active
        self.reading = None

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        return torch.eye(7, device=energy.device, dtype=energy.dtype).repeat(
            (*energy.shape, 1, 1)
        )

    def track(self, incoming: Beam) -> Beam:
        if incoming is Beam.empty:
            self.reading = None
        elif isinstance(incoming, ParameterBeam):
            self.reading = torch.stack([incoming.mu_x, incoming.mu_y])
        elif isinstance(incoming, ParticleBeam):
            self.reading = torch.stack([incoming.mu_x, incoming.mu_y])
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

        return deepcopy(incoming)

    def broadcast(self, shape: Size) -> Element:
        new_bpm = self.__class__(is_active=self.is_active, name=self.name)
        new_bpm.length = self.length.repeat(shape)
        return new_bpm

    def split(self, resolution: torch.Tensor) -> list[Element]:
        return [self]

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        patch = Rectangle(
            (s, -0.3), 0, 0.3 * 2, color="darkkhaki", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={repr(self.name)})"
