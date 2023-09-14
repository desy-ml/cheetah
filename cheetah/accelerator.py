from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.constants import physical_constants
from scipy.stats import multivariate_normal

from cheetah.dontbmad import convert_bmad_lattice
from cheetah.particles import Beam, ParameterBeam, ParticleBeam
from cheetah.track_methods import base_rmatrix, misalignment_matrix, rotation_matrix

ELEMENT_COUNT = 0
REST_ENERGY = (
    constants.electron_mass
    * constants.speed_of_light**2
    / constants.elementary_charge
)  # electron mass
electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class DeviceError(Exception):
    """
    Used to create an exception, in case the device used for the beam
    and the elements are different.
    """

    def __init__(self):
        super().__init__(
            "Warning! The device used for calculating the elements is not the same, "
            "as the device used to calculate the Beam."
        )


class Element(ABC):
    """
    Base class for elements of particle accelerators.

    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(self, name: Optional[str] = None, device: str = "auto") -> None:
        global ELEMENT_COUNT
        if name is not None:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}_{ELEMENT_COUNT:06d}"
        ELEMENT_COUNT += 1

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def transfer_map(self, energy: float) -> torch.Tensor:
        """
        Generates the element's transfer map that describes how the beam and its
        particles are transformed when traveling through the element. The state vector
        consists of 6 values with a physical meaning:
        - x: Position in x direction
        - xp: Momentum in x direction
        - y: Position in y direction
        - yp: Momentum in y direction
        - s: Position in z direction, the zero value is set to the middle of the pulse
        - sp: Momentum in s direction
        As well as a seventh value used to add constants to some of the prior values if
        necessary. Through this seventh state, the addition of constants can be
        represented using a matrix multiplication.

        :param energy: Energy of the Beam. Read from the fed-in Cheetah Beam.
        :return: A 7x7 Matrix for further calculations.
        """
        raise NotImplementedError

    def __call__(self, incoming: Beam) -> Beam:
        """
        Track particles through the element. The input can be a `ParameterBeam` or a
        `ParticleBeam`.

        :param incoming: Beam of particles entering the element.
        :return: Beam of particles exiting the element.
        """
        if incoming is Beam.empty:
            return incoming
        elif isinstance(incoming, ParameterBeam):
            if self.device != incoming.device:
                raise DeviceError
            tm = self.transfer_map(incoming.energy)
            mu = torch.matmul(tm, incoming._mu)
            cov = torch.matmul(tm, torch.matmul(incoming._cov, tm.t()))
            return ParameterBeam(mu, cov, incoming.energy, device=incoming.device)
        elif isinstance(incoming, ParticleBeam):
            if self.device != incoming.device:
                raise DeviceError
            tm = self.transfer_map(incoming.energy)
            new_particles = torch.matmul(incoming.particles, tm.t())
            return ParticleBeam(new_particles, incoming.energy, device=incoming.device)
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the element. The input can be a `ParameterBeam` or a
        `ParticleBeam`.

        NOTE: The purpose of this method is to provide a nicer more readable interface
        to `__call__`. The latter coninues to work, but we encourage the use of `track`.

        :param incoming: Beam of particles entering the element.
        :return: Beam of particles exiting the element.
        """
        return self(incoming)

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
        return ["name", "device"]

    @abstractmethod
    def split(self, resolution: float) -> list["Element"]:
        """
        Split the element into slices no longer than `resolution`. Some elements may not
        be splittable, in which case a list containing only the element itself is
        returned.

        :param resolution: Length of the longest allowed split in meters.
        :return: Ordered sequence of sliced elements.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        """
        Plot a representation of this element into a `matplotlib` Axes at position `s`.

        :param ax: Axes to plot the representation into.
        :param s: Position of the object along s in meters.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self.name}", device="{self.device}")'

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Element):
            return False
        return all(
            getattr(self, feature) == getattr(other, feature)
            for feature in self.defining_features
        )


class Drift(Element):
    """
    Drift section in a particle accelerator.

    :param length: Length in meters.
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self, length: float, name: Optional[str] = None, device: str = "auto"
    ) -> None:
        super().__init__(name=name, device=device)

        self.length = length

    def transfer_map(self, energy: float) -> torch.Tensor:
        gamma = energy / REST_ENERGY
        igamma2 = 1 / gamma**2 if gamma != 0 else 0

        return torch.tensor(
            [
                [1, self.length, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, self.length, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, self.length * igamma2, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: float) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Drift(min(resolution, remaining), device=self.device)
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        pass

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length"]

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(length={self.length:.2f}, name="{self.name}",'
            f' device="{self.device}")'
        )


class Quadrupole(Element):
    """
    Quadrupole magnet in a particle accelerator.

    :param length: Length in meters.
    :param k1: Strength of the quadrupole in rad/m.
    :param misalignment: Misalignment vector of the quadrupole in x- and y-directions.
    :param tilt: Tilt angle of the quadrupole in x-y plane [rad]. pi/4 for
        skew-quadrupole.
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        length: float,
        k1: float = 0.0,
        misalignment: tuple[float, float] = (0, 0),
        tilt: float = 0.0,
        name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(name=name, device=device)

        self.length = length
        self.k1 = k1
        self.misalignment = tuple(misalignment)
        self.tilt = tilt

    def transfer_map(self, energy: float) -> torch.Tensor:
        R = base_rmatrix(
            length=self.length,
            k1=self.k1,
            hx=0,
            tilt=self.tilt,
            energy=energy,
            device=self.device,
        )

        if self.misalignment[0] == 0 and self.misalignment[1] == 0:
            return R
        else:
            R_exit, R_entry = misalignment_matrix(self.misalignment, self.device)
            R = torch.matmul(R_exit, torch.matmul(R, R_entry))
            return R

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return self.k1 != 0

    def split(self, resolution: float) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Quadrupole(
                min(resolution, remaining),
                self.k1,
                misalignment=self.misalignment,
                device=self.device,
            )
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.k1) if self.is_active else 1)
        patch = Rectangle(
            (s, 0), self.length, height, color="tab:red", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "k1", "misalignment", "tilt"]

    def __repr__(self) -> None:
        return (
            f"{self.__class__.__name__}(length={self.length:.2f}, "
            + f"k1={self.k1}, "
            + f"misalignment={self.misalignment}, "
            + f"tilt={self.tilt:.2f}, "
            + f'name="{self.name}", '
            + f'device="{self.device}")'
        )


class Dipole(Element):
    """
    Dipole magnet (by default a sector bending magnet).

    :param length: Length in meters.
    :param angle: Deflection angle in rad.
    :param e1: The angle of inclination of the entrance face [rad].
    :param e2: The angle of inclination of the exit face [rad].
    :param tilt: Tilt of the magnet in x-y plane [rad].
    :param fringe_integral: Fringe field integral (of the enterance face).
    :param fringe_integral_exit: (only set if different from `fint`) Fringe field
        integral of the exit face.
    :param gap: The magnet gap [m], NOTE in MAD and ELEGANT: HGAP = gap/2
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        e1: float = 0.0,
        e2: float = 0.0,
        tilt: float = 0.0,
        fringe_integral: float = 0.0,
        fringe_integral_exit: Optional[float] = None,
        gap: float = 0.0,
        name: Optional[str] = None,
        device: str = "auto",
    ):
        super().__init__(name=name, device=device)

        self.length = length
        self.angle = angle
        self.gap = gap
        self.tilt = tilt
        self.name = name
        self.fringe_integral = fringe_integral
        self.fringe_integral_exit = (
            fringe_integral if fringe_integral_exit is None else fringe_integral_exit
        )
        # Rectangular bend
        self.e1 = e1
        self.e2 = e2

        if self.length == 0.0:
            self.hx = 0.0
        else:
            self.hx = self.angle / self.length

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self):
        return self.angle != 0

    def transfer_map(self, energy: float) -> torch.Tensor:
        R_enter = self._transfer_map_enter(energy)
        R_exit = self._transfer_map_exit(energy)

        if self.length != 0.0:  # Bending magnet with finite length
            R = base_rmatrix(
                length=self.length,
                k1=0,
                hx=self.hx,
                tilt=0.0,
                energy=energy,
                device=self.device,
            )
        else:  # Reduce to Thin-Corrector
            R = torch.tensor(
                [
                    [1, self.length, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, self.angle],
                    [0, 0, 1, self.length, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=self.device,
            )
        # Apply fringe fields
        R = torch.matmul(R_exit, torch.matmul(R, R_enter))
        # Apply rotation for tilted magnets
        R = torch.matmul(
            rotation_matrix(-self.tilt), torch.matmul(R, rotation_matrix(self.tilt))
        )

        return R

    def _transfer_map_enter(self, energy: float) -> torch.Tensor:
        if self.fringe_integral == 0:
            return torch.eye(7, device=self.device)
        else:
            sec_e = 1.0 / np.cos(self.e1)
            phi = (
                self.fringe_integral
                * self.hx
                * self.gap
                * sec_e
                * (1 + np.sin(self.e1) ** 2)
            )
            return torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [self.hx * np.tan(self.e1), 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, -self.hx * np.tan(self.e1 - phi), 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=self.device,
            )

    def _transfer_map_exit(self, energy: float) -> torch.Tensor:
        if self.fringe_integral_exit == 0:
            return torch.eye(7, device=self.device)
        else:
            sec_e = 1.0 / np.cos(self.e2)
            phi = (
                self.fringe_integral
                * self.hx
                * self.gap
                * sec_e
                * (1 + np.sin(self.e2) ** 2)
            )
            return torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [self.hx * np.tan(self.e2), 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, -self.hx * np.tan(self.e2 - phi), 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=self.device,
            )

    def split(self, resolution: float) -> list[Element]:
        # TODO: Implement splitting for dipole properly, for now just returns the
        # element itself
        return [self]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(length={self.length:.2f}, "
            + f"angle={self.angle}, "
            + f"e1={self.e1:.2f},"
            + f"e2={self.e2:.2f},"
            + f"tilt={self.tilt:.2f},"
            + f"fringe_integral={self.fringe_integral:.2f},"
            + f"fringe_integral_exit={self.fringe_integral_exit:.2f},"
            + f"gap={self.gap:.2f},"
            + f'name="{self.name}", '
            + f'device="{self.device}")'
        )

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "angle",
            "e1",
            "e2",
            "tilt",
            "fringe_integral",
            "fringe_integral_exit",
            "gap",
        ]

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.angle) if self.is_active else 1)

        patch = Rectangle(
            (s, 0), self.length, height, color="tab:green", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)


class RBend(Dipole):
    """
    Rectangular bending magnet.

    :param length: Length in meters.
    :param angle: Deflection angle in rad.
    :param e1: The angle of inclination of the entrance face [rad].
    :param e2: The angle of inclination of the exit face [rad].
    :param tilt: Tilt of the magnet in x-y plane [rad].
    :param fringe_integral: Fringe field integral (of the enterance face).
    :param fringe_integral_exit: (only set if different from `fint`) Fringe field
        integral of the exit face.
    :param gap: The magnet gap [m], NOTE in MAD and ELEGANT: HGAP = gap/2
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        e1: float = 0.0,
        e2: float = 0.0,
        tilt: float = 0.0,
        fringe_integral: float = 0.0,
        fringe_integral_exit: Optional[float] = None,
        gap: float = 0.0,
        name: Optional[str] = None,
        device: str = "auto",
    ):
        e1 = e1 + angle / 2
        e2 = e2 + angle / 2

        super().__init__(
            length=length,
            angle=angle,
            e1=e1,
            e2=e2,
            tilt=tilt,
            fringe_integral=fringe_integral,
            fringe_integral_exit=fringe_integral_exit,
            gap=gap,
            name=name,
            device=device,
        )


class HorizontalCorrector(Element):
    """
    Horizontal corrector magnet in a particle accelerator.

    :param length: Length in meters.
    :param angle: Particle deflection angle in the horizontal plane in rad.
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(name=name, device=device)

        self.length = length
        self.angle = angle

    def transfer_map(self, energy: float) -> torch.Tensor:
        return torch.tensor(
            [
                [1, self.length, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, self.angle],
                [0, 0, 1, self.length, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return self.angle != 0

    def split(self, resolution: float) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = min(resolution, remaining)
            element = HorizontalCorrector(
                length, self.angle * length / self.length, device=self.device
            )
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.angle) if self.is_active else 1)

        patch = Rectangle(
            (s, 0), self.length, height, color="tab:blue", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "angle"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={self.length:.2f}, "
            + f"angle={self.angle}, "
            + f'name="{self.name}", '
            + f'device="{self.device}")'
        )


class VerticalCorrector(Element):
    """
    Verticle corrector magnet in a particle accelerator.

    :param length: Length in meters.
    :param angle: Particle deflection angle in the vertical plane in rad.
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        length: float,
        angle: float = 0.0,
        name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(name=name, device=device)

        self.length = length
        self.angle = angle

    def transfer_map(self, energy: float) -> torch.Tensor:
        return torch.tensor(
            [
                [1, self.length, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, self.length, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, self.angle],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return self.angle != 0

    def split(self, resolution: float) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = min(resolution, remaining)
            element = VerticalCorrector(
                length, self.angle * length / self.length, device=self.device
            )
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.angle) if self.is_active else 1)

        patch = Rectangle(
            (s, 0), self.length, height, color="tab:cyan", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "angle"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={self.length:.2f}, "
            + f"angle={self.angle}, "
            + f'name="{self.name}", '
            + f'device="{self.device}")'
        )


class Cavity(Element):
    """
    Accelerating cavity in a particle accelerator.

    :param length: Length in meters.
    :param voltage: Voltage of the cavity in volts.
    :param phase: Phase of the cavity in degrees.
    :param frequency: Frequency of the cavity in Hz.
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        length: float,
        voltage: float = 0.0,
        phase: float = 0.0,
        frequency: float = 0.0,
        name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(name=name, device=device)

        self.length = length
        self.voltage = voltage
        self.phase = phase
        self.frequency = frequency

    @property
    def is_active(self) -> bool:
        return self.voltage != 0

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def transfer_map(self, energy: float) -> torch.Tensor:
        if self.voltage > 0:
            return self._cavity_rmatrix(energy)
        else:
            return base_rmatrix(
                length=self.length,
                k1=0.0,
                hx=0.0,
                tilt=0.0,
                energy=energy,
                device=self.device,
            )

    def __call__(self, incoming: Beam) -> Beam:
        """
        Track particles through the cavity. The input can be a `ParameterBeam` or a
        `ParticleBeam`. For a cavity, this does a little more than just the transfer map
        multiplication done by most elements.

        :param incoming: Beam of particles entering the element.
        :return: Beam of particles exiting the element.
        """
        if incoming is Beam.empty:
            return incoming
        elif isinstance(incoming, (ParameterBeam, ParticleBeam)):
            return self._track_beam(incoming)
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

    def _track_beam(self, incoming: ParticleBeam) -> ParticleBeam:
        beta0 = 1
        igamma2 = 0
        g0 = 1e10
        if incoming.energy != 0:
            g0 = incoming.energy / electron_mass_eV
            igamma2 = 1 / g0**2
            beta0 = torch.sqrt(1 - igamma2)

        phi = torch.deg2rad(torch.tensor(self.phase))

        tm = self.transfer_map(incoming.energy)
        if isinstance(incoming, ParameterBeam):
            outgoing_mu = torch.matmul(tm, incoming._mu)
            outgoing_cov = torch.matmul(tm, torch.matmul(incoming._cov, tm.t()))
        else:  # ParticleBeam
            outgoing_particles = torch.matmul(incoming.particles, tm.t())
        delta_energy = self.voltage * torch.cos(phi)

        T566 = 1.5 * self.length * igamma2 / beta0**3
        T556 = 0
        T555 = 0
        if incoming.energy + delta_energy > 0:
            k = 2 * torch.pi * self.frequency / constants.speed_of_light
            outgoing_energy = incoming.energy + delta_energy
            g1 = outgoing_energy / electron_mass_eV
            beta1 = torch.sqrt(1 - 1 / g1**2)

            if isinstance(incoming, ParameterBeam):
                outgoing_mu[5] = (
                    incoming._mu[5]
                    + incoming.energy * beta0 / (outgoing_energy * beta1)
                    + self.voltage
                    * beta0
                    / (outgoing_energy * beta1)
                    * (torch.cos(incoming._mu[4] * beta0 * k + phi) - torch.cos(phi))
                )
                outgoing_cov[5, 5] = incoming._cov[5, 5]
                # outgoing_cov[5, 5] = (
                #     incoming._cov[5, 5]
                #     + incoming.energy * beta0 / (outgoing_energy * beta1)
                #     + self.voltage
                #     * beta0
                #     / (outgoing_energy * beta1)
                #     * (torch.cos(incoming._mu[4] * beta0 * k + phi) - torch.cos(phi))
                # )
            else:  # ParticleBeam
                outgoing_particles[:, 5] = (
                    incoming.particles[:, 5]
                    + incoming.energy * beta0 / (outgoing_energy * beta1)
                    + self.voltage
                    * beta0
                    / (outgoing_energy * beta1)
                    * (
                        torch.cos(incoming.particles[:, 4] * beta0 * k + phi)
                        - torch.cos(phi)
                    )
                )

            dgamma = self.voltage / electron_mass_eV
            if delta_energy > 0:
                T566 = (
                    self.length
                    * (beta0**3 * g0**3 - beta1**3 * g1**3)
                    / (2 * beta0 * beta1**3 * g0 * (g0 - g1) * g1**3)
                )
                T556 = (
                    beta0
                    * k
                    * self.length
                    * dgamma
                    * g0
                    * (beta1**3 * g1**3 + beta0 * (g0 - g1**3))
                    * torch.sin(phi)
                    / (beta1**3 * g1**3 * (g0 - g1) ** 2)
                )
                T555 = (
                    beta0**2
                    * k**2
                    * self.length
                    * dgamma
                    / 2.0
                    * (
                        dgamma
                        * (
                            2 * g0 * g1**3 * (beta0 * beta1**3 - 1)
                            + g0**2
                            + 3 * g1**2
                            - 2
                        )
                        / (beta1**3 * g1**3 * (g0 - g1) ** 3)
                        * torch.sin(phi) ** 2
                        - (g1 * g0 * (beta1 * beta0 - 1) + 1)
                        / (beta1 * g1 * (g0 - g1) ** 2)
                        * torch.cos(phi)
                    )
                )

            if isinstance(incoming, ParameterBeam):
                outgoing_mu[4] = (
                    T566 * incoming._mu[5] ** 2
                    + T556 * incoming._mu[4] * incoming._mu[5]
                    + T555 * incoming._mu[4] ** 2
                )
                outgoing_cov[4, 4] = (
                    T566 * incoming._cov[5, 5] ** 2
                    + T556 * incoming._cov[4, 5] * incoming._cov[5, 5]
                    + T555 * incoming._cov[4, 4] ** 2
                )
                outgoing_cov[4, 5] = (
                    T566 * incoming._cov[5, 5] ** 2
                    + T556 * incoming._cov[4, 5] * incoming._cov[5, 5]
                    + T555 * incoming._cov[4, 4] ** 2
                )
                outgoing_cov[5, 4] = outgoing_cov[4, 5]
            else:  # ParticleBeam
                outgoing_particles[:, 4] = (
                    T566 * incoming.particles[:, 5] ** 2
                    + T556 * incoming.particles[:, 4] * incoming.particles[:, 5]
                    + T555 * incoming.particles[:, 4] ** 2
                )

        if isinstance(incoming, ParameterBeam):
            outgoing = ParameterBeam(outgoing_mu, outgoing_cov, outgoing_energy)
            return outgoing
        else:  # ParticleBeam
            outgoing = ParticleBeam(
                outgoing_particles, outgoing_energy, device=incoming.device
            )
            return outgoing

    def _cavity_rmatrix(self, energy: float) -> torch.Tensor:
        """Produces an R-matrix for a cavity when it is on, i.e. voltage > 0.0."""
        phi = torch.deg2rad(torch.tensor(self.phase))
        delta_energy = torch.tensor(self.voltage) * torch.cos(phi)
        # Comment from Ocelot: Pure pi-standing-wave case
        eta = torch.tensor(1)
        Ei = energy / electron_mass_eV
        Ef = (energy + delta_energy) / electron_mass_eV
        Ep = (Ef - Ei) / torch.tensor(self.length)  # Derivative of the energy
        assert Ei > 0, "Initial energy must be larger than 0"

        alpha = torch.sqrt(eta / 8) / torch.cos(phi) * torch.log(Ef / Ei)

        r11 = torch.cos(alpha) - torch.sqrt(2 / eta) * torch.cos(phi) * torch.sin(alpha)

        # In Ocelot r12 is defined as below only if abs(Ep) > 10, and self.length
        # otherwise. This is implemented differently here in order to achieve results
        # closer to Bmad.
        r12 = torch.sqrt(8 / eta) * Ei / Ep * torch.cos(phi) * torch.sin(alpha)

        r21 = (
            -Ep
            / Ef
            * (
                torch.cos(phi) / torch.sqrt(2 * eta)
                + torch.sqrt(eta / 8) / torch.cos(phi)
            )
            * torch.sin(alpha)
        )

        r22 = (
            Ei
            / Ef
            * (
                torch.cos(alpha)
                + torch.sqrt(2 / eta) * torch.cos(phi) * torch.sin(alpha)
            )
        )

        r56 = 0
        beta0 = 1
        beta1 = 1

        k = 2 * torch.pi * self.frequency / constants.speed_of_light
        r55_cor = 0
        if self.voltage != 0 and energy != 0:  # TODO: Do we need this if?
            beta0 = torch.sqrt(1 - 1 / Ei**2)
            beta1 = torch.sqrt(1 - 1 / Ef**2)

            r56 = -self.length / (Ef**2 * Ei * beta1) * (Ef + Ei) / (beta1 + beta0)
            g0 = Ei
            g1 = Ef
            r55_cor = (
                k
                * self.length
                * beta0
                * self.voltage
                / electron_mass_eV
                * torch.sin(phi)
                * (g0 * g1 * (beta0 * beta1 - 1) + 1)
                / (beta1 * g1 * (g0 - g1) ** 2)
            )

        r66 = Ei / Ef * beta0 / beta1
        r65 = k * torch.sin(phi) * self.voltage / (Ef * beta1 * electron_mass_eV)

        R = torch.tensor(
            [
                [r11, r12, 0, 0, 0, 0, 0],
                [r21, r22, 0, 0, 0, 0, 0],
                [0, 0, r11, r12, 0, 0, 0],
                [0, 0, r21, r22, 0, 0, 0],
                [0, 0, 0, 0, 1 + r55_cor, r56, 0],
                [0, 0, 0, 0, r65, r66, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        return R

    def split(self, resolution: float) -> list[Element]:
        # TODO: Implement splitting for cavity properly, for now just returns the
        # element itself
        return [self]

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle(
            (s, 0), self.length, height, color="gold", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "voltage", "phase", "frequency"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={self.length:.2f}, "
            + f"voltage={self.voltage:.2f}, "
            + f"phase={self.phase:.2f}, "
            + f"frequency={self.frequency:.2f}, "
            + f'name="{self.name}", '
            + f'device="{self.device}")'
        )


class BPM(Element):
    """
    Beam Position Monitor (BPM) in a particle accelerator.

    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(self, name: Optional[str] = None, device: str = "auto") -> None:
        super().__init__(name=name, device=device)

        self.reading = (None, None)

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def transfer_map(self, energy: float) -> torch.Tensor:
        return torch.eye(7, device=self.device)

    def __call__(self, incoming: Beam) -> Beam:
        if incoming is Beam.empty:
            self.reading = (None, None)
            return Beam.empty
        elif isinstance(incoming, ParameterBeam):
            self.reading = (incoming.mu_x, incoming.mu_y)
            return ParameterBeam(incoming._mu, incoming._cov, incoming.energy)
        elif isinstance(incoming, ParticleBeam):
            self.reading = (incoming.mu_x, incoming.mu_y)
            return ParticleBeam(incoming.particles, incoming.energy, device=self.device)
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

    def split(self, resolution: float) -> list[Element]:
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
        return f'{self.__class__.__name__}(name={self.name}, device="{self.device}")'


class Marker(Element):
    """
    General Marker / Monitor element

    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(self, name: Optional[str] = None, device: str = "auto") -> None:
        super().__init__(name=name, device=device)

    def transfer_map(self, energy):
        return torch.eye(7, device=self.device)

    def __call__(self, incoming):
        return incoming

    @property
    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: float) -> list[Element]:
        return [self]

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        # Do nothing on purpose. Maybe later we decide markers should be shown, but for
        # now they are invisible.
        pass

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, device="{self.device}")'


class Screen(Element):
    """
    Diagnostic screen in a particle accelerator.

    :param resolution: Resolution of the camera sensor looking at the screen given as a
        tuple `(width, height)`.
    :param pixel_size: Size of a pixel on the screen in meters given as a tuple
        `(width, height)`.
    :param binning: Binning used by the camera.
    :param misalignment: Misalignment of the screen in meters given as a tuple `(x, y)`.
    :param is_active: If `True` the screen is active and will record the beam's
        distribution. If `False` the screen is inactive and will not record the beam's
        distribution.
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (1024, 1024),
        pixel_size: tuple[float, float] = (1e-3, 1e-3),
        binning: int = 1,
        misalignment: tuple[float, float] = (0, 0),
        is_active: bool = False,
        name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(name=name, device=device)

        self.resolution = tuple(resolution)
        self.pixel_size = tuple(pixel_size)
        self.binning = binning
        self.misalignment = tuple(misalignment)
        self.is_active = is_active

        self.read_beam = None
        self.cached_reading = None

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    @property
    def effective_resolution(self) -> tuple[int, int]:
        return (
            int(self.resolution[0] / self.binning),
            int(self.resolution[1] / self.binning),
        )

    @property
    def effective_pixel_size(self) -> tuple[float, float]:
        return (self.pixel_size[0] * self.binning, self.pixel_size[0] * self.binning)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (
            -self.resolution[0] * self.pixel_size[0] / 2,
            self.resolution[0] * self.pixel_size[0] / 2,
            -self.resolution[1] * self.pixel_size[1] / 2,
            self.resolution[1] * self.pixel_size[1] / 2,
        )

    @property
    def pixel_bin_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.linspace(
                -self.resolution[0] * self.pixel_size[0] / 2,
                self.resolution[0] * self.pixel_size[0] / 2,
                self.effective_resolution[0] + 1,
            ),
            torch.linspace(
                -self.resolution[1] * self.pixel_size[1] / 2,
                self.resolution[1] * self.pixel_size[1] / 2,
                self.effective_resolution[1] + 1,
            ),
        )

    def transfer_map(self, energy: float) -> torch.Tensor:
        return torch.eye(7, device=self.device)

    def __call__(self, incoming: Beam) -> Beam:
        if self.is_active:
            if isinstance(incoming, ParameterBeam):
                self.read_beam = deepcopy(incoming)
                self.read_beam._mu[0] -= self.misalignment[0]
                self.read_beam._mu[2] -= self.misalignment[1]
            elif isinstance(incoming, ParticleBeam):
                self.read_beam = deepcopy(incoming)
                x_offset = np.full(len(self.read_beam), self.misalignment[0])
                y_offset = np.full(len(self.read_beam), self.misalignment[1])
                self.read_beam.particles[:, 0] -= x_offset
                self.read_beam.particles[:, 1] -= y_offset
            else:
                self.read_beam = incoming

            return Beam.empty
        else:
            return incoming

    @property
    def reading(self) -> torch.Tensor:
        if self.cached_reading is not None:
            return self.cached_reading

        if self.read_beam is Beam.empty or self.read_beam is None:
            image = torch.zeros(
                (self.effective_resolution[1], self.effective_resolution[0])
            )
        elif isinstance(self.read_beam, ParameterBeam):
            transverse_mu = np.array([self.read_beam._mu[0], self.read_beam._mu[2]])
            transverse_cov = np.array(
                [
                    [self.read_beam._cov[0, 0], self.read_beam._cov[0, 2]],
                    [self.read_beam._cov[2, 0], self.read_beam._cov[2, 2]],
                ]
            )
            dist = multivariate_normal(
                mean=transverse_mu, cov=transverse_cov, allow_singular=True
            )

            left = self.extent[0]
            right = self.extent[1]
            hstep = self.pixel_size[0] * self.binning
            bottom = self.extent[2]
            top = self.extent[3]
            vstep = self.pixel_size[1] * self.binning
            x, y = np.mgrid[left:right:hstep, bottom:top:vstep]
            pos = np.dstack((x, y))
            image = dist.pdf(pos)
            image = np.flipud(image.T)
        elif isinstance(self.read_beam, ParticleBeam):
            image, _ = torch.histogramdd(
                torch.stack((self.read_beam.xs, self.read_beam.ys)).T,
                bins=self.pixel_bin_edges,
            )
            image = torch.flipud(image.T)
            image = image.cpu()
        else:
            raise TypeError(f"Read beam is of invalid type {type(self.read_beam)}")

        self.cached_reading = image
        return image

    @property
    def read_beam(self) -> Beam:
        return self._read_beam

    @read_beam.setter
    def read_beam(self, value: Beam) -> None:
        self._read_beam = value
        self.cached_reading = None

    def split(self, resolution: float) -> list[Element]:
        return [self]

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        patch = Rectangle(
            (s, -0.6), 0, 0.6 * 2, color="tab:green", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "resolution",
            "pixel_size",
            "binning",
            "misalignment",
            "is_active",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(resolution={self.resolution}, "
            + f"pixel_size={self.pixel_size}, "
            + f"binning={self.binning}, "
            + f"misalignment={self.misalignment}, "
            + f"is_active={self.is_active}, "
            + f'name="{self.name}", '
            + f'device="{self.device}")'
        )


class Aperture(Element):
    """
    Physical aperture.

    :param x_max: half size horizontal offset in [m]
    :param y_max: half size vertical offset in [m]
    :param shape: Shape of the aperture. Can be "rectangular" or "elliptical".
    :param is_active: If the aperture actually blocks particles.
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        x_max: float = np.inf,
        y_max: float = np.inf,
        shape: Literal["rectangular", "elliptical"] = "rectangular",
        is_active: bool = True,
        name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(name=name, device=device)

        self.x_max = x_max
        self.y_max = y_max
        self.shape = shape
        self.is_active = is_active

        self.lost_particles = None

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    def transfer_map(self, energy: float) -> torch.Tensor:
        return torch.eye(7, device=self.device)

    def __call__(self, incoming: Beam) -> Beam:
        # Only apply aperture to particle beams and if the element is active
        if not (isinstance(incoming, ParticleBeam) and self.is_active):
            return incoming

        assert self.x_max >= 0 and self.y_max >= 0
        assert self.shape in [
            "rectangular",
            "elliptical",
        ], f"Unknown aperture shape {self.shape}"

        if self.shape == "rectangular":
            survived_mask = torch.logical_and(
                torch.logical_and(incoming.xs > -self.x_max, incoming.xs < self.x_max),
                torch.logical_and(incoming.ys > -self.y_max, incoming.ys < self.y_max),
            )
        elif self.shape == "elliptical":
            survived_mask = (
                incoming.xs**2 / self.x_max**2 + incoming.ys**2 / self.y_max**2
            ) <= 1.0
        outgoing_particles = incoming.particles[survived_mask]

        self.lost_particles = incoming.particles[torch.logical_not(survived_mask)]

        return (
            ParticleBeam(outgoing_particles, incoming.energy, device=incoming.device)
            if outgoing_particles.shape[0] > 0
            else ParticleBeam.empty
        )

    def split(self, resolution: float) -> list[Element]:
        # TODO: Implement splitting for aperture properly, for now just return self
        return [self]

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.4

        dummy_length = 0.0

        patch = Rectangle(
            (s, 0), dummy_length, height, color="tab:pink", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "x_max",
            "y_max",
            "shape",
            "is_active",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(x_max={self.x_max:.2f}, "
            + f"y_max={self.y_max:.2f}, "
            + f'shape="{self.shape}", '
            + f"is_active={self.is_active}, "
            + f'name="{self.name}", '
            + f'device="{self.device}")'
        )


class Undulator(Element):
    """
    Element representing an undulator in a particle accelerator.

    NOTE Currently behaves like a drift section but is plotted distinctively.

    :param length: Length in meters.
    :param is_active: Indicates if the undulator is active or not. Currently has no
        effect.
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        length: float,
        is_active: bool = False,
        name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(name=name, device=device)

        self.length = length
        self.is_active = is_active

    def transfer_map(self, energy: float) -> torch.Tensor:
        gamma = energy / REST_ENERGY
        igamma2 = 1 / gamma**2 if gamma != 0 else 0

        return torch.tensor(
            [
                [1, self.length, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, self.length, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, self.length * igamma2, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: float) -> list[Element]:
        # TODO: Implement splitting for undulator properly, for now just return self
        return [self]

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle(
            (s, 0), self.length, height, color="tab:purple", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={self.length:.2f}, "
            + f"is_active={self.is_active}, "
            + f'name="{self.name}", '
            + f'device="{self.device}")'
        )


class Solenoid(Element):
    """
    Solenoid magnet.

    Implemented according to A.W.Chao P74

    :param length: Length in meters.
    :param k: Normalised strength of the solenoid magnet B0/(2*Brho). B0 is the field
        inside the solenoid, Brho is the momentum of central trajectory.
    :param misalignment: Misalignment vector of the solenoid magnet in x- and
        y-directions.
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        length: float = 0,
        k: float = 0,
        misalignment: tuple[float, float] = (0, 0),
        name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(name=name, device=device)

        self.length = length
        self.k = k
        self.misalignment = tuple(misalignment)

    def transfer_map(self, energy: float) -> torch.Tensor:
        gamma = energy / REST_ENERGY
        c = np.cos(self.length * self.k)
        s = np.sin(self.length * self.k)
        if self.k == 0:
            s_k = self.length
        else:
            s_k = s / self.k
        r56 = 0.0
        if gamma != 0:
            gamma2 = gamma * gamma
            beta = np.sqrt(1.0 - 1.0 / gamma2)
            r56 -= self.length / (beta * beta * gamma2)
        R = torch.tensor(
            [
                [c * c, c * s_k, s * c, s * s_k, 0, 0, 0],
                [-self.k * s * c, c * c, -self.k * s * s, s * c, 0, 0, 0],
                [-s * c, -s * s_k, c * c, c * s_k, 0, 0, 0],
                [self.k * s * s, -s * c, -self.k * s * c, c * c, 0, 0, 0],
                [0, 0, 0, 0, 1, r56, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        ).real
        if self.misalignment[0] == 0 and self.misalignment[1] == 0:
            return R
        else:
            R_exit, R_entry = misalignment_matrix(self.misalignment, self.device)
            R = torch.matmul(R_exit, torch.matmul(R, R_entry))
            return R

    @property
    def is_active(self) -> bool:
        return self.k != 0

    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: float) -> list[Element]:
        # TODO: Implement splitting for solenoid properly, for now just return self
        return [self]

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8

        patch = Rectangle(
            (s, 0), self.length, height, color="tab:orange", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "k", "misalignment"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={self.length:.2f}, "
            + f"k={self.k:.2f}, "
            + f"misalignment={self.misalignment}, "
            + f'name="{self.name}", '
            + f'device="{self.device}")'
        )


class Segment(Element):
    """
    Segment of a particle accelerator consisting of several elements.

    :param cell: List of Cheetah elements that describe an accelerator (section).
    :param name: Unique identifier of the element.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self, cell: list[Element], name: Optional[str] = None, device: str = "auto"
    ) -> None:
        global ELEMENT_COUNT
        if name is not None:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}_{ELEMENT_COUNT:06d}"
        ELEMENT_COUNT += 1

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.elements = cell

        for element in self.elements:
            element.device = self.device

            # Make elements accessible via .name attribute. If multiple elements have
            # the same name, they are accessible via a list.
            if element.name in self.__dict__:
                if isinstance(self.__dict__[element.name], list):
                    self.__dict__[element.name].append(element)
                else:  # Is instance of cheetah.Element
                    self.__dict__[element.name] = [self.__dict__[element.name], element]
            else:
                self.__dict__[element.name] = element

    def subcell(self, start: str, end: str, **kwargs) -> "Segment":
        """Extract a subcell `[start, end]` from an this segment."""
        subcell = []
        is_in_subcell = False
        for element in self.elements:
            if element.name == start:
                is_in_subcell = True
            if is_in_subcell:
                subcell.append(element)
            if element.name == end:
                break

        return self.__class__(subcell, device=self.device, **kwargs)

    def __eq__(self, other: "Segment") -> bool:
        for my_element, other_element in zip(self.elements, other.elements):
            if my_element != other_element:
                return False
        return True

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

        return Segment(cell=flattened_elements, name=self.name, device=self.device)

    @classmethod
    def from_ocelot(
        cls, cell, name: Optional[str] = None, warnings: bool = True, **kwargs
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
        :return: Cheetah segment closely resembling the Ocelot cell.
        """
        from cheetah.utils import ocelot2cheetah

        converted = [ocelot2cheetah(element, warnings=warnings) for element in cell]
        return cls(converted, name=name, **kwargs)

    @classmethod
    def from_bmad(
        cls, bmad_lattice_file_path: str, environment_variables: Optional[dict] = None
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
        :return: Cheetah `Segment` representing the Bmad lattice.
        """
        bmad_lattice_file_path = Path(bmad_lattice_file_path)
        return convert_bmad_lattice(bmad_lattice_file_path, environment_variables)

    @property
    def is_skippable(self) -> bool:
        return all(element.is_skippable for element in self.elements)

    @property
    def length(self) -> float:
        return sum(
            element.length for element in self.elements if hasattr(element, "length")
        )

    def transfer_map(self, energy: float) -> torch.Tensor:
        if self.is_skippable:
            tm = torch.eye(7, dtype=torch.float32, device=self.device)
            for element in self.elements:
                tm = torch.matmul(element.transfer_map(energy), tm)
            return tm
        else:
            return None

    def __call__(self, incoming: Beam) -> Beam:
        if self.is_skippable:
            return super().__call__(incoming)
        else:
            todos = []
            for element in self.elements:
                if not element.is_skippable:
                    todos.append(element)
                elif not todos or not todos[-1].is_skippable:
                    todos.append(Segment([element], device=self.device))
                else:
                    todos[-1].elements.append(element)

            for todo in todos:
                incoming = todo(incoming)

            return incoming

    def split(self, resolution: float) -> list[Element]:
        return [
            split_element
            for element in self.elements
            for split_element in element.split(resolution)
        ]

    def plot(self, ax: matplotlib.axes.Axes, s: float) -> None:
        element_lengths = [
            element.length if hasattr(element, "length") else 0.0
            for element in self.elements
        ]
        element_ss = [0] + [
            sum(element_lengths[: i + 1]) for i, _ in enumerate(element_lengths)
        ]
        element_ss = [s + element_s for element_s in element_ss]

        ax.plot([0, element_ss[-1]], [0, 0], "--", color="black")

        for element, s in zip(self.elements, element_ss[:-1]):
            element.plot(ax, s)

        ax.set_ylim(-1, 1)
        ax.set_xlabel("s (m)")
        ax.set_yticks([])

    def plot_reference_particle_traces(
        self,
        axx: matplotlib.axes.Axes,
        axy: matplotlib.axes.Axes,
        beam: Optional[Beam] = None,
        num_particles: int = 10,
        resolution: float = 0.01,
    ) -> None:
        """
        Plot `n` reference particles along the segment view in x- and y-direction.

        :param axx: Axes to plot the particle traces into viewed in x-direction.
        :param axy: Axes to plot the particle traces into viewed in y-direction.
        :param beam: Entering beam from which the reference particles are sampled.
        :param num_particles: Number of reference particles to plot. Must not be larger
            than number of particles passed in `beam`.
        :param resolution: Minimum resolution of the tracking of the reference particles
            in the plot.
        """
        reference_segment = deepcopy(self)
        splits = reference_segment.split(resolution)

        split_lengths = [split.length for split in splits]
        ss = [0] + [sum(split_lengths[: i + 1]) for i, _ in enumerate(split_lengths)]

        references = []
        if beam is None:
            initial = ParticleBeam.make_linspaced(
                num_particles=num_particles, device="cpu"
            )
            references.append(initial)
        else:
            initial = ParticleBeam.make_linspaced(
                num_particles=num_particles,
                mu_x=beam.mu_x,
                mu_xp=beam.mu_xp,
                mu_y=beam.mu_y,
                mu_yp=beam.mu_yp,
                sigma_x=beam.sigma_x,
                sigma_xp=beam.sigma_xp,
                sigma_y=beam.sigma_y,
                sigma_yp=beam.sigma_yp,
                sigma_s=beam.sigma_s,
                sigma_p=beam.sigma_p,
                energy=beam.energy,
                device="cpu",
            )
            references.append(initial)
        for split in splits:
            sample = split(references[-1])
            references.append(sample)

        for particle_index in range(num_particles):
            xs = [
                float(reference_beam.xs[particle_index].cpu())
                for reference_beam in references
                if reference_beam is not Beam.empty
            ]
            axx.plot(ss[: len(xs)], xs)
        axx.set_xlabel("s (m)")
        axx.set_ylabel("x (m)")
        axx.grid()

        for particle_index in range(num_particles):
            ys = [
                float(reference_beam.ys[particle_index].cpu())
                for reference_beam in references
                if reference_beam is not Beam.empty
            ]
            axy.plot(ss[: len(ys)], ys)
        axx.set_xlabel("s (m)")
        axy.set_ylabel("y (m)")
        axy.grid()

    def plot_overview(
        self,
        fig: Optional[matplotlib.figure.Figure] = None,
        beam: Optional[Beam] = None,
        n: int = 10,
        resolution: float = 0.01,
    ) -> None:
        """
        Plot an overview of the segment with the lattice and traced reference particles.

        :param fig: Figure to plot the overview into.
        :param beam: Entering beam from which the reference particles are sampled.
        :param n: Number of reference particles to plot. Must not be larger than number
            of particles passed in `beam`.
        :param resolution: Minimum resolution of the tracking of the reference particles
            in the plot.
        """
        if fig is None:
            fig = plt.figure()
        gs = fig.add_gridspec(3, hspace=0, height_ratios=[2, 2, 1])
        axs = gs.subplots(sharex=True)

        axs[0].set_title("Reference Particle Traces")
        self.plot_reference_particle_traces(axs[0], axs[1], beam, n, resolution)

        self.plot(axs[2], 0)

        plt.tight_layout()

    def plot_twiss(self, beam: Beam, ax: Optional[Any] = None) -> None:
        """Plot twiss parameters along the segment."""
        longitudinal_beams = [beam]
        s_positions = [0.0]
        for element in self.elements:
            if not hasattr(element, "length") or element.length == 0:
                continue

            outgoing = element.track(longitudinal_beams[-1])

            longitudinal_beams.append(outgoing)
            s_positions.append(s_positions[-1] + element.length)

        beta_x = [beam.beta_x for beam in longitudinal_beams]
        beta_y = [beam.beta_y for beam in longitudinal_beams]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.set_title("Twiss Parameters")
        ax.set_xlabel("s (m)")
        ax.set_ylabel(r"$\beta$ (m)")

        ax.plot(s_positions, beta_x, label=r"$\beta_x$", c="tab:red")
        ax.plot(s_positions, beta_y, label=r"$\beta_y$", c="tab:green")

        ax.legend()
        plt.tight_layout()

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["elements"]

    def plot_twiss_over_lattice(self, beam: Beam, figsize=(8, 4)) -> None:
        """Plot twiss parameters in a plot over a plot of the lattice."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])
        axs = gs.subplots(sharex=True)

        self.plot_twiss(beam, ax=axs[0])
        self.plot(axs[1], 0)

        plt.tight_layout()

    def __repr__(self) -> str:
        start = f"{self.__class__.__name__}(["

        s = start + self.elements[0].__repr__()
        x = [", " + element.__repr__() for element in self.elements[1:]]
        s += "".join(x)
        s += "])"

        return s
