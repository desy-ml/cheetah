from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import constants
import torch
from torch._C import device

from cheetah import utils
from cheetah.particles import Beam


ELEMENT_COUNT = 0
REST_ENERGY = constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
        

class Element:
    """
    Base class for elements of particle accelerators.

    Parameters
    ----------
    name : string, optional
        Unique identifier of the element.
    
    Attributes
    ---------
    is_active : bool
        Is set to `True` when the element is in operation. May be defined differently for each type
        of element.
    is_skippable : bool
        Marking an element as skippable allows the transfer map to be combined with those of
        preceeding and succeeding elements and results in faster particle tracking. This property
        has to be defined by subclasses of `Element` and made be set dynamically depending on their
        current mode of operation.
    device : string
        Device to move the beam's particle array to. If set to `"auto"` a CUDA GPU is selected if
        available. The CPU is used otherwise.
    """

    is_active = False
    is_skippable = True

    def __init__(self, name=None, device="auto"):
        global ELEMENT_COUNT
        if name is not None:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}_{ELEMENT_COUNT:06d}"
        ELEMENT_COUNT += 1

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
    
    def transfer_map(self, energy):
        raise NotImplementedError

    def __call__(self, incoming):
        """
        Track particles through the element.
        
        Pramameters
        -----------
        incoming : cheetah.Beam
            Beam of particles entering the element.

        Returns
        -------
        cheetah.Beam
            Beam of particles exiting the element.
        """
        if incoming.is_empty:
            return incoming
        else:
            tm = self.transfer_map(incoming.energy)
            new_particles = torch.matmul(incoming.particles, tm.t())
            return Beam(new_particles, incoming.energy)

    def split(self, resolution):
        """
        Split the element into slices no longer than `resolution`.

        Parameters
        ----------
        resolution : float
            Length of the longest allowed split in meters.

        Returns
        -------
        list
            Ordered sequence of sliced elements.
        
        Raises
        ------
        NotImplementedError
            If not split function is implemented for the given `Element` subclass.
        """
        raise NotImplementedError
    
    def plot(self, ax, s):
        """
        Plot a representation of this element into a `matplotlib` Axes at position `s`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot the representation into.
        s : float
            Position of the object along s in meters.
        
        Raises
        ------
        NotImplementedError
            If not split function is implemented for the given `Element` subclass.
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name=\"{self.name}\")"


class Drift(Element):
    """
    Drift section in a particle accelerator.

    Parameters
    ----------
    length : float
        Length in meters.
    energy : float, optional
    name : string, optional
        Unique identifier of the element.
    
    Attributes
    ---------
    is_active : bool
        Drifts are always set as active.
    """

    is_active = True
    is_skippable = True

    def __init__(self, length, name=None, **kwargs):
        self.length = length

        super().__init__(name=name, **kwargs)
    
    def transfer_map(self, energy):
        gamma = energy / REST_ENERGY
        igamma2 = 1 / gamma**2 if gamma != 0 else 0
        
        return torch.tensor([[1, self.length, 0,           0, 0,                     0, 0],
                             [0,           1, 0,           0, 0,                     0, 0],
                             [0,           0, 1, self.length, 0,                     0, 0],
                             [0,           0, 0,           1, 0,                     0, 0],
                             [0,           0, 0,           0, 1, self.length * igamma2, 0],
                             [0,           0, 0,           0, 0,                     1, 0],
                             [0,           0, 0,           0, 0,                     0, 1]],
                            dtype=torch.float32, device=self.device)
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Drift(min(resolution, remaining), device=self.device)
            split_elements.append(element)
            remaining -= resolution
        return split_elements
    
    def plot(self, ax, s):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(length={self.length:.2f}, name=\"{self.name}\")"


class Quadrupole(Element):
    """
    Quadrupole magnet in a particle accelerator.

    Parameters
    ----------
    length : float
        Length in meters.
    k1 : float, optional
        Strength of the quadrupole in rad/m.
    misalignment : (float, float), optional
        Misalignment vector of the quadrupole in x- and y-directions.
    energy : float, optional
    name : string, optional
        Unique identifier of the element.
    
    Attributes
    ---------
    is_active : bool
        Is set `True` when `k1 != 0`.
    """

    is_skippable = True

    def __init__(self, length, k1=0.0, misalignment=(0,0), name=None, **kwargs):
        self.length = length
        self.k1 = k1
        self.misalignment = misalignment

        super().__init__(name=name, **kwargs)
    
    def transfer_map(self, energy):
        gamma = energy / REST_ENERGY
        igamma2 = 1 / gamma**2 if gamma != 0 else 0

        beta = np.sqrt(1 - igamma2)
        
        hx = 0
        kx2 = self.k1 + hx**2
        ky2 = -self.k1
        kx = np.sqrt(kx2 + 0.j)
        ky = np.sqrt(ky2 + 0.j)
        cx = np.cos(kx * self.length).real
        cy = np.cos(ky * self.length).real
        sy = (np.sin(ky * self.length) / ky).real if ky != 0 else self.length

        if kx != 0:
            sx = (np.sin(kx * self.length) / kx).real
            dx = hx / kx2 * (1. - cx)
            r56 = hx**2 * (self.length - sx) / kx2 / beta**2
        else:
            sx = self.length
            dx = self.length**2 * hx / 2
            r56 = hx**2 * self.length**3 / 6 / beta**2
        
        r56 -= self.length / beta**2 * igamma2

        R = torch.tensor([[            cx,        sx,         0,  0, 0,      dx / beta, 0],
                          [     -kx2 * sx,        cx,         0,  0, 0, sx * hx / beta, 0],
                          [             0,         0,        cy, sy, 0,              0, 0],
                          [             0,         0, -ky2 * sy, cy, 0,              0, 0],
                          [sx * hx / beta, dx / beta,         0,  0, 1,            r56, 0],
                          [             0,         0,         0,  0, 0,              1, 0],
                          [             0,         0,         0,  0, 0,              0, 1]],
                         dtype=torch.float32, device=self.device)
        
        if self.misalignment[0] == 0 and self.misalignment[1] == 0:
            return R
        else:
            R_entry = torch.tensor([[1, 0, 0, 0, 0, 0, self.misalignment[0]],
                                    [0, 1, 0, 0, 0, 0,                    0],
                                    [0, 0, 1, 0, 0, 0, self.misalignment[1]],
                                    [0, 0, 0, 1, 0, 0,                    0],
                                    [0, 0, 0, 0, 1, 0,                    0],
                                    [0, 0, 0, 0, 0, 1,                    0],
                                    [0, 0, 0, 0, 0, 0,                    1]],
                                   dtype=torch.float32, device=self.device)
            R_exit = torch.tensor([[1, 0, 0, 0, 0, 0, -self.misalignment[0]],
                                   [0, 1, 0, 0, 0, 0,                     0],
                                   [0, 0, 1, 0, 0, 0, -self.misalignment[1]],
                                   [0, 0, 0, 1, 0, 0,                     0],
                                   [0, 0, 0, 0, 1, 0,                     0],
                                   [0, 0, 0, 0, 0, 1,                     0],
                                   [0, 0, 0, 0, 0, 0,                     1]],
                                   dtype=torch.float32, device=self.device)
            R = torch.matmul(R_entry, R)
            R = torch.matmul(R, R_exit)
            return R
    
    @property
    def is_active(self):
        return self.k1 != 0
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Quadrupole(min(resolution, remaining),
                                 self.k1,
                                 misalignment=self.misalignment,
                                 device=self.device)
            split_elements.append(element)
            remaining -= resolution
        return split_elements
    
    def plot(self, ax, s):
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.k1) if self.is_active else 1)
        patch = Rectangle((s, 0),
                           self.length,
                           height,
                           color="tab:red",
                           alpha=alpha,
                           zorder=2)
        ax.add_patch(patch)

    def __repr__(self):
        return f"{self.__class__.__name__}(length={self.length:.2f}, " + \
                                         f"k1={self.k1}, " + \
                                         f"name=\"{self.name}\")"


class HorizontalCorrector(Element):
    """
    Horizontal corrector magnet in a particle accelerator.

    Parameters
    ----------
    length : float
        Length in meters.
    angle : float, optional
        Particle deflection angle in the horizontal plane in rad.
    energy : float, optional
    name : string, optional
        Unique identifier of the element.
    
    Attributes
    ---------
    is_active : bool
        Is set `True` when `angle != 0`.
    """

    is_skippable = True

    def __init__(self, length, angle=0.0, energy=1e+8, name=None, **kwargs):
        self.length = length
        self.angle = angle

        super().__init__(name=name, **kwargs)

    def transfer_map(self, energy):
        return torch.tensor([[1, self.length, 0,           0, 0, 0,          0],
                             [0,           1, 0,           0, 0, 0, self.angle],
                             [0,           0, 1, self.length, 0, 0,          0],
                             [0,           0, 0,           1, 0, 0,          0],
                             [0,           0, 0,           0, 1, 0,          0],
                             [0,           0, 0,           0, 0, 1,          0],
                             [0,           0, 0,           0, 0, 0,          1]],
                            dtype=torch.float32, device=self.device)
    
    @property
    def is_active(self):
        return self.angle != 0
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = min(resolution, remaining)
            element = HorizontalCorrector(length,
                                          self.angle * length / self.length,
                                          device=self.device)
            split_elements.append(element)
            remaining -= resolution
        return split_elements
    
    def plot(self, ax, s):
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.angle) if self.is_active else 1)

        patch = Rectangle((s, 0),
                           self.length,
                           height,
                           color="tab:blue",
                           alpha=alpha,
                           zorder=2)
        ax.add_patch(patch)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(length={self.length:.2f}, " + \
                                         f"angle={self.angle}, " + \
                                         f"name=\"{self.name}\")"


class VerticalCorrector(Element):
    """
    Verticle corrector magnet in a particle accelerator.

    Parameters
    ----------
    length : float
        Length in meters.
    angle : float, optional
        Particle deflection angle in the vertical plane in rad.
    energy : float, optional
    name : string, optional
        Unique identifier of the element.
    
    Attributes
    ---------
    is_active : bool
        Is set `True` when `angle != 0`.
    """

    is_skippable = True

    def __init__(self, length, angle=0.0, energy=1e+8, name=None, **kwargs):
        self.length = length
        self.angle = angle

        super().__init__(name=name, **kwargs)

    def transfer_map(self, energy):
        return torch.tensor([[1, self.length, 0,           0, 0, 0,          0],
                             [0,           1, 0,           0, 0, 0,          0],
                             [0,           0, 1, self.length, 0, 0,          0],
                             [0,           0, 0,           1, 0, 0, self.angle],
                             [0,           0, 0,           0, 1, 0,          0],
                             [0,           0, 0,           0, 0, 1,          0],
                             [0,           0, 0,           0, 0, 0,          1]],
                            dtype=torch.float32, device=self.device)
    
    @property
    def is_active(self):
        return self.angle != 0
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = min(resolution, remaining)
            element = VerticalCorrector(length,
                                        self.angle * length / self.length,
                                        device=self.device)
            split_elements.append(element)
            remaining -= resolution
        return split_elements
    
    def plot(self, ax, s):
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.angle) if self.is_active else 1)

        patch = Rectangle((s, 0),
                           self.length,
                           height,
                           color="tab:cyan",
                           alpha=alpha,
                           zorder=2)
        ax.add_patch(patch)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(length={self.length:.2f}, " + \
                                         f"angle={self.angle}, " + \
                                         f"name=\"{self.name}\")"


class Cavity(Element):
    """
    Accelerating cavity in a particle accelerator.

    Parameters
    ----------
    length : float
        Length in meters.
    delta_energy : float, optional
        Energy added to the beam by the accelerating cavity.
    name : string, optional
        Unique identifier of the element.
    """

    def __init__(self, length, delta_energy=0, name=None, **kwargs):
        self.length = length
        self.delta_energy = 0

        super().__init__(name=name, **kwargs)
    
    @property
    def is_active(self):
        return self.delta_energy != 0

    @property
    def is_skippable(self):
        return not self.is_active
    
    def transfer_map(self, energy):
        gamma = energy / REST_ENERGY
        igamma2 = 1 / gamma**2 if gamma != 0 else 0

        return torch.tensor([[1, self.length, 0,           0, 0,                     0, 0],
                             [0,           1, 0,           0, 0,                     0, 0],
                             [0,           0, 1, self.length, 0,                     0, 0],
                             [0,           0, 0,           1, 0,                     0, 0],
                             [0,           0, 0,           0, 1, self.length * igamma2, 0],
                             [0,           0, 0,           0, 0,                     1, 0],
                             [0,           0, 0,           0, 0,                     0, 1]],
                            dtype=torch.float32, device=self.device)
    
    def __call__(self, incoming):
        outgoing = super().__call__(incoming)
        if not outgoing.is_empty:
            outgoing.energy += self.delta_energy
        return outgoing
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            split_length = min(resolution, remaining)
            split_delta_energy = self.delta_energy * split_length / self.length
            element = Cavity(split_length, delta_energy=split_delta_energy, device=self.device)
            split_elements.append(element)
            remaining -= resolution
        return split_elements
    
    def plot(self, ax, s):
        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle((s, 0),
                           self.length,
                           height,
                           color="gold",
                           alpha=alpha,
                           zorder=2)
        ax.add_patch(patch)

    def __repr__(self):
        return f"{self.__class__.__name__}(length={self.length:.2f}, delta_energy={self.delta_energy}, name=\"{self.name}\")"


class BPM(Element):
    """
    Beam Position Monitor (BPM) in a particle accelerator.

    Parameters
    ----------
    name : string, optional
        Unique identifier of the element.
    
    Attributes
    ---------
    is_active : bool
        Can be set by the user. Merely influences how the element is displayed in a lattice plot.
    reading : (float, float)
        Beam position read by the BPM. Is refreshed when the BPM is active and a beam is tracked
        through it. Before tracking a beam through here, the reading is initialised as `(None, None)`.
    """

    length = 0
    is_skippable = True # TODO: Temporary
    
    reading = (None, None)

    @property
    def is_skippable(self):
        return not self.is_active

    def transfer_map(self, energy):
        return torch.eye(7, device=self.device)
    
    def __call__(self, incoming):
        if incoming.is_empty:
            self.reading = (None, None)
        else:
            self.reading = (incoming.mu_x, incoming.mu_y)
        return Beam(incoming.particles, incoming.energy, device=self.device)
    
    def split(self, resolution):
        return [self]
    
    def plot(self, ax, s):
        alpha = 1 if self.is_active else 0.2
        patch = Rectangle((s, -0.3),
                           0,
                           0.3 * 2,
                           color="darkkhaki",
                           alpha=alpha,
                           zorder=2)
        ax.add_patch(patch)


class Screen(Element):
    """
    Diagnostic screen in a particle accelerator.

    Parameters
    ----------
    name : string, optional
        Unique identifier of the element.
    resolution : (int, int)
        Resolution of the camera sensor looking at the screen given as a tuple `(width, height)`.
    binning : int, optional
        Binning used by the camera.
    
    Attributes
    ---------
    is_active : bool
        Can be set by the user. An active screen records an image and blocks all particles when a
        beam is tracked through it.
    """

    length = 0

    def __init__(self, resolution, pixel_size, binning=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.resolution = resolution
        self.pixel_size = pixel_size
        self.binning = binning

        x, y = int(resolution[0] / binning), int(resolution[1] / binning)
        self.reading = torch.zeros((y,x), device=self.device)
        
    @property
    def is_skippable(self):
        return not self.is_active
    
    @property
    def extent(self):
        return (-self.resolution[0] * self.pixel_size[0] / 2,
                self.resolution[0] * self.pixel_size[0] / 2,
                -self.resolution[1] * self.pixel_size[1] / 2,
                self.resolution[1] * self.pixel_size[1] / 2)
    
    @property
    def pixel_bin_edges(self):
        return (torch.linspace(-self.resolution[0] * self.pixel_size[0] / 2,
                               self.resolution[0] * self.pixel_size[0] / 2,
                               int(self.resolution[0] / self.binning) + 1),
                torch.linspace(-self.resolution[1] * self.pixel_size[1] / 2,
                               self.resolution[1] * self.pixel_size[1] / 2,
                               int(self.resolution[1] / self.binning) + 1))

    def transfer_map(self, energy):
        return torch.eye(7, device=self.device)

    def __call__(self, incoming):
        if self.is_active:
            if incoming.is_empty:
                x = int(self.resolution[0] / self.binning)
                y = int(self.resolution[1] / self.binning)
                self.reading = torch.zeros((y,x))
            else:
                # image, _, _ = np.histogram2d(incoming.xs, incoming.ys, bins=self.pixel_bin_edges)
                image, _ = utils.histogramdd(torch.stack((incoming.xs,incoming.ys)), bins=self.pixel_bin_edges)
                image = torch.flipud(image.T)

                self.reading = image.cpu()
                self.read_beam = incoming

            return Beam([], 0, device=self.device)
        else:
            return incoming
    
    def split(self, resolution):
        return [self]
    
    def plot(self, ax, s):
        alpha = 1 if self.is_active else 0.2
        patch = Rectangle((s, -0.6),
                           0,
                           0.6 * 2,
                           color="tab:green",
                           alpha=alpha,
                           zorder=2)
        ax.add_patch(patch)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(resolution={self.resolution}, pixel_size={self.pixel_size}, binning={self.binning}, name=\"{self.name}\")"


class Undulator(Element):
    """
    Element representing an undulator in a particle accelerator.

    Parameters
    ----------
    length : float
        Length in meters.
    name : string, optional
        Unique identifier of the element.
    
    Notes
    -----
    Currently behaves like a drift section but is plotted distinctively.
    """

    is_skippable = True # TODO: Temporary?

    def __init__(self, length, name=None, **kwargs):
        self.length = length

        super().__init__(name=name, **kwargs)
    
    def transfer_map(self, energy):
        gamma = energy / REST_ENERGY
        igamma2 = 1 / gamma**2 if gamma != 0 else 0

        return torch.tensor([[1, self.length, 0,           0, 0,                     0, 0],
                             [0,           1, 0,           0, 0,                     0, 0],
                             [0,           0, 1, self.length, 0,                     0, 0],
                             [0,           0, 0,           1, 0,                     0, 0],
                             [0,           0, 0,           0, 1, self.length * igamma2, 0],
                             [0,           0, 0,           0, 0,                     1, 0],
                             [0,           0, 0,           0, 0,                     0, 1]],
                            dtype=torch.float32, device=self.device)
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Cavity(min(resolution, remaining), device=self.device)
            split_elements.append(element)
            remaining -= resolution
        return split_elements
    
    def plot(self, ax, s):
        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle((s, 0),
                           self.length,
                           height,
                           color="tab:purple",
                           alpha=alpha,
                           zorder=2)
        ax.add_patch(patch)

    def __repr__(self):
        return f"{self.__class__.__name__}(length={self.length:.2f}, name=\"{self.name}\")"


class Segment(Element):
    """
    Segment of a particle accelerator consisting of several elements.

    Parameters
    ----------
    cell : list
        List of Cheetah elements that describe an accelerator (section).
    name : string, optional
        Unique identifier of the element.
    """

    def __init__(self, cell, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.elements = cell
        
        for element in self.elements:
            element.device = self.device
            self.__dict__[element.name] = element
    
    @classmethod
    def from_ocelot(cls, cell, name=None, **kwargs):
        converted = [utils.ocelot2cheetah(element) for element in cell]
        return cls(converted, name=name, **kwargs)
    
    def to_device(self, device):
        for element in self.elements:
            element.device = device
    
    @property
    def is_skippable(self):
        return all(element.is_skippable for element in self.elements)
    
    @property
    def length(self):
        return sum(element.length for element in self.elements)
    
    def transfer_map(self, energy):
        if self.is_skippable:
            tm = torch.eye(7, dtype=torch.float32, device=self.device)
            for element in self.elements:
                tm = torch.matmul(element.transfer_map(energy), tm)
            return tm
        else:
            return None
    
    def __call__(self, incoming):
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
    
    def split(self, resolution):
        return [split_element for element in self.elements
                              for split_element in element.split(resolution)]
    
    def plot(self, ax, s):
        element_lengths = [element.length for element in self.elements]
        element_ss = [0] + [sum(element_lengths[:i+1]) for i, _ in enumerate(element_lengths)]
        element_ss = [s + element_s for element_s in element_ss]

        ax.plot([0, element_ss[-1]], [0, 0], "--", color="black")

        for element, s in zip(self.elements, element_ss[:-1]):
            element.plot(ax, s)

        ax.set_ylim(-1, 1)
        ax.set_xlabel("s (m)")
        ax.set_yticks([])
        ax.grid()
    
    def plot_reference_particle_traces(self, axx, axy, beam=None, n=10, resolution=0.01):
        """
        Plot `n` reference particles along the segment view in x- and y-direction.

        Parameters
        ----------
        axx : matplotlib.axes.Axes
            Axes to plot the particle traces into viewed in x-direction.
        axy : matplotlib.axes.Axes
            Axes to plot the particle traces into viewed in y-direction.
        beam : cheetah.Beam, optional
            Entering beam from which the reference particles are sampled.
        n : int, optional
            Number of reference particles to plot. Must not be larger than number of particles
            passed in `particles`.
        resolution : float, optional
            Minimum resolution of the tracking of the reference particles in the plot.
        """
        reference_segment = deepcopy(self).to_device("cpu")
        splits = reference_segment.split(resolution)

        split_lengths = [split.length for split in splits]
        ss = [0] + [sum(split_lengths[:i+1]) for i, _ in enumerate(split_lengths)]

        references = []
        if beam is None:
            initial = Beam.make_linspaced(n=n, device="cpu")
            references.append(initial)
        else:
            initial = Beam.make_linspaced(n=n, mu_x=beam.mu_x, mu_xp=beam.mu_xp, mu_y=beam.mu_y,
                                          mu_yp=beam.mu_yp, sigma_x=beam.sigma_x,
                                          sigma_xp=beam.sigma_xp, sigma_y=beam.sigma_y,
                                          sigma_yp=beam.sigma_yp, sigma_s=beam.sigma_s,
                                          sigma_p=beam.sigma_p, energy=beam.energy, device="cpu")
            references.append(initial)
        for split in splits:
            sample = split(references[-1])
            references.append(sample)
        
        for particle_index in range(n):
            xs = [reference_beam.xs[particle_index] for reference_beam in references
                                                    if reference_beam.xs is not None]
            axx.plot(ss[:len(xs)], xs)
        axx.set_xlabel("s (m)")
        axx.set_ylabel("x (m)")
        axx.grid()

        for particle_index in range(n):
            ys = [reference_beam.ys[particle_index] for reference_beam in references
                                                    if reference_beam.ys is not None]
            axy.plot(ss[:len(ys)], ys)
        axx.set_xlabel("s (m)")
        axy.set_ylabel("y (m)")
        axy.grid()
    
    def plot_overview(self, fig=None, beam=None, n=10, resolution=0.01):
        """
        Plot an overview of the segment with the lattice and traced reference particles.

        Parameters
        ----------
        fig: matplotlib.figure.Figure, optional
            Figure to plot the overview into.
        beam : cheetah.Beam, optional
            Entering beam from which the reference particles are sampled.
        n : int, optional
            Number of reference particles to plot. Must not be larger than number of particles
            passed in `beam`.
        resolution : float, optional
            Minimum resolution of the tracking of the reference particles in the plot.
        """
        if fig is None:
            fig = plt.figure()
        gs = fig.add_gridspec(3, hspace=0, height_ratios=[2,2,1])
        axs = gs.subplots(sharex=True)

        axs[0].set_title("Reference Particle Traces")
        self.plot_reference_particle_traces(axs[0], axs[1], beam, n, resolution)

        self.plot(axs[2], 0)

        plt.tight_layout()

    def __repr__(self):
        start = f"{self.__class__.__name__}(["

        s = start + self.elements[0].__repr__()
        x = [", " + element.__repr__() for element in self.elements[1:]]
        s += "".join(x)
        s += "])"

        return s
