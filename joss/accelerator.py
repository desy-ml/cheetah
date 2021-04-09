import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import constants

from joss.particles import generate_particles
from joss.utils import ocelot2joss


ELEMENT_COUNT = 0
REST_ENERGY = constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
        

class Element:
    """
    Base class for elements of particle accelerators.

    Parameters
    ----------
    name : string, optional
        Unique identifier of the element.
    """

    def __init__(self, name=None):
        global ELEMENT_COUNT

        if name is not None:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}_{ELEMENT_COUNT:06d}"
        
        ELEMENT_COUNT += 1
    
    @property
    def transfer_map(self):
        raise NotImplementedError

    def __call__(self, particles):
        """
        Track particles through the element.
        
        Pramameters
        -----------
        particles : numpy.ndarray
            Array of particles entering the element.

        Returns
        -------
        numpy.ndarray
            Particles exiting the element.
        """
        return np.matmul(particles, self.transfer_map.transpose())
    
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
    """

    def __init__(self, length, energy=1e+8, name=None):
        self.length = length
        self.energy = energy

        super().__init__(name=name)
    
    @property
    def transfer_map(self):
        gamma = self.energy / REST_ENERGY
        igamma2 = 1 / gamma**2 if gamma != 0 else 0

        return np.array([[1, self.length, 0,           0, 0,                     0, 0],
                         [0,           1, 0,           0, 0,                     0, 0],
                         [0,           0, 1, self.length, 0,                     0, 0],
                         [0,           0, 0,           1, 0,                     0, 0],
                         [0,           0, 0,           0, 1, self.length * igamma2, 0],
                         [0,           0, 0,           0, 0,                     1, 0],
                         [0,           0, 0,           0, 0,                     0, 1]])
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Drift(min(resolution, remaining), energy=self.energy)
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
    k1 : float
        Strength of the quadrupole in rad/m.
    energy : float, optional
    name : string, optional
        Unique identifier of the element.
    """

    def __init__(self, length, k1, energy=1e+8, name=None):
        self.length = length
        self.k1 = k1
        self.energy = energy

        super().__init__(name=name)
    
    @property
    def transfer_map(self):
        gamma = self.energy / REST_ENERGY
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

        return np.array([[            cx,        sx,         0,  0, 0,      dx / beta, 0],
                         [     -kx2 * sx,        cx,         0,  0, 0, sx * hx / beta, 0],
                         [             0,         0,        cy, sy, 0,              0, 0],
                         [             0,         0, -ky2 * sy, cy, 0,              0, 0],
                         [sx * hx / beta, dx / beta,         0,  0, 1,            r56, 0],
                         [             0,         0,         0,  0, 0,              1, 0],
                         [             0,         0,         0,  0, 0,              0, 1]])
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Quadrupole(min(resolution, remaining), self.k1, energy=self.energy)
            split_elements.append(element)
            remaining -= resolution
        return split_elements
    
    def plot(self, ax, s):
        is_active = self.k1 != 0

        alpha = 1 if is_active else 0.2
        height = 0.8 * (np.sign(self.k1) if is_active else 1)
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
    angle : float
        Particle deflection angle in the horizontal plane in rad.
    energy : float, optional
    name : string, optional
        Unique identifier of the element.
    """

    def __init__(self, length, angle, energy=1e+8, name=None):
        self.length = length
        self.angle = angle

        super().__init__(name=name)

    @property
    def transfer_map(self):
        return np.array([[1, self.length, 0,           0, 0, 0,          0],
                         [0,           1, 0,           0, 0, 0, self.angle],
                         [0,           0, 1, self.length, 0, 0,          0],
                         [0,           0, 0,           1, 0, 0,          0],
                         [0,           0, 0,           0, 1, 0,          0],
                         [0,           0, 0,           0, 0, 1,          0],
                         [0,           0, 0,           0, 0, 0,          1]])
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = min(resolution, remaining)
            element = HorizontalCorrector(length,
                                          self.angle * length / self.length)
            split_elements.append(element)
            remaining -= resolution
        return split_elements
    
    def plot(self, ax, s):
        is_active = self.angle != 0

        alpha = 1 if is_active else 0.2
        height = 0.8 * (np.sign(self.angle) if is_active else 1)

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
    angle : float
        Particle deflection angle in the vertical plane in rad.
    energy : float, optional
    name : string, optional
        Unique identifier of the element.
    """

    def __init__(self, length, angle, energy=1e+8, name=None):
        self.length = length
        self.angle = angle

        super().__init__(name=name)

    @property
    def transfer_map(self):
        return np.array([[1, self.length, 0,           0, 0, 0,          0],
                         [0,           1, 0,           0, 0, 0,          0],
                         [0,           0, 1, self.length, 0, 0,          0],
                         [0,           0, 0,           1, 0, 0, self.angle],
                         [0,           0, 0,           0, 1, 0,          0],
                         [0,           0, 0,           0, 0, 1,          0],
                         [0,           0, 0,           0, 0, 0,          1]])
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = min(resolution, remaining)
            element = HorizontalCorrector(length,
                                          self.angle * length / self.length)
            split_elements.append(element)
            remaining -= resolution
        return split_elements
    
    def plot(self, ax, s):
        is_active = self.angle != 0

        alpha = 1 if is_active else 0.2
        height = 0.8 * (np.sign(self.angle) if is_active else 1)

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


class Screen(Element):
    """
    Screen in a particle accelerator.

    Parameters
    ----------
    name : string, optional
        Unique identifier of the element.
    """

    length = 0
    transfer_map = np.eye(7)

    def __call__(self, particles):
        return particles
    
    def split(self, resolution):
        return []
    
    def plot(self, ax, s):
        patch = Rectangle((s, -0.6),
                           0,
                           0.6 * 2,
                           color="gold",
                           zorder=2)
        ax.add_patch(patch)


class Segment(Element):
    """
    Segment of a particle accelerator consisting of several elements.

    Parameters
    ----------
    ocelot_cell : list
        List of ocelot elements that describe an accelerator (section).
    name : string, optional
        Unique identifier of the element.
    """

    def __init__(self, ocelot_cell, name=None):
        self.elements = [ocelot2joss(element) for element in ocelot_cell]
        for element in self.elements:
            self.__dict__[element.name] = element
        
        super().__init__(name=name)
    
    @property
    def transfer_map(self):
        transfer_map = np.eye(7)
        for element in self.elements:
            transfer_map = np.matmul(element.transfer_map, transfer_map)
        return transfer_map
    
    def split(self, resolution):
        return [split_element for element in self.elements
                              for split_element in element.split(resolution)]
    
    def plot(self, ax, s):
        element_lengths = [element.length for element in self.elements]
        element_ss = [0] + [sum(element_lengths[:i+1]) for i, _ in enumerate(element_lengths)]

        ax.plot([0, element_ss[-1]], [0, 0], "--", color="black")

        for element, s in zip(self.elements, element_ss[:-1]):
            element.plot(ax, s)

        ax.set_ylim(-1, 1)
        ax.set_xlabel("s (m)")
        ax.set_yticks([])
        ax.grid()
    
    def plot_reference_particle_traces(self, axx, axy, particles=None, n=10, resolution=0.01):
        """
        Plot `n` reference particles along the segment view in x- and y-direction.

        Parameters
        ----------
        axx : matplotlib.axes.Axes
            Axes to plot the particle traces into viewed in x-direction.
        axy : matplotlib.axes.Axes
            Axes to plot the particle traces into viewed in y-direction.
        particles : numpy.ndarray, optional
            Entering particles from which the reference particles are sampled.
        n : int, optional
            Number of reference particles to plot. Must not be larger than number of particles
            passed in `particles`.
        resolution : float, optional
            Minimum resolution of the tracking of the reference particles in the plot.
        """
        splits = self.split(resolution)

        split_lengths = [split.length for split in splits]
        ss = [0] + [sum(split_lengths[:i+1]) for i, _ in enumerate(split_lengths)]

        if particles is None:
            particles = generate_particles(n=n)
        references = np.zeros((len(ss), n, particles.shape[1]))
        references[0] = particles[np.random.choice(len(particles), n, replace=False)]
        for i, split in enumerate(splits):
            references[i+1] = split(references[i])
        
        for particle in range(references.shape[1]):
            axx.plot(ss, references[:,particle,0])
        axx.set_xlabel("s (m)")
        axx.set_ylabel("x (m)")
        axx.grid()

        for particle in range(references.shape[1]):
            axy.plot(ss, references[:,particle,2])
        axx.set_xlabel("s (m)")
        axy.set_ylabel("y (m)")
        axy.grid()
    
    def plot_overview(self, particles=None, n=10, resolution=0.01):
        """
        Plot an overview of the segment with the lattice and traced reference particles.

        Parameters
        ----------
        particles : numpy.ndarray, optional
            Entering particles from which the reference particles are sampled.
        n : int, optional
            Number of reference particles to plot. Must not be larger than number of particles
            passed in `particles`.
        resolution : float, optional
            Minimum resolution of the tracking of the reference particles in the plot.
        """
        fig = plt.figure()
        gs = fig.add_gridspec(3, hspace=0, height_ratios=[2,2,1])
        axs = gs.subplots(sharex=True)

        axs[0].set_title("Reference Particle Traces")
        self.plot_reference_particle_traces(axs[0], axs[1], particles, n, resolution)

        self.plot(axs[2], 0)

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        start = f"{self.__class__.__name__}(["

        s = start + self.elements[0].__repr__()
        x = ["\n" + (" " * len(start)) + element.__repr__() for element in self.elements[1:]]
        s += "".join(x)
        s += "])"

        return s
