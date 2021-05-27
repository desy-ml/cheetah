import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import ocelot as oc
from scipy import constants

from joss.particles import Beam
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
    
    Attributes
    ---------
    is_active : bool
        Is set to `True` when the element is in operation. May be defined differently for each type
        of element.
    """

    is_active = False

    def __init__(self, name=None):
        global ELEMENT_COUNT

        if name is not None:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}_{ELEMENT_COUNT:06d}"
        
        ELEMENT_COUNT += 1
    
    def transfer_map(self, energy):
        raise NotImplementedError

    def __call__(self, incoming):
        """
        Track particles through the element.
        
        Pramameters
        -----------
        incoming : joss.Beam
            Beam of particles entering the element.

        Returns
        -------
        joss.Beam
            Beam of particles exiting the element.
        """
        tm = self.transfer_map(incoming.energy)
        new_particles = np.matmul(incoming.particles, tm.transpose())
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

    def __init__(self, length, name=None):
        self.length = length

        super().__init__(name=name)
    
    def transfer_map(self, energy):
        gamma = energy / REST_ENERGY
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
            element = Drift(min(resolution, remaining))
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

    def __init__(self, length, k1=0.0, misalignment=(0,0), name=None):
        self.length = length
        self.k1 = k1
        self.misalignment = misalignment

        super().__init__(name=name)
    
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

        R = np.array([[            cx,        sx,         0,  0, 0,      dx / beta, 0],
                      [     -kx2 * sx,        cx,         0,  0, 0, sx * hx / beta, 0],
                      [             0,         0,        cy, sy, 0,              0, 0],
                      [             0,         0, -ky2 * sy, cy, 0,              0, 0],
                      [sx * hx / beta, dx / beta,         0,  0, 1,            r56, 0],
                      [             0,         0,         0,  0, 0,              1, 0],
                      [             0,         0,         0,  0, 0,              0, 1]])
        
        if self.misalignment[0] == 0 and self.misalignment[1] == 0:
            return R
        else:
            R_entry = np.array([[1, 0, 0, 0, 0, 0, self.misalignment[0]],
                                [0, 1, 0, 0, 0, 0,                    0],
                                [0, 0, 1, 0, 0, 0, self.misalignment[1]],
                                [0, 0, 0, 1, 0, 0,                    0],
                                [0, 0, 0, 0, 1, 0,                    0],
                                [0, 0, 0, 0, 0, 1,                    0],
                                [0, 0, 0, 0, 0, 0,                    1]])
            R_exit = np.array([[1, 0, 0, 0, 0, 0, -self.misalignment[0]],
                               [0, 1, 0, 0, 0, 0,                     0],
                               [0, 0, 1, 0, 0, 0, -self.misalignment[1]],
                               [0, 0, 0, 1, 0, 0,                     0],
                               [0, 0, 0, 0, 1, 0,                     0],
                               [0, 0, 0, 0, 0, 1,                     0],
                               [0, 0, 0, 0, 0, 0,                     1]])
            R = np.matmul(R_entry, R)
            R = np.matmul(R, R_exit)
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
                                 misalignment=self.misalignment)
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

    def __init__(self, length, angle=0.0, energy=1e+8, name=None):
        self.length = length
        self.angle = angle

        super().__init__(name=name)

    def transfer_map(self, energy):
        return np.array([[1, self.length, 0,           0, 0, 0,          0],
                         [0,           1, 0,           0, 0, 0, self.angle],
                         [0,           0, 1, self.length, 0, 0,          0],
                         [0,           0, 0,           1, 0, 0,          0],
                         [0,           0, 0,           0, 1, 0,          0],
                         [0,           0, 0,           0, 0, 1,          0],
                         [0,           0, 0,           0, 0, 0,          1]])
    
    @property
    def is_active(self):
        return self.angle != 0
    
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

    def __init__(self, length, angle=0.0, energy=1e+8, name=None):
        self.length = length
        self.angle = angle

        super().__init__(name=name)

    def transfer_map(self, energy):
        return np.array([[1, self.length, 0,           0, 0, 0,          0],
                         [0,           1, 0,           0, 0, 0,          0],
                         [0,           0, 1, self.length, 0, 0,          0],
                         [0,           0, 0,           1, 0, 0, self.angle],
                         [0,           0, 0,           0, 1, 0,          0],
                         [0,           0, 0,           0, 0, 1,          0],
                         [0,           0, 0,           0, 0, 0,          1]])
    
    @property
    def is_active(self):
        return self.angle != 0
    
    def split(self, resolution):
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = min(resolution, remaining)
            element = VerticalCorrector(length,
                                        self.angle * length / self.length)
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
    name : string, optional
        Unique identifier of the element.
    
    Notes
    -----
    Currently behaves like a drift section but is plotted distinctively.
    """

    def __init__(self, length, name=None):
        self.length = length

        super().__init__(name=name)
    
    def transfer_map(self, energy):
        gamma = energy / REST_ENERGY
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
            element = Cavity(min(resolution, remaining))
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
        return f"{self.__class__.__name__}(length={self.length:.2f}, name=\"{self.name}\")"


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
    """

    length = 0

    def transfer_map(self, energy):
        return np.eye(7)
    
    def __call__(self, incoming):
        return Beam(incoming.particles, incoming.energy)
    
    def split(self, resolution):
        return [self]
    
    def plot(self, ax, s):
        alpha = 1 if self.is_active else 0.2
        patch = Rectangle((s, -0.6),
                           0,
                           0.6,
                           color="darkkhaki",
                           alpha=alpha,
                           zorder=2)
        ax.add_patch(patch)


class Screen(Element):
    """
    Screen in a particle accelerator.

    Parameters
    ----------
    name : string, optional
        Unique identifier of the element.
    
    Attributes
    ---------
    is_active : bool
        Can be set by the user. Merely influences how the element is displayed in a lattice plot.
    """

    length = 0

    def transfer_map(self, energy):
        return np.eye(7)

    def __call__(self, incoming):
        return Beam(incoming.particles, incoming.energy)
    
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

    def __init__(self, length, name=None):
        self.length = length

        super().__init__(name=name)
    
    def transfer_map(self, energy):
        gamma = energy / REST_ENERGY
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
            element = Cavity(min(resolution, remaining))
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
        List of JOSS or Ocelot elements that describe an accelerator (section).
    name : string, optional
        Unique identifier of the element.
    """

    def __init__(self, cell, name=None):
        if isinstance(cell[0], Element):
            self.elements = cell
        elif isinstance(cell[0], oc.Element):
            self.elements = [ocelot2joss(element) for element in cell]
        else:
            raise ValueError("Parameter cell must be either list of JOSS or Ocelot elements.")
        
        for element in self.elements:
            self.__dict__[element.name] = element
        
        super().__init__(name=name)
    
    def transfer_map(self, energy):
        tm = np.eye(7)
        for element in self.elements:
            tm = np.matmul(element.transfer_map(energy), tm)
        return tm
    
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
    
    def plot_reference_particle_traces(self, axx, axy, beam=None, n=10, resolution=0.01):
        """
        Plot `n` reference particles along the segment view in x- and y-direction.

        Parameters
        ----------
        axx : matplotlib.axes.Axes
            Axes to plot the particle traces into viewed in x-direction.
        axy : matplotlib.axes.Axes
            Axes to plot the particle traces into viewed in y-direction.
        beam : joss.Beam, optional
            Entering beam from which the reference particles are sampled.
        n : int, optional
            Number of reference particles to plot. Must not be larger than number of particles
            passed in `particles`.
        resolution : float, optional
            Minimum resolution of the tracking of the reference particles in the plot.
        """
        splits = self.split(resolution)

        split_lengths = [split.length for split in splits]
        ss = [0] + [sum(split_lengths[:i+1]) for i, _ in enumerate(split_lengths)]

        references = []
        if beam is None:
            initial = Beam.make_linspaced(n=n)
            references.append(initial)
        else:
            initial = Beam.make_linspaced(n=n, mu_x=beam.mu_x, mu_xp=beam.mu_xp, mu_y=beam.mu_y,
                                          mu_yp=beam.mu_yp, sigma_x=beam.sigma_x,
                                          sigma_xp=beam.sigma_xp, sigma_y=beam.sigma_y,
                                          sigma_yp=beam.sigma_yp, sigma_s=beam.sigma_s,
                                          sigma_p=beam.sigma_p, energy=beam.energy)
            references.append(initial)
        for split in splits:
            sample = split(references[-1])
            references.append(sample)
        
        for particle_index in range(n):
            xs = [reference_beam.xs[particle_index] for reference_beam in references]
            axx.plot(ss, xs)
        axx.set_xlabel("s (m)")
        axx.set_ylabel("x (m)")
        axx.grid()

        for particle_index in range(n):
            ys = [reference_beam.ys[particle_index] for reference_beam in references]
            axy.plot(ss, ys)
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
        beam : joss.Beam, optional
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
        x = [",\n" + (" " * len(start)) + element.__repr__() for element in self.elements[1:]]
        s += "".join(x)
        s += "])"

        return s
