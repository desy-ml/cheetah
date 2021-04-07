from itertools import chain

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import constants

from joss.utils import ocelot2joss


ELEMENT_COUNT = 0
REST_ENERGY = constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge

ELEMENT_ZORDER = 3
DRIFT_ZORDER = 2
        

class Element:

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
        return np.matmul(particles, self.transfer_map.transpose())
    
    def split(self, resolution):
        raise NotImplementedError
    
    def plot(self, ax, s):
        raise NotImplementedError


class Drift(Element):

    def __init__(self, length, energy=1e+8, name=None):
        """Create the transfer matrix of a drift section of given length."""
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
        ax.plot([s, s+self.length], [0, 0], color="black", zorder=DRIFT_ZORDER)


class Quadrupole(Element):

    def __init__(self, length, k1, energy=1e+8, name=None):
        """Create the transfer matrix of a quadrupole magnet of the given parameters."""
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
        patch = Rectangle((s, -0.4), self.length, 0.8, color="tab:red", zorder=ELEMENT_ZORDER)
        ax.add_patch(patch)


class HorizontalCorrector(Element):

    def __init__(self, length, angle, energy=1e+8, name=None):
        """Create the transfer matrix of a horizontal corrector magnet of the given parameters."""
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
        patch = Rectangle((s, -0.4), self.length, 0.8, color="tab:blue", zorder=ELEMENT_ZORDER)
        ax.add_patch(patch)


class VerticalCorrector(Element):

    def __init__(self, length, angle, energy=1e+8, name=None):
        """Create the transfer matrix of a vertical corrector magnet of the given parameters."""
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
        patch = Rectangle((s, -0.4), self.length, 0.8, color="tab:cyan", zorder=ELEMENT_ZORDER)
        ax.add_patch(patch)


class Segment(Element):

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

    def plot_reference_particles(self, particles, n=10, resolution=0.01):
        splits = self.split(resolution)

        split_lengths = [split.length for split in splits]
        ss = [0] + [sum(split_lengths[:i+1]) for i, _ in enumerate(split_lengths)]

        references = np.zeros((len(ss), n, particles.shape[1]))
        references[0] = particles[np.random.choice(len(particles), n, replace=False)]
        for i, split in enumerate(splits):
            references[i+1] = split(references[i])

        fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True)
        
        for particle in range(references.shape[1]):
            ax0.plot(ss, references[:,particle,0])
        ax0.set_xlabel("s [m]")
        ax0.set_ylabel("x [m]")
        ax0.grid()

        for particle in range(references.shape[1]):
            ax1.plot(ss, references[:,particle,2])
        ax1.set_xlabel("s [m]")
        ax1.set_ylabel("y [m]")
        ax1.grid()

        element_lengths = [element.length for element in self.elements]
        element_ss = [0] + [sum(element_lengths[:i+1]) for i, _ in enumerate(element_lengths[:-1])]
        for element, s in zip(self.elements, element_ss):
            element.plot(ax2, s)
        ax2.set_ylim(-0.5, 0.5)
        ax2.grid()

        plt.tight_layout()
        plt.show()
