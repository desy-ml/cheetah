import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

from joss.utils import ocelot2joss


ELEMENT_COUNT = 0
REST_ENERGY = constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
        

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
    
    def plot_reference_particles(self, particles, n=10):
        lengths = [element.length for element in self.elements]
        ss = [0] + [sum(lengths[:i+1]) for i, _ in enumerate(lengths)]

        references = np.zeros((len(ss), n, particles.shape[1]))
        references[0] = particles[np.random.choice(len(particles), n, replace=False)]
        for i, element in enumerate(self.elements):
            references[i+1] = element(references[i])

        plt.subplot(211)
        for particle in range(references.shape[1]):
            plt.plot(ss, references[:,particle,0])
        plt.xlabel("s [m]")
        plt.ylabel("x [m]")
        plt.grid()

        plt.subplot(212)
        for particle in range(references.shape[1]):
            plt.plot(ss, references[:,particle,2])
        plt.xlabel("s [m]")
        plt.ylabel("y [m]")
        plt.grid()

        plt.tight_layout()
        plt.show()
