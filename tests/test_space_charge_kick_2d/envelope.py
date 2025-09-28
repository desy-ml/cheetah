import numpy as np
import scipy.integrate


class LinearLattice:
    def __init__(self, length: float) -> None:
        self.length = length

    def kx(self, s: float) -> float:
        return 0.0
    
    def ky(self, s: float) -> float:
        return 0.0
    

class DriftLattice(LinearLattice):
    def __init__(self, length: float) -> None:
        super().__init__(length)


class FODOLattice(LinearLattice):
    def __init__(self, length: float, kq: float) -> None:
        super().__init__(length)
        self.kq = kq

    def kx(self, s: float) -> float:
        s = (s % self.length) / self.length
        delta = 0.125
        if s < delta or s > 1 - delta:
            return +self.kq
        elif 0.5 - delta <= s < 0.5 + delta:
            return -self.kq
        return 0.0

    def ky(self, s: float) -> float:
        return -self.kx(s)

    
class KVEnvelopeTracker:
    def __init__(self, lattice: LinearLattice, perveance: float, eps_x: float, eps_y: float) -> None:
        self.lattice = lattice
        self.perveance = perveance
        self.eps_x = eps_x
        self.eps_y = eps_y

    def derivatives(self, params: np.ndarray, s: float) -> None:
        rx = params[0]
        ry = params[2]
        rxp = params[1]
        ryp = params[3]

        kx = self.lattice.kx(s)
        ky = self.lattice.ky(s)
        Q = self.perveance

        derivs = np.zeros(4)
        derivs[0] = rxp
        derivs[2] = ryp
        derivs[1] = -kx * rx + 2.0 * Q / (rx + ry) + 16.0 * self.eps_x**2 / (rx**3)
        derivs[3] = -ky * ry + 2.0 * Q / (rx + ry) + 16.0 * self.eps_y**2 / (ry**3)
        return derivs
    
    def track(self, params: np.ndarray, positions: np.ndarray) -> np.ndarray:
        params_list = scipy.integrate.odeint(self.derivatives, params, positions)
        sizes = params_list[:, (0, 2)]
        sizes = sizes * 1000.0
        return sizes
