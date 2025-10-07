import copy
import math
import random

import numpy as np
import scipy.integrate


def get_lorentz_factors(rest_energy: float, kin_energy: float) -> tuple[float, float]:
    gamma = 1.0 + (kin_energy / rest_energy)
    beta = math.sqrt(1.0 - (1.0 / gamma) ** 2)
    return (gamma, beta)


def get_perveance(line_density: float, rest_energy: float, kin_energy: float) -> float:
    gamma, beta = get_lorentz_factors(rest_energy, kin_energy)
    classical_proton_radius = 1.53469e-18
    return (2.0 * classical_proton_radius * line_density) / (beta**2 * gamma**3)


class LinearLattice:
    """Represents linear, uncoupled lattice with focusing strengths kx, ky."""

    def __init__(self, length: float) -> None:
        self.length = length

    def kx(self, s: float) -> float:
        return 0.0

    def ky(self, s: float) -> float:
        return 0.0


class DriftLattice(LinearLattice):
    """Represents drift."""

    def __init__(self, length: float) -> None:
        super().__init__(length)


class FODOLattice(LinearLattice):
    """Represents linear FODO lattice."""

    def __init__(self, length: float, kq: float, fill_frac: float = 0.5) -> None:
        super().__init__(length)
        self.kq = kq
        self.fill_frac = fill_frac

    def kx(self, s: float) -> float:
        s = (s % self.length) / self.length
        delta = self.fill_frac * 0.25
        if s < delta or s > 1 - delta:
            return +self.kq
        elif (0.5 - delta) <= s < (0.5 + delta):
            return -self.kq
        return 0.0

    def ky(self, s: float) -> float:
        return -self.kx(s)


class KVEnvelope:
    """Represents Kapchinskij-Vladimirskij (KV) distribution."""

    def __init__(
        self,
        params: np.ndarray,
        eps_x: float,
        eps_y: float,
        rest_energy: float,
        kin_energy: float,
        line_density: float = 0.0,
    ) -> None:
        self.params = np.array(params)
        self.eps_x = eps_x
        self.eps_y = eps_y

        self.line_density = line_density
        self.rest_energy = rest_energy
        self.kin_energy = kin_energy
        self.perveance = get_perveance(line_density, rest_energy, kin_energy)

    def set_params(self, params: np.ndarray) -> None:
        self.params = params

    def set_line_density(self, line_density: float) -> None:
        self.line_density = line_density
        self.perveance = get_perveance(line_density, self.rest_energy, self.kin_energy)

    def copy(self):
        return copy.deepcopy(self)

    def cov(self) -> np.ndarray:
        (cx, cxp, cy, cyp) = self.params

        cov_matrix = np.zeros((4, 4))
        cov_matrix[0, 0] = 0.25 * cx**2
        cov_matrix[2, 2] = 0.25 * cy**2
        cov_matrix[1, 1] = 0.25 * cxp**2 + 4.0 * (self.eps_x / cx) ** 2
        cov_matrix[3, 3] = 0.25 * cyp**2 + 4.0 * (self.eps_y / cy) ** 2
        cov_matrix[0, 1] = cov_matrix[1, 0] = 0.25 * cx * cxp
        cov_matrix[2, 3] = cov_matrix[3, 2] = 0.25 * cy * cyp
        return cov_matrix

    def set_cov(self, cov_matrix: np.ndarray) -> None:
        self.eps_x = np.sqrt(np.linalg.det(cov_matrix[0:2, 0:2]))
        self.eps_y = np.sqrt(np.linalg.det(cov_matrix[2:4, 2:4]))
        cx = np.sqrt(4.0 * cov_matrix[0, 0])
        cy = np.sqrt(4.0 * cov_matrix[2, 2])
        cxp = 2.0 * cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0])
        cyp = 2.0 * cov_matrix[2, 3] / np.sqrt(cov_matrix[2, 2])
        self.set_params([cx, cxp, cy, cyp])

    def sample(self, n: int) -> np.ndarray:
        x = np.random.normal(size=(n, 4))
        x /= np.linalg.norm(x, axis=1)[:, None]
        x /= np.std(x, axis=0)

        S = self.cov()
        L = np.linalg.cholesky(S)
        x = np.matmul(x, L.T)
        return x


def kv_envelope_derivatives(
    params: np.ndarray,
    s: float,
    lattice: LinearLattice,
    envelope: KVEnvelope,
) -> np.ndarray:
    """Return derivatives of envelope parameters with respect to s.."""
    rx = params[0]
    ry = params[2]
    rxp = params[1]
    ryp = params[3]

    kx = lattice.kx(s)
    ky = lattice.ky(s)
    Q = envelope.perveance
    eps_x = envelope.eps_x
    eps_y = envelope.eps_y

    dvec = np.zeros(4)
    dvec[0] = rxp
    dvec[2] = ryp
    dvec[1] = -kx * rx + 2.0 * Q / (rx + ry) + 16.0 * eps_x**2 / (rx**3)
    dvec[3] = -ky * ry + 2.0 * Q / (rx + ry) + 16.0 * eps_y**2 / (ry**3)
    return dvec


class KVEnvelopeTracker:
    """Tracks KV distribution envelope."""

    def __init__(self, lattice: LinearLattice, positions: np.ndarray) -> None:
        self.lattice = lattice
        self.positions = positions

    def track(self, env: KVEnvelope, **odeint_kws) -> dict[float, np.ndarray]:
        """Track envelope through lattice."""
        odeint_kws.setdefault("rtol", 1e-12)

        params_list = scipy.integrate.odeint(
            kv_envelope_derivatives,
            env.params,
            self.positions,
            args=(self.lattice, env),
            **odeint_kws,
        )
        env.set_params(params_list[-1])

        history = {}
        history["s"] = np.copy(self.positions)
        history["rx"] = params_list[:, 0]
        history["ry"] = params_list[:, 2]
        history["rxp"] = params_list[:, 1]
        history["ryp"] = params_list[:, 3]
        return history
