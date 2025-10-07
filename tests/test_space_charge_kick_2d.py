import copy
import math

import numpy as np
import torch
import scipy.integrate
from scipy import constants

import cheetah
from cheetah import Segment
from cheetah import SpaceChargeKick2D


def get_lorentz_factors(rest_energy: float, kin_energy: float) -> tuple[float, float]:
    gamma = 1.0 + (kin_energy / rest_energy)
    beta = math.sqrt(1.0 - (1.0 / gamma) ** 2)
    return (gamma, beta)


def get_perveance(line_density: float, rest_energy: float, kin_energy: float) -> float:
    gamma, beta = get_lorentz_factors(rest_energy, kin_energy)
    classical_proton_radius = 1.53469e-18
    return (2.0 * classical_proton_radius * line_density) / (beta**2 * gamma**3)


def add_space_charge_elements(segment: Segment, **kwargs) -> Segment:
    new_elements = []
    for element in segment.elements:
        sc_kick = SpaceChargeKick2D(element.length, **kwargs)
        new_elements.append(sc_kick)
        new_elements.append(element)
    return Segment(new_elements)


def split_segment(segment: Segment, n: int) -> Segment:
    slice_length = segment.length / float(n)
    elements = segment.split(resolution=slice_length)
    return Segment(elements)


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
    """Return derivatives of KV envelope parameters with respect to s."""
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


def test_kv_drift():
    """
    Test that tracks a KV distribution through a drift and compares the final rms beam
    sizes to the KV envelope equations (integrated using `scipy.integrate.odeint`).
    """
    cfg = {
        "beam": {
            "length": 10.0,
            "kin_energy": 1e6,
            "intensity": 1e10,
            "rx": 0.010,
            "ry": 0.010,
            "rxp": 0.001,
            "ryp": 0.001,
            "eps_x": 0.01e-6,
            "eps_y": 0.01e-6,
        },
        "lattice": {
            "length": 5.0,
        },
    }

    data = {}
    for key1 in ["pic", "env"]:
        data[key1] = {}
        for key2 in ["cov_matrix"]:
            data[key1][key2] = []

    # Track envelope
    # ----------------------------------------------------------------------------------
    rest_energy = 0.938272029e09  # [eV]
    kin_energy = cfg["beam"]["kin_energy"]  # [eV]
    energy = kin_energy + rest_energy

    line_density = cfg["beam"]["intensity"] / cfg["beam"]["length"]

    env = KVEnvelope(
        params=[
            cfg["beam"]["rx"],
            cfg["beam"]["rxp"],
            cfg["beam"]["ry"],
            cfg["beam"]["ryp"],
        ],
        eps_x=cfg["beam"]["eps_x"],
        eps_y=cfg["beam"]["eps_y"],
        line_density=line_density,
        rest_energy=rest_energy,
        kin_energy=kin_energy,
    )
    env_init = env.copy()

    data["env"]["cov_matrix"].append(torch.tensor(env.cov()).float())

    env_lattice = DriftLattice(length=cfg["lattice"]["length"])
    env_positions = np.linspace(0.0, cfg["lattice"]["length"], 500)
    env_tracker = KVEnvelopeTracker(env_lattice, env_positions)
    env_history = env_tracker.track(env)

    data["env"]["cov_matrix"].append(torch.tensor(env.cov()).float())

    # Track beam
    # ----------------------------------------------------------------------------------
    length = torch.as_tensor(cfg["lattice"]["length"])
    segment = cheetah.Segment([cheetah.Drift(length)])
    segment = split_segment(segment, 100)
    segment = add_space_charge_elements(
        segment,
        grid_shape=(64, 64),
        grid_extent_x=torch.tensor(3.0),
        grid_extent_y=torch.tensor(3.0),
    )
    n = 128_000

    particles = torch.randn((n, 7))
    particles[:, :4] = torch.tensor(env_init.sample(n))
    particles[:, 4] = cfg["beam"]["length"] * (torch.rand(n) - 0.5)
    particles[:, 5] = torch.ones(n) * 0.0
    particles[:, 6] = torch.ones(n)

    beam_intensity = torch.tensor(cfg["beam"]["intensity"])
    beam_charge = constants.elementary_charge * beam_intensity
    particle_charge = beam_charge / particles.shape[0]

    beam = cheetah.ParticleBeam(
        particles=particles,
        energy=torch.as_tensor(energy),
        species=cheetah.Species("proton"),
        particle_charges=particle_charge,
    )

    data["pic"]["cov_matrix"].append(torch.cov(beam.particles[:, :4].T))
    beam = segment.track(beam)
    data["pic"]["cov_matrix"].append(torch.cov(beam.particles[:, :4].T))

    # Analysis
    # ----------------------------------------------------------------------------------
    for i in range(2):
        cov_matrix_a = 1e6 * data["env"]["cov_matrix"][i]
        cov_matrix_b = 1e6 * data["pic"]["cov_matrix"][i]
        scale_a = torch.sqrt(torch.diag(cov_matrix_a))
        scale_b = torch.sqrt(torch.diag(cov_matrix_b))
        assert torch.all(torch.isclose(scale_a, scale_b, rtol=0.001))


def test_vectorized():
    """
    Tests that the space charge kick can be applied to a vectorized beam.
    """

    # Simulation parameters
    section_length = torch.tensor(0.42)
    R0 = torch.tensor(0.001)
    energy = torch.tensor(2.5e8)
    rest_energy = torch.tensor(
        constants.electron_mass
        * constants.speed_of_light**2
        / constants.elementary_charge
    )
    gamma = energy / rest_energy
    beta = torch.sqrt(1 - 1 / gamma**2)

    incoming = cheetah.ParticleBeam.uniform_3d_ellipsoid(
        num_particles=10_000,
        total_charge=torch.tensor([[1e-9, 2e-9], [3e-9, 4e-9], [5e-9, 6e-9]]),
        energy=energy.expand([3, 2]),
        radius_x=R0.expand([3, 2]),
        radius_y=R0.expand([3, 2]),
        radius_tau=R0.expand([3, 2]) / gamma / beta,
        # Duration of the beam in the lab frame
        sigma_px=torch.tensor(1e-15).expand([3, 2]),
        sigma_py=torch.tensor(1e-15).expand([3, 2]),
        sigma_p=torch.tensor(1e-15).expand([3, 2]),
    )

    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(section_length / 6),
            cheetah.SpaceChargeKick2D(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick2D(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick2D(section_length / 3),
            cheetah.Drift(section_length / 6),
        ]
    )

    outgoing = segment.track(incoming)

    assert outgoing.particles.shape == (3, 2, 10_000, 7)


def test_incoming_beam_not_modified():
    """
    Tests that the incoming beam is not modified when calling the track method.
    """
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000, sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
    )
    # Initial beam properties
    incoming_beam_before = incoming_beam.particles

    section_length = torch.tensor(1.0)
    segment_space_charge = cheetah.Segment(
        elements=[
            cheetah.Drift(section_length / 6),
            cheetah.SpaceChargeKick2D(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick2D(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick2D(section_length / 3),
            cheetah.Drift(section_length / 6),
        ]
    )
    # Calling the track method
    segment_space_charge.track(incoming_beam)

    # Final beam properties
    incoming_beam_after = incoming_beam.particles

    assert torch.allclose(incoming_beam_before, incoming_beam_after)


def test_does_not_break_segment_length():
    """
    Test that the computation of a `Segment`'s length does not break when
    `SpaceChargeKick2D` is used.
    """
    section_length = torch.tensor(1.0)
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(section_length / 6),
            cheetah.SpaceChargeKick2D(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick2D(section_length / 3),
            cheetah.Drift(section_length / 3),
            cheetah.SpaceChargeKick2D(section_length / 3),
            cheetah.Drift(section_length / 6),
        ]
    )

    assert segment.length.shape == torch.Size([])
    assert torch.allclose(segment.length, torch.tensor(1.0))


def test_space_charge_with_ares_astra_beam():
    """
    Tests running space charge through a 1m drift with an Astra beam from the ARES
    linac. This test is added because running this code would throw an error:
    `IndexError: index -38 is out of bounds for dimension 3 with size 32`.
    """
    segment = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.SpaceChargeKick2D(effect_length=torch.tensor(1.0)),
        ]
    )
    beam = cheetah.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    _ = segment.track(beam)


def test_space_charge_with_aperture_cutoff():
    """
    Tests that the space charge kick is correctly applied only to surviving particles,
    by comparing the results with and without an aperture that results in beam losses.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.2)),
            cheetah.Aperture(
                x_max=torch.tensor(1e-4),
                y_max=torch.tensor(1e-4),
                shape="rectangular",
                is_active=False,
                name="aperture",
            ),
            cheetah.Drift(length=torch.tensor(0.25)),
            cheetah.SpaceChargeKick2D(effect_length=torch.tensor(0.5)),
            cheetah.Drift(length=torch.tensor(0.25)),
        ]
    )
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        total_charge=torch.tensor(1e-9),
        mu_x=torch.tensor(5e-5),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
    )

    # Track with inactive aperture
    outgoing_beam_without_aperture = segment.track(incoming_beam)

    # Activate the aperture and track the beam
    segment.aperture.is_active = True
    outgoing_beam_with_aperture = segment.track(incoming_beam)

    # Check that with particle loss the space charge kick is different
    assert not torch.allclose(
        outgoing_beam_with_aperture.particles, outgoing_beam_without_aperture.particles
    )
    # Check that the number of surviving particles is less than the initial number
    assert outgoing_beam_with_aperture.survival_probabilities.sum(dim=-1).max() < 10_000
