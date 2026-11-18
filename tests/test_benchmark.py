from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import cheetah
from cheetah.utils.bmadx import bmad_to_cheetah_coords, cheetah_to_bmad_coords

from .resources import ARESlatticeStage3v1_9 as ares

if TYPE_CHECKING:
    from pytao import Tao

COORDINATES = ("x", "px", "y", "py", "z", "pz")
TWISS_PARAMETERS = {
    "beta_x": "beta_a",
    "alpha_x": "alpha_a",
    "beta_y": "beta_b",
    "alpha_y": "alpha_b",
}


def load_standalone_cavity() -> tuple[Tao, cheetah.Segment, cheetah.ParameterBeam]:
    """Load matching models from the standalone Bmad fixture without `LCLS_LATTICE`.

    :return: Tao, a Cheetah cavity with inactive end drifts, and entrance optics.
    """
    pytao = pytest.importorskip("pytao")
    lattice = Path(__file__).resolve().parent / "resources/bmad/l0a_cavity.bmad"
    tao = pytao.Tao(lattice_file=lattice, noinit=True, nostartup=True, noplot=True)
    attributes = tao.ele_gen_attribs("L0A")
    end_drift = (attributes["L"] - attributes["L_ACTIVE"]) / 2.0
    cavity = cheetah.Cavity(
        length=torch.tensor(attributes["L_ACTIVE"], dtype=torch.float64),
        voltage=torch.tensor(attributes["VOLTAGE"], dtype=torch.float64),
        frequency=torch.tensor(attributes["RF_FREQUENCY"], dtype=torch.float64),
        phase=torch.tensor(-360.0 * attributes["PHI0"], dtype=torch.float64),
        cavity_type="traveling_wave",
        name="L0A",
    )
    model = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(end_drift, dtype=torch.float64)),
            cavity,
            cheetah.Drift(length=torch.tensor(end_drift, dtype=torch.float64)),
        ]
    )
    twiss = tao.ele_twiss("BEGINNING")
    incoming = cheetah.ParameterBeam.from_twiss(
        **{
            name: torch.tensor(twiss[bmad_name], dtype=torch.float64)
            for name, bmad_name in TWISS_PARAMETERS.items()
        },
        energy=torch.tensor(attributes["E_TOT_START"], dtype=torch.float64),
        dtype=torch.float64,
    )
    return tao, model, incoming


def track_bmad(tao: Tao, particles: torch.Tensor) -> np.ndarray:
    """Track each six-vector from BEGINNING to the isolated L0A exit."""
    outgoing = []
    for particle in particles:
        tao.cmd("set global lattice_calc_on = F")
        for name, value in zip(COORDINATES, particle.tolist(), strict=True):
            tao.cmd(f"set particle_start {name} = {value:.17g}")
        tao.cmd("set global lattice_calc_on = T")
        orbit = tao.ele_orbit("L0A")
        if orbit["state"] != "Alive":
            raise RuntimeError(f"Bmad lost input particle {particle.tolist()}: {orbit}")
        outgoing.append([orbit[name] for name in COORDINATES])
    return np.asarray(outgoing)


def compare_twiss(
    tao: Tao, cavity: cheetah.Element, incoming: cheetah.ParameterBeam
) -> dict[str, dict[str, float]]:
    """Compare exit beta/alpha about the reference orbit in both cavity models.

    :param tao: Isolated Bmad lattice with the current L0A phase setting.
    :param cavity: Matching Cheetah model, including inactive end drifts.
    :param incoming: Uncoupled beam with the same entrance beta/alpha as Bmad.
    :return: Bmad, Cheetah, and signed residual values for each Twiss parameter.
    """
    # Bmad recalculates its linear optics about the last tracked particle
    track_bmad(tao, incoming.energy.new_zeros((1, 6)))
    bmad_twiss = tao.ele_twiss("L0A")
    with torch.no_grad():
        outgoing = cavity.track(incoming)
    comparison = {}
    for name, bmad_name in TWISS_PARAMETERS.items():
        bmad_value = bmad_twiss[bmad_name]
        cheetah_value = getattr(outgoing, name).item()
        if not np.isfinite([bmad_value, cheetah_value]).all():
            raise RuntimeError(f"Non-finite exit Twiss parameter: {name}")
        if name.startswith("beta") and min(bmad_value, cheetah_value) <= 0.0:
            raise RuntimeError(f"Non-positive exit Twiss parameter: {name}")
        comparison[name] = {
            "bmad": bmad_value,
            "cheetah": cheetah_value,
            "residual": cheetah_value - bmad_value,
        }
    return comparison


def compare_transport_matrices(
    tao: Tao, cavity: cheetah.Element, incoming: cheetah.ParameterBeam
) -> dict[str, np.ndarray]:
    """Compare Bmad's matrix to Cheetah's full tracking Jacobian at the reference.

    :param tao: Isolated Bmad lattice with the current L0A phase setting.
    :param cavity: Matching Cheetah model, including inactive end drifts.
    :param incoming: Beam supplying the entrance reference energy and species.
    :return: Bmad, Cheetah Jacobian, and residual 6x6 matrices in Bmad coordinates.
    """
    track_bmad(tao, incoming.energy.new_zeros((1, 6)))
    rows = tao.ele_mat6("L0A")
    bmad_matrix = np.stack([rows[str(index)] for index in range(1, 7)])
    p0c = (incoming.energy.square() - incoming.species.mass_eV.square()).sqrt()

    def track_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
        """Track a single Bmad six-vector through the full Cheetah cavity model."""
        particles, energy = bmad_to_cheetah_coords(
            coordinates.unsqueeze(0), p0c, incoming.species.mass_eV
        )
        beam = cheetah.ParticleBeam(
            particles=particles,
            energy=energy,
            species=incoming.species,
            dtype=particles.dtype,
            device=particles.device,
        )
        outgoing = cavity.track(beam)
        result, _ = cheetah_to_bmad_coords(
            outgoing.particles, outgoing.energy, incoming.species.mass_eV
        )
        return result.squeeze(0)

    cheetah_matrix = torch.autograd.functional.jacobian(
        track_coordinates, incoming.energy.new_zeros(6)
    ).numpy()
    if bmad_matrix.shape != (6, 6) or cheetah_matrix.shape != (6, 6):
        raise ValueError("Expected unvectorized 6x6 transport matrices")
    if not (np.isfinite(bmad_matrix).all() and np.isfinite(cheetah_matrix).all()):
        raise RuntimeError("Non-finite first-order transport matrix")
    return {
        "bmad": bmad_matrix,
        "cheetah": cheetah_matrix,
        "residual": cheetah_matrix - bmad_matrix,
    }


@pytest.fixture
def isolated_cavity():
    """Create matching low-energy accelerating cavities for optics comparisons."""
    return load_standalone_cavity()


@pytest.mark.parametrize("phase_offset", [-20.0, -10.0, 0.0, 10.0, 20.0])
def test_benchmark_bmad_cavity(benchmark, isolated_cavity, phase_offset):
    """Benchmark Bmad R-matrix versus Cheetah tracking-Jacobian transverse agreement.

    """
    tao, model, incoming = isolated_cavity
    model.L0A.phase = model.L0A.phase - phase_offset
    tao.cmd(f"set ele L0A phi0 = {-model.L0A.phase.item() / 360.0:.17g}")
    matrices = benchmark(compare_transport_matrices, tao, model, incoming)
    torch.testing.assert_close(
        torch.from_numpy(matrices["cheetah"][:4, :4]),
        torch.from_numpy(matrices["bmad"][:4, :4]),
        atol=1e-8,
        rtol=1e-8,
    )


@pytest.mark.parametrize("phase_offset", [-20.0, -10.0, 0.0, 10.0, 20.0])
def test_transverse_tracking_matches_standalone_bmad(isolated_cavity, phase_offset):
    """Small transverse rays and exit Twiss must agree with standalone Bmad."""
    tao, model, incoming = isolated_cavity
    model.L0A.phase = model.L0A.phase - phase_offset
    tao.cmd(f"set ele L0A phi0 = {-model.L0A.phase.item() / 360.0:.17g}")
    probes = torch.zeros((9, 6), dtype=torch.float64)
    probes[1:5, :4] = torch.diag(
        torch.tensor([1e-6, 1e-7, 1e-6, 1e-7], dtype=torch.float64)
    )
    probes[5:, :4] = -probes[1:5, :4]
    p0c = (incoming.energy.square() - incoming.species.mass_eV.square()).sqrt()
    particles, energy = bmad_to_cheetah_coords(probes, p0c, incoming.species.mass_eV)
    outgoing = model.track(
        cheetah.ParticleBeam(
            particles=particles, energy=energy, species=incoming.species
        )
    )
    actual, _ = cheetah_to_bmad_coords(
        outgoing.particles, outgoing.energy, outgoing.species.mass_eV
    )
    expected = track_bmad(tao, probes)
    np.testing.assert_allclose(
        actual.numpy()[:, :4], expected[:, :4], atol=1e-10, rtol=1e-6
    )
    for values in compare_twiss(tao, model, incoming).values():
        np.testing.assert_allclose(
            values["cheetah"], values["bmad"], atol=1e-7, rtol=1e-7
        )


def test_twiss_comparison_resets_bmad_reference_orbit(isolated_cavity):
    """Prior off-axis probes must not change the reported reference Twiss optics."""
    tao, cavity, incoming = isolated_cavity
    reference = compare_twiss(tao, cavity, incoming)
    tao.cmd("set particle_start x = 0.001")
    tao.cmd("set particle_start pz = 0.02")
    comparison = compare_twiss(tao, cavity, incoming)
    assert set(comparison) == {"beta_x", "alpha_x", "beta_y", "alpha_y"}
    for name, bmad_name in TWISS_PARAMETERS.items():
        values = comparison[name]
        np.testing.assert_allclose(
            list(values.values()), list(reference[name].values())
        )
        assert values["bmad"] == tao.ele_twiss("L0A")[bmad_name]
        assert values["residual"] == values["cheetah"] - values["bmad"]
        np.testing.assert_allclose(
            values["cheetah"], values["bmad"], atol=1e-7, rtol=1e-7
        )
    assert tao.ele_orbit("BEGINNING")["pz"] == 0.0


@pytest.mark.parametrize("phase", [-18.9, 1.1, 21.1])
def test_bmad_transport_matrices_use_full_tracking_jacobian(isolated_cavity, phase):
    """Check both matrices against full tracking finite differences off crest."""
    tao, cavity, incoming = isolated_cavity
    cavity.L0A.phase = torch.tensor(phase, dtype=torch.float64)
    tao.cmd(f"set ele L0A phi0 = {-phase / 360.0:.17g}")
    tao.cmd("set particle_start pz = 0.02")
    matrices = compare_transport_matrices(tao, cavity, incoming)
    p0c = (incoming.energy.square() - incoming.species.mass_eV.square()).sqrt()
    step = 1e-7
    probes = torch.eye(6, dtype=torch.float64) * step
    probes = torch.cat((probes, -probes))
    particles, energy = bmad_to_cheetah_coords(probes, p0c, incoming.species.mass_eV)
    beam = cheetah.ParticleBeam(
        particles=particles,
        energy=energy,
        species=incoming.species,
        dtype=torch.float64,
    )
    outgoing = cavity.track(beam)
    cheetah_output, _ = cheetah_to_bmad_coords(
        outgoing.particles, outgoing.energy, incoming.species.mass_eV
    )
    for model, tracked in (
        ("bmad", track_bmad(tao, probes)),
        ("cheetah", cheetah_output.numpy()),
    ):
        finite_difference = ((tracked[:6] - tracked[6:]) / (2.0 * step)).T
        np.testing.assert_allclose(
            matrices[model], finite_difference, atol=1e-8, rtol=1e-5
        )
    np.testing.assert_array_equal(
        matrices["residual"], matrices["cheetah"] - matrices["bmad"]
    )
    np.testing.assert_allclose(
        matrices["cheetah"][:4, :4], matrices["bmad"][:4, :4], atol=1e-8, rtol=1e-8
    )


@pytest.mark.filterwarnings(
    r"ignore:.*Invalid tracking method.*:cheetah.utils.PhysicsWarning"
)
@pytest.mark.filterwarnings(
    r"ignore:.*default screen properties.*:cheetah.utils.DefaultParameterWarning"
)
@pytest.mark.parametrize(
    ("beam_cls", "tracking_method"),
    [
        (cheetah.ParticleBeam, "linear"),
        (cheetah.ParticleBeam, "second_order"),
        (cheetah.ParticleBeam, "drift_kick_drift"),
        pytest.param(
            cheetah.ParameterBeam,
            "linear",
            marks=pytest.mark.filterwarnings(
                r"ignore:.*Aperture tracking.*:cheetah.utils.PhysicsWarning"
            ),
        ),
    ],
)
def test_benchmark_ares_lattice(benchmark, beam_cls, tracking_method):
    """Benchmark for tracking through the ARES lattice with fixed beam."""
    incoming = beam_cls.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    segment = cheetah.Segment.from_ocelot(ares.cell)
    segment.set_attrs_on_every_element(tracking_method=tracking_method, num_steps=5)

    benchmark(segment.track, incoming=incoming)
