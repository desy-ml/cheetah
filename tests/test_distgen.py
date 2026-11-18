import pytest
import torch
import yaml

import cheetah

try:
    from distgen import Generator
except ImportError:
    Generator = None


pytestmark = pytest.mark.skipif(
    Generator is None,
    reason="distgen (https://github.com/ColwynGulliford/distgen) is not installed.",
)


def _base_config(start: dict, longitudinal: dict, include_pz: bool = True) -> dict:
    """Build a minimal `distgen` configuration dictionary for testing."""
    config = {
        "n_particle": 1000,
        "species": "electron",
        "total_charge": {"value": 100.0, "units": "pC"},
        "start": start,
        "r_dist": {
            "type": "radial_gaussian",
            "sigma_xy": {"value": 1.0, "units": "mm"},
        },
        **longitudinal,
    }
    if include_pz:
        config["pz_dist"] = {
            "type": "gaussian",
            "avg_pz": {"value": 10.0, "units": "MeV/c"},
            "sigma_pz": {"value": 0.1, "units": "MeV/c"},
        }
    return config


def test_from_distgen_dict():
    """Test creating a `ParticleBeam` from a `distgen` config dictionary."""
    config = _base_config(
        start={"type": "free"},
        longitudinal={
            "t_dist": {"type": "gaussian", "sigma_t": {"value": 2.0, "units": "ps"}}
        },
    )

    beam = cheetah.ParticleBeam.from_distgen(config, dtype=torch.float64)

    assert beam.num_particles == 1000
    assert beam.species.name == "electron"
    assert beam.particles.dtype == torch.float64
    torch.testing.assert_close(
        beam.total_charge, torch.tensor(-1e-10, dtype=torch.float64)
    )
    torch.testing.assert_close(
        beam.sigma_x, torch.tensor(1e-3, dtype=torch.float64), rtol=0.1, atol=0.0
    )
    torch.testing.assert_close(
        beam.sigma_y, torch.tensor(1e-3, dtype=torch.float64), rtol=0.1, atol=0.0
    )
    torch.testing.assert_close(
        beam.sigma_tau,
        torch.tensor(2e-12 * 299792458.0, dtype=torch.float64),
        rtol=0.1,
        atol=0.0,
    )


def test_from_distgen_time_start():
    """Test that `time` start (z-distributed) beams are drifted to populate `tau`."""
    config = _base_config(
        start={"type": "time", "tstart": {"value": 0.0, "units": "ps"}},
        longitudinal={
            "z_dist": {"type": "gaussian", "sigma_z": {"value": 1.0, "units": "mm"}}
        },
    )

    beam = cheetah.ParticleBeam.from_distgen(config, dtype=torch.float64)

    torch.testing.assert_close(
        beam.sigma_tau, torch.tensor(1e-3, dtype=torch.float64), rtol=0.1, atol=0.0
    )


def test_from_distgen_cathode_start():
    """Test that `cathode` start beams have `survival_probabilities` equal to 1.0."""
    config = _base_config(
        start={"type": "cathode", "MTE": {"value": 150.0, "units": "meV"}},
        longitudinal={
            "t_dist": {"type": "gaussian", "sigma_t": {"value": 2.0, "units": "ps"}}
        },
        include_pz=False,
    )

    beam = cheetah.ParticleBeam.from_distgen(config, dtype=torch.float64)

    torch.testing.assert_close(
        beam.survival_probabilities, torch.ones(1000, dtype=torch.float64)
    )


def test_from_distgen_yaml_path(tmp_path):
    """Test creating a `ParticleBeam` from a `distgen` YAML file (`Path` and `str`)."""
    config = _base_config(
        start={"type": "free"},
        longitudinal={
            "t_dist": {"type": "gaussian", "sigma_t": {"value": 2.0, "units": "ps"}}
        },
    )
    yaml_path = tmp_path / "distgen.yaml"
    yaml_path.write_text(yaml.safe_dump(config))

    beam_from_path = cheetah.ParticleBeam.from_distgen(yaml_path)
    beam_from_str = cheetah.ParticleBeam.from_distgen(str(yaml_path))

    assert beam_from_path.num_particles == 1000
    assert beam_from_str.num_particles == 1000


def test_from_distgen_generator_and_custom_energy():
    """Test passing a `distgen.Generator` instance, custom `energy`, and `dtype`."""
    config = _base_config(
        start={"type": "free"},
        longitudinal={
            "t_dist": {"type": "gaussian", "sigma_t": {"value": 2.0, "units": "ps"}}
        },
    )
    generator = Generator(config)
    custom_energy = torch.tensor(1e7, dtype=torch.float32)

    beam = cheetah.ParticleBeam.from_distgen(
        generator, energy=custom_energy, dtype=torch.float32
    )

    assert beam.particles.dtype == torch.float32
    torch.testing.assert_close(beam.energy, custom_energy)
