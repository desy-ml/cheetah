import argparse
import cheetah
import numpy as np
import pytest
import torch
from scipy.constants import elementary_charge
from scipy.constants import speed_of_light

# local
from envelope import KVEnvelope
from envelope import KVEnvelopeTracker
from envelope import DriftLattice
from utils import add_space_charge_elements
from utils import split_segment


def test_kv_drift():

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
    beam_charge = elementary_charge * beam_intensity
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
