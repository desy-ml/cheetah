import argparse
import math
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import cheetah
from scipy import constants

# local
import envelope
from lattice import add_space_charge_elements
from lattice import slice_segment
from utils import get_perveance

plt.style.use("style.mplstyle")


# Parse arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nparts", type=int, default=100_000)
parser.add_argument("--kin-energy", type=float, default=1e9, help="GeV")
parser.add_argument("--intensity", type=float, default=1e16)
parser.add_argument("--beam-length", type=float, default=100.0)

parser.add_argument("--length", type=float, default=5.0)
parser.add_argument("--sc", type=int, default=1)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

data = {}
for key in ["beam", "env"]:
    data[key] = {}
    for dim in ["x_rms", "y_rms", "s"]:
        data[key][dim] = []


# Envelope tracking
# --------------------------------------------------------------------------------------

# Set distribution parameters
kin_energy = args.kin_energy  # [eV]
rest_energy = 0.938272029e+09  # [eV / c^2]
energy = kin_energy + rest_energy

lorentz_gamma = energy / rest_energy
lorentz_beta = math.sqrt(1.0 - 1.0 / lorentz_gamma**2)

line_density = args.intensity / args.beam_length
perveance = get_perveance(rest_energy, kin_energy, line_density)
if not args.sc:
    perveance = 0.0

rx = 0.020
ry = 0.020
rxp = 0.0
ryp = 0.0
env_params = [rx, rxp, ry, ryp]

# Integrate envelope equations
lattice = envelope.DriftLattice(length=args.length)
npoints = 1000
positions = np.linspace(0, lattice.length, npoints)

env_tracker = envelope.KVEnvelopeTracker(
    lattice=lattice, 
    perveance=perveance,
    eps_x=0.0, 
    eps_y=0.0,
)
env_sizes = env_tracker.track(env_params, positions)

# Store data
data["env"]["s"] = positions.copy()
data["env"]["x_rms"] = 0.5 * env_sizes[:, 0].copy()
data["env"]["y_rms"] = 0.5 * env_sizes[:, 1].copy()


# Beam tracking
# --------------------------------------------------------------------------------------

# Make lattice
length = torch.as_tensor(args.length)
segment = cheetah.Segment([cheetah.Drift(length)])
segment = slice_segment(segment, 100)

if args.sc:
    segment = add_space_charge_elements(
        segment,
        grid_shape=(128, 128),
        grid_extent_x=torch.tensor(3.0),
        grid_extent_y=torch.tensor(3.0),
    )

# Make beam (uniform density disk, zero emittance)
particles = torch.randn((args.nparts, 7))
particles[:, :4] /= torch.linalg.norm(particles[:, :4], axis=1)[:, None]
particles[:, :4] /= 0.5 * torch.std(particles[:, 4], axis=0)
particles[:, 0] *= rx * 0.5
particles[:, 2] *= ry * 0.5
particles[:, 1] *= 0.0
particles[:, 3] *= 0.0
particles[:, 4] = args.beam_length * (torch.rand(args.nparts) - 0.5)
particles[:, 5] = torch.zeros(args.nparts)

# Set macroparticle sizes
intensity = torch.tensor(args.intensity)
particle_macrosize = intensity / particles.shape[0]
particle_charge = particle_macrosize * constants.elementary_charge
particle_charges = particle_charge * torch.ones(particles.shape[0])

beam = cheetah.ParticleBeam(
    particles=particles,
    energy=torch.as_tensor(energy),
    species=cheetah.Species("proton"),
    particle_charges=particle_charges,
)

# Track
for index, element in enumerate(segment.elements):
    beam = element.track(beam)

    with torch.no_grad():
        if type(element) is cheetah.SpaceChargeKick2D:
            continue

        xrms = 1000.0 * torch.std(beam.particles[:, 0])
        yrms = 1000.0 * torch.std(beam.particles[:, 2])

        data["beam"]["x_rms"].append(float(xrms))
        data["beam"]["y_rms"].append(float(yrms))

        message = "step={} xrms={:0.3f} yrms={:0.3f}".format(
            index,
            xrms,
            yrms,
        )
        print(message)

# Store data
data["beam"]["x_rms"] = torch.as_tensor(data["beam"]["x_rms"])
data["beam"]["y_rms"] = torch.as_tensor(data["beam"]["y_rms"])
data["beam"]["s"] = torch.linspace(0.0, length, len(data["beam"]["x_rms"]))


# Analysis
# --------------------------------------------------------------------------------------

# Plot
fig, axs = plt.subplots(figsize=(4, 4), nrows=2, sharex=True, sharey=True)
axs[0].plot(data["env"]["s"], data["env"]["x_rms"], color="black", label="ENV")
axs[1].plot(data["env"]["s"], data["env"]["y_rms"], color="black", label="ENV")
axs[0].plot(data["beam"]["s"], data["beam"]["x_rms"], color="red", ls=":", label="PIC")
axs[1].plot(data["beam"]["s"], data["beam"]["y_rms"], color="red", ls=":", label="PIC")
axs[0].set_ylabel("x [mm]")
axs[1].set_ylabel("y [mm]")
axs[1].set_xlabel("s / L")
for ax in axs:
    ax.set_ylim(0.0, ax.get_ylim()[1] * 1.1)
    ax.legend(loc="upper left", fontsize="small")

filename = "fig_rms_beam_sizes"
filename = os.path.join(output_dir, filename)
plt.savefig(filename)
plt.close()