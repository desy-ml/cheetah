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
from lattice import build_fodo_segment
from lattice import add_space_charge_elements
from lattice import slice_segment
from utils import get_perveance
from utils import build_norm_matrix_from_twiss_2d

plt.style.use("style.mplstyle")


# Parse arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nparts", type=int, default=100_000)
parser.add_argument("--kin-energy", type=float, default=1e9, help="GeV")
parser.add_argument("--intensity", type=float, default=1e15)
parser.add_argument("--beam-length", type=float, default=100.0)

parser.add_argument("--length", type=float, default=5.0)
parser.add_argument("--kq", type=float, default=0.55)
parser.add_argument("--periods", type=int, default=4)

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

alpha_x = 0.0
alpha_y = 0.0
beta_x = 8.0
beta_y = 1.5
eps_x = 10.0e-06
eps_y = 10.0e-06

rx = 2.0 * np.sqrt(eps_x * beta_x)
ry = 2.0 * np.sqrt(eps_y * beta_y)
rxp = 0.0
ryp = 0.0
env_params = [rx, rxp, ry, ryp]

# Integrate envelope equations
lattice = envelope.FODOLattice(length=args.length, kq=args.kq)
npoints = 400 * args.periods
positions = np.linspace(0, lattice.length * args.periods, npoints)

env_tracker = envelope.KVEnvelopeTracker(
    lattice=lattice, 
    perveance=perveance,
    eps_x=eps_x, 
    eps_y=eps_y
)
env_sizes = env_tracker.track(env_params, positions)

# Store data
data["env"]["s"] = positions.copy()
data["env"]["x_rms"] = 0.5 * env_sizes[:, 0].copy()
data["env"]["y_rms"] = 0.5 * env_sizes[:, 1].copy()


# Beam tracking
# --------------------------------------------------------------------------------------

# Make lattice
kq = torch.as_tensor(args.kq)
length = torch.as_tensor(args.length)

segment = build_fodo_segment(length=length, kq=kq, nslice=50)
if args.sc:
    segment = add_space_charge_elements(
        segment,
        grid_shape=(64, 64),
        grid_extent_x=torch.tensor(3.0),
        grid_extent_y=torch.tensor(3.0),
    )

# Make beam
norm_matrix = torch.eye(4)
norm_matrix[0:2, 0:2] = build_norm_matrix_from_twiss_2d(
    alpha=alpha_x, beta=beta_x, eps=eps_x
)
norm_matrix[2:4, 2:4] = build_norm_matrix_from_twiss_2d(
    alpha=alpha_y, beta=beta_y, eps=eps_y
)
unnorm_matrix = torch.linalg.inv(norm_matrix)

particles = torch.randn((args.nparts, 7))
particles[:, :4] /= torch.linalg.norm(particles[:, :4], axis=1)[:, None]
particles[:, :4] /= 0.5 * torch.std(particles[:, 4], axis=0)
particles[:, :4] = torch.matmul(particles[:, :4], unnorm_matrix.T)
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
for period in range(args.periods):
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
data["beam"]["s"] = torch.linspace(0.0, length * args.periods, len(data["beam"]["x_rms"]))


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