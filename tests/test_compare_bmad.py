import os

import pytest
import torch
from pytao import Tao
from scipy.constants import physical_constants

import cheetah
from cheetah.utils.bmadx import bmad_to_cheetah_z_pz, cheetah_to_bmad_coords

atomic_mass_eV = (
    physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6
)

cheetah_elements_and_respective_bmad_strs = {
    "drift": {
        "cheetah": cheetah.Drift(
            length=torch.tensor(1.0), tracking_method="bmadx", dtype=torch.double
        ),
        "bmad": "d1: drift, L = 1.0",
    },
    "dipole": {
        "cheetah": cheetah.Dipole(
            length=torch.tensor(1.0),
            angle=torch.tensor(0.2),
            dipole_e1=torch.tensor(0.1),
            dipole_e2=torch.tensor(0.1),
            tilt=torch.tensor(0.1),
            fringe_integral=torch.tensor(0.5),
            fringe_integral_exit=torch.tensor(0.5),
            gap=torch.tensor(0.03),
            gap_exit=torch.tensor(0.03),
            fringe_at="both_ends",
            fringe_type="linear_edge",
            tracking_method="bmadx",
            dtype=torch.double,
        ),
        "bmad": (
            "b1: sbend, L = 1.0, angle = 0.2, fringe_at = both_ends, "
            "fringe_type = linear_edge, E1 = 0.1, E2 = 0.1, FINT = 0.5, HGAP = 0.03, "
            "FINTX = 0.5, HGAPX = 0.03, ref_tilt = 0.1"
        ),
    },
    "quadrupole": {
        "cheetah": cheetah.Quadrupole(
            length=torch.tensor(1.0), k1=torch.tensor(1.0), tracking_method="bmadx"
        ),
        "bmad": "q1: quad, L = 1.0, K1 = 1.0",
    },
}


def tao_set_particle_start(tao: Tao, coordinates: torch.Tensor) -> None:
    """Set the initial coordinates of the particle in Tao."""

    tao.cmd(f"set particle_start x={coordinates[0]}")
    tao.cmd(f"set particle_start px={coordinates[1]}")
    tao.cmd(f"set particle_start y={coordinates[2]}")
    tao.cmd(f"set particle_start py={coordinates[3]}")
    tao.cmd(f"set particle_start z={coordinates[4]}")
    tao.cmd(f"set particle_start pz={coordinates[5]}")


@pytest.mark.parametrize(
    "species_name", ["proton", "electron", "positron", "antiproton", "deuteron"]
)
@pytest.mark.parametrize("element_name", ["drift", "dipole", "quadrupole"])
def test_different_species_in_different_elements(tmp_path, species_name, element_name):
    """
    Test that tracking different particle species through a drift element in Cheetah
    agrees with Bmad results.
    """
    particle_species = cheetah.Species(species_name)

    bmad_drift_lattice_str = f"""
    parameter[lattice] = test_drift

    beginning[beta_a] = 10
    beginning[beta_b] = 10

    beginning[p0c] = 5.0e7
    parameter[particle] = {species_name}
    parameter[geometry] = open

    {cheetah_elements_and_respective_bmad_strs[element_name]["bmad"]}

    lat: line = (d1)

    use, lat
    """
    bmad_lattice_path = tmp_path / f"test_drift_{species_name}.bmad"
    with open(bmad_lattice_path, "w") as f:
        f.write(bmad_drift_lattice_str)

    tao = Tao(f"-lat {bmad_lattice_path} -noplot")

    coordinate_list = [1e-3, 2e-3, -3e-3, -1e-3, 2e-3, -1e-3]
    coordinates = torch.tensor(coordinate_list, dtype=torch.double)

    p0c = torch.tensor(5.0e7, dtype=torch.double)
    mc2 = particle_species.mass_eV

    tau, delta, ref_energy = bmad_to_cheetah_z_pz(
        coordinates[4], coordinates[5], p0c, mc2
    )

    cheetah_coordinates = torch.ones((1, 7), dtype=torch.double)
    cheetah_coordinates[:, :4] = coordinates[:4]
    cheetah_coordinates[:, 4] = tau
    cheetah_coordinates[:, 5] = delta

    incoming = cheetah.ParticleBeam(
        particles=cheetah_coordinates,
        energy=ref_energy,
        species=particle_species,
        dtype=torch.double,
    )

    element = cheetah_elements_and_respective_bmad_strs[element_name]["cheetah"]
    # Track with Cheetah using bmadx routines
    outgoing = element(incoming)
    # Convert to Bmad coordinates
    outgoing_bmad_coordinates, _ = cheetah_to_bmad_coords(
        outgoing.particles, ref_energy=outgoing.energy, mc2=outgoing.mass_eV
    )

    # Track with Tao
    tao_set_particle_start(tao, coordinate_list)
    orbit_out = tao.orbit_at_s(ele=1)

    x_tao = torch.tensor(
        [
            orbit_out["x"],
            orbit_out["px"],
            orbit_out["y"],
            orbit_out["py"],
            orbit_out["z"],
            orbit_out["pz"],
        ],
        dtype=torch.double,
    )

    assert torch.allclose(outgoing_bmad_coordinates, x_tao, atol=1e-14)


def test_track_a_quadrupole_with_ion():
    """
    Test that tracking ions through a quadrupole element in
    Cheetah agrees with Bmad results.
    Here we use carbon-12 with +3 charge.
    """
    particle_species = cheetah.Species(
        name="#12C+3",
        charge=3,
        mass=12 * atomic_mass_eV,
    )

    test_drift_lat = """
    parameter[lattice] = test_quad

    beginning[beta_a] = 10
    beginning[beta_b] = 10

    beginning[p0c] = 10.0e7
    parameter[particle] = #12C+3
    parameter[geometry] = open

    q1: quad, L = 0.5, K1 = 1.0

    lat: line = (q1)

    use, lat
    """

    lattice_fname = "tests/bmad_benchmarks/test_quad_ion.bmad"

    open(lattice_fname, "w").write(test_drift_lat)
    tao = Tao(f"-lat {lattice_fname} -noplot")

    coords_list = [1e-3, 2e-3, -3e-3, -1e-3, 2e-3, -1e-3]
    coords = torch.tensor(coords_list, dtype=torch.double)

    p0c = torch.tensor(10.0e7, dtype=torch.double)
    mc2 = particle_species.mass_eV

    tau, delta, ref_energy = bmad_to_cheetah_z_pz(coords[4], coords[5], p0c, mc2)

    coords_cheetah = torch.ones(7, dtype=torch.double)
    coords_cheetah[:4] = coords[:4]
    coords_cheetah[4] = tau
    coords_cheetah[5] = delta

    p_in = cheetah.ParticleBeam(
        particles=coords_cheetah.unsqueeze(0),
        energy=ref_energy,
        species=particle_species,
        dtype=torch.double,
    )

    quadrupole = cheetah.Quadrupole(
        length=torch.tensor([0.5], dtype=torch.double),
        k1=torch.tensor([1.0], dtype=torch.double),
        tracking_method="bmadx",
        dtype=torch.double,
    )
    # Track with Cheetah using bmadx routines
    p_out = quadrupole(p_in)
    # Convert to Bmad coordinates
    p_out_bmad_coords, _ = cheetah_to_bmad_coords(
        p_out.particles, ref_energy=p_out.energy, mc2=p_out.mass_eV
    )

    # Track with Tao
    tao_set_particle_start(tao, coords_list)
    orbit_out = tao.orbit_at_s(ele=1)

    x_tao = torch.tensor(
        [
            orbit_out["x"],
            orbit_out["px"],
            orbit_out["y"],
            orbit_out["py"],
            orbit_out["z"],
            orbit_out["pz"],
        ],
        dtype=torch.double,
    )

    assert torch.allclose(p_out_bmad_coords, x_tao, atol=1e-14)

    # Clean up
    os.remove(lattice_fname)
