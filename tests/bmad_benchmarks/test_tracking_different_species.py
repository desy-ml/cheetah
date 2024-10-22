import os

import pytest
import torch
from pytao import Tao
from scipy.constants import physical_constants

from cheetah import Dipole, Drift, ParticleBeam, Quadrupole
from cheetah.utils.bmadx import bmad_to_cheetah_z_pz, cheetah_to_bmad_coords
from cheetah.utils.particle_species import Species

atomic_mass_eV = (
    physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6
)


def set_coordinates_in_tao(tao, coords):
    tao.cmd("set particle_start x=" + str(coords[0]))
    tao.cmd("set particle_start px=" + str(coords[1]))
    tao.cmd("set particle_start y=" + str(coords[2]))
    tao.cmd("set particle_start py=" + str(coords[3]))
    tao.cmd("set particle_start z=" + str(coords[4]))
    tao.cmd("set particle_start pz=" + str(coords[5]))


@pytest.mark.parametrize(
    "species_name", ["proton", "electron", "positron", "antiproton", "deuteron"]
)
def test_track_a_drift_with_different_speices(species_name):
    """
    Test that tracking different particle species through a drift element in Cheetah
    agrees with Bmad results.
    """
    particle_species = Species(species_name)

    test_drift_lat = f"""
    parameter[lattice] = test_drift

    beginning[beta_a] = 10
    beginning[beta_b] = 10

    beginning[p0c] = 5.0e7
    parameter[particle] = {species_name}
    parameter[geometry] = open

    d1: drift, L = 1.0

    lat: line = (d1)

    use, lat
    """

    lattice_fname = f"tests/bmad_benchmarks/test_drift_{species_name}.bmad"

    open(lattice_fname, "w").write(test_drift_lat)
    tao = Tao(f"-lat {lattice_fname} -noplot")

    coords_list = [1e-3, 2e-3, -3e-3, -1e-3, 2e-3, -1e-3]
    coords = torch.tensor(coords_list, dtype=torch.double)

    p0c = torch.tensor(5.0e7, dtype=torch.double)
    mc2 = particle_species.mass_eV

    tau, delta, ref_energy = bmad_to_cheetah_z_pz(coords[4], coords[5], p0c, mc2)

    coords_cheetah = torch.ones(7, dtype=torch.double)
    coords_cheetah[:4] = coords[:4]
    coords_cheetah[4] = tau
    coords_cheetah[5] = delta

    p_in = ParticleBeam(
        particles=coords_cheetah.unsqueeze(0),
        energy=ref_energy,
        species=particle_species,
        dtype=torch.double,
    )

    d = Drift(
        length=torch.tensor(1.0, dtype=torch.double),
        tracking_method="bmadx",
        dtype=torch.double,
    )
    # Track with Cheetah using bmadx routines
    p_out = d(p_in)
    # Convert to Bmad coordinates
    p_out_bmad_coords, _ = cheetah_to_bmad_coords(
        p_out.particles, ref_energy=p_out.energy, mc2=p_out.mass_eV
    )

    # Track with Tao
    set_coordinates_in_tao(tao, coords_list)
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


@pytest.mark.parametrize(
    "species_name", ["proton", "electron", "positron", "antiproton", "deuteron"]
)
def test_track_a_dipole_with_different_speices(species_name):
    """
    Test that tracking different particle species through a dipole element in Cheetah
    agrees with Bmad results.
    """
    particle_species = Species(species_name)

    test_drift_lat = f"""
    parameter[lattice] = test_bend

    beginning[beta_a] = 10
    beginning[beta_b] = 10

    beginning[p0c] = 5.0e7
    parameter[particle] = {species_name}
    parameter[geometry] = open

    b1: sbend, L = 0.5, angle = 0.2, fringe_at = both_ends, fringe_type = linear_edge,
    E1 = 0.1, E2 = 0.1, FINT = 0.5, HGAP = 0.03, FINTX = 0.5, HGAPX = 0.03,
    ref_tilt = 0.1

    lat: line = (b1)

    use, lat
    """

    lattice_fname = f"tests/bmad_benchmarks/test_dipole_{species_name}.bmad"

    open(lattice_fname, "w").write(test_drift_lat)
    tao = Tao(f"-lat {lattice_fname} -noplot")

    coords_list = [1e-3, 2e-3, -3e-3, -1e-3, 2e-3, -1e-3]
    coords = torch.tensor(coords_list, dtype=torch.double)

    p0c = torch.tensor(5.0e7, dtype=torch.double)
    mc2 = particle_species.mass_eV

    tau, delta, ref_energy = bmad_to_cheetah_z_pz(coords[4], coords[5], p0c, mc2)

    coords_cheetah = torch.ones(7, dtype=torch.double)
    coords_cheetah[:4] = coords[:4]
    coords_cheetah[4] = tau
    coords_cheetah[5] = delta

    p_in = ParticleBeam(
        particles=coords_cheetah.unsqueeze(0),
        energy=ref_energy,
        species=particle_species,
        dtype=torch.double,
    )

    angle = torch.tensor(0.2, dtype=torch.double)
    e1 = angle / 2
    e2 = angle - e1
    dipole = Dipole(
        length=torch.tensor([0.5], dtype=torch.double),
        angle=angle,
        e1=e1,
        e2=e2,
        tilt=torch.tensor([0.1], dtype=torch.double),
        fringe_integral=torch.tensor([0.5], dtype=torch.double),
        fringe_integral_exit=torch.tensor([0.5], dtype=torch.double),
        gap=torch.tensor([0.06], dtype=torch.double),
        gap_exit=torch.tensor([0.06], dtype=torch.double),
        fringe_at="both",
        fringe_type="linear_edge",
        tracking_method="bmadx",
        dtype=torch.double,
    )
    # Track with Cheetah using bmadx routines
    p_out = dipole(p_in)
    # Convert to Bmad coordinates
    p_out_bmad_coords, _ = cheetah_to_bmad_coords(
        p_out.particles, ref_energy=p_out.energy, mc2=p_out.mass_eV
    )

    # Track with Tao
    set_coordinates_in_tao(tao, coords_list)
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


@pytest.mark.parametrize("species_name", ["proton"])
def test_track_a_quadrupole_with_different_species(species_name):
    """
    Test that tracking different particle species through a quadrupole element in
    Cheetah agrees with Bmad results.
    """
    particle_species = Species(species_name)

    test_drift_lat = f"""
    parameter[lattice] = test_quad

    beginning[beta_a] = 10
    beginning[beta_b] = 10

    beginning[p0c] = 5.0e7
    parameter[particle] = {species_name}
    parameter[geometry] = open

    q1: quad, L = 0.5, K1 = 1.0

    lat: line = (q1)

    use, lat
    """

    lattice_fname = f"tests/bmad_benchmarks/test_quad_{species_name}.bmad"

    open(lattice_fname, "w").write(test_drift_lat)
    tao = Tao(f"-lat {lattice_fname} -noplot")

    coords_list = [1e-3, 2e-3, -3e-3, -1e-3, 2e-3, -1e-3]
    coords = torch.tensor(coords_list, dtype=torch.double)

    p0c = torch.tensor(5.0e7, dtype=torch.double)
    mc2 = particle_species.mass_eV

    tau, delta, ref_energy = bmad_to_cheetah_z_pz(coords[4], coords[5], p0c, mc2)

    coords_cheetah = torch.ones(7, dtype=torch.double)
    coords_cheetah[:4] = coords[:4]
    coords_cheetah[4] = tau
    coords_cheetah[5] = delta

    p_in = ParticleBeam(
        particles=coords_cheetah.unsqueeze(0),
        energy=ref_energy,
        species=particle_species,
        dtype=torch.double,
    )

    quadrupole = Quadrupole(
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
    set_coordinates_in_tao(tao, coords_list)
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


def test_track_a_quadrupole_with_ion():
    """
    Test that tracking ions through a quadrupole element in
    Cheetah agrees with Bmad results.
    Here we use carbon-12 with +3 charge.
    """
    particle_species = Species(
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

    p_in = ParticleBeam(
        particles=coords_cheetah.unsqueeze(0),
        energy=ref_energy,
        species=particle_species,
        dtype=torch.double,
    )

    quadrupole = Quadrupole(
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
    set_coordinates_in_tao(tao, coords_list)
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
