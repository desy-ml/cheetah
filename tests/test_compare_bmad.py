import sys

import pytest

if sys.platform.startswith("win"):
    pytest.skip("Skipping Bmad comparison tests on Windows", allow_module_level=True)

import torch
from pytao import Tao
from scipy.constants import physical_constants

import cheetah
from cheetah.utils.bmadx import bmad_to_cheetah_z_pz, cheetah_to_bmad_coords

atomic_mass_eV = (
    physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6
)


def tao_set_particle_start(tao: Tao, coordinates: torch.Tensor) -> None:
    """Helper function to set the initial coordinates of the particle in Tao."""
    tao.cmd(f"set particle_start x={coordinates[0]}")
    tao.cmd(f"set particle_start px={coordinates[1]}")
    tao.cmd(f"set particle_start y={coordinates[2]}")
    tao.cmd(f"set particle_start py={coordinates[3]}")
    tao.cmd(f"set particle_start z={coordinates[4]}")
    tao.cmd(f"set particle_start pz={coordinates[5]}")


@pytest.mark.parametrize(
    "species",
    [
        cheetah.Species("proton"),
        cheetah.Species("electron"),
        cheetah.Species("positron"),
        cheetah.Species("antiproton"),
        cheetah.Species("deuteron"),
        cheetah.Species(
            "#12C+3", num_elementary_charges=3, mass_eV=12 * atomic_mass_eV
        ),
    ],
)
@pytest.mark.parametrize(
    ["cheetah_element", "bmad_element_str"],
    [
        (
            cheetah.Drift(
                length=torch.tensor(1.0), tracking_method="bmadx", dtype=torch.double
            ),
            "e1: drift, L = 1.0",
        ),
        (
            cheetah.Dipole(
                length=torch.tensor(0.5),
                angle=torch.tensor(0.2),
                dipole_e1=torch.tensor(0.1),
                dipole_e2=torch.tensor(0.1),
                tilt=torch.tensor(0.1),
                fringe_integral=torch.tensor(0.5),
                fringe_integral_exit=torch.tensor(0.5),
                gap=torch.tensor(0.06),
                gap_exit=torch.tensor(0.06),
                fringe_at="both",
                fringe_type="linear_edge",
                tracking_method="bmadx",
                dtype=torch.double,
            ),
            "e1: sbend, L = 0.5, angle = 0.2, fringe_at = both_ends, "
            "fringe_type = linear_edge, E1 = 0.1, E2 = 0.1, FINT = 0.5, HGAP = 0.03, "
            "FINTX = 0.5, HGAPX = 0.03, ref_tilt = 0.1",
        ),
        (
            cheetah.Quadrupole(
                length=torch.tensor(0.5),
                k1=torch.tensor(1.0),
                tracking_method="bmadx",
                dtype=torch.double,
            ),
            "e1: quad, L = 0.5, K1 = 1.0",
        ),
    ],
)
def test_different_species_in_different_elements(
    tmp_path, species, cheetah_element, bmad_element_str
):
    """
    Test that tracking different particle species through different elements in Cheetah
    agrees with Bmad results.
    """
    bmad_drift_lattice_str = f"""
    parameter[lattice] = test_drift

    beginning[beta_a] = 10
    beginning[beta_b] = 10

    beginning[p0c] = 5.0e7
    parameter[particle] = {species.name}
    parameter[geometry] = open

    {bmad_element_str}

    lat: line = (e1)

    use, lat
    """
    bmad_lattice_path = tmp_path / f"test_drift_{species.name}.bmad"
    with open(bmad_lattice_path, "w") as f:
        f.write(bmad_drift_lattice_str)

    tao = Tao(f"-lat {bmad_lattice_path} -noplot")

    coordinate_list = [1e-3, 2e-3, -3e-3, -1e-3, 2e-3, -1e-3]
    coordinates = torch.tensor(coordinate_list, dtype=torch.double)

    p0c = torch.tensor(5.0e7, dtype=torch.double)
    mc2 = species.mass_eV

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
        species=species,
        dtype=torch.double,
    )

    # Track with Cheetah using bmadx routines
    outgoing = cheetah_element(incoming)
    # Convert to Bmad coordinates
    outgoing_bmad_coordinates, _ = cheetah_to_bmad_coords(
        outgoing.particles, ref_energy=outgoing.energy, mc2=outgoing.species.mass_eV
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


@pytest.mark.parametrize("cavity_type", ["standing_wave", "traveling_wave"])
@pytest.mark.parametrize("phase", [0.0, 30.0])
def test_cavity(tmp_path, cavity_type, phase):
    """
    Test that tracking a particle through a cavity in Cheetah agrees with Bmad results.
    """
    bmad_cavity_lattice_str = f"""
    parameter[lattice] = test_cavity

    beginning[beta_a] = 5.91253677 ! m a-mode beta function
    beginning[beta_b] = 5.91253677 ! m b-mode beta function
    beginning[alpha_a] = 3.55631308 ! a-mode alpha function
    beginning[alpha_b] = 3.55631308 ! b-mode alpha function
    beginning[p0c] = 6e6 ! eV    Or can set beginning[p0c]

    parameter[geometry] = open  ! Or closed
    parameter[particle] = electron  ! Reference particle.

    c: lcavity, rf_frequency = 1.3e9, l = 1.0377, voltage = 0.01815975e9, phi0 = {phase}, cavity_type = {cavity_type}

    lat: line = (c) ! List of lattice elements
    use, lat ! Line used to construct the lattice
    """
    bmad_lattice_path = tmp_path / f"test_cavity_{cavity_type}_{phase}.bmad"
    with open(bmad_lattice_path, "w") as f:
        f.write(bmad_cavity_lattice_str)

    tao = Tao(f"-lat {bmad_lattice_path} -noplot")

    coordinate_list = [1e-3, 2e-3, -3e-3, -1e-3, 2e-3, -1e-3]
    coordinates = torch.tensor(coordinate_list, dtype=torch.double)

    p0c = torch.tensor(6e6, dtype=torch.double)
    mc2 = cheetah.Species("electron").mass_eV

    tau, delta, ref_energy = bmad_to_cheetah_z_pz(
        coordinates[4], coordinates[5], p0c, mc2
    )

    cheetah_element = cheetah.Cavity(
        length=torch.tensor(1.0377),
        voltage=torch.tensor(0.01815975e9),
        phase=torch.tensor(phase),
        frequency=torch.tensor(1.3e9),
        cavity_type=cavity_type,
        dtype=torch.double,
    )

    cheetah_coordinates = torch.ones((1, 7), dtype=torch.double)
    cheetah_coordinates[:, :4] = coordinates[:4]
    cheetah_coordinates[:, 4] = tau
    cheetah_coordinates[:, 5] = delta

    incoming = cheetah.ParticleBeam(
        particles=cheetah_coordinates,
        energy=ref_energy,
        species=cheetah.Species("electron"),
        dtype=torch.double,
    )

    # Track with Cheetah
    outgoing = cheetah_element(incoming)
    # Convert to Bmad coordinates
    outgoing_bmad_coordinates, _ = cheetah_to_bmad_coords(
        outgoing.particles, ref_energy=outgoing.energy, mc2=outgoing.species.mass_eV
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
