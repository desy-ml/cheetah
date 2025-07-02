from pathlib import Path

import pytest
import torch
from scipy.constants import physical_constants

import cheetah
from cheetah.utils.bmadx import bmad_to_cheetah_z_pz, cheetah_to_bmad_coords

atomic_mass_eV = (
    physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6
)


@pytest.mark.parametrize(
    "species",
    [
        cheetah.Species("proton", dtype=torch.float64),
        cheetah.Species("electron", dtype=torch.float64),
        cheetah.Species("positron", dtype=torch.float64),
        cheetah.Species("antiproton", dtype=torch.float64),
        cheetah.Species("deuteron", dtype=torch.float64),
        cheetah.Species(
            "#12C+3",
            num_elementary_charges=3,
            mass_eV=12 * atomic_mass_eV,
            dtype=torch.float64,
        ),
    ],
)
@pytest.mark.parametrize(
    "cheetah_element",
    [
        (
            cheetah.Drift(
                length=torch.tensor(1.0), tracking_method="bmadx", dtype=torch.float64
            )
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
                dtype=torch.float64,
            )
        ),
        (
            cheetah.Quadrupole(
                length=torch.tensor(0.5),
                k1=torch.tensor(1.0),
                tracking_method="bmadx",
                dtype=torch.float64,
            )
        ),
    ],
)
def test_different_species_in_different_elements(species, cheetah_element):
    """
    Test that tracking different particle species through different elements in Cheetah
    agrees with Bmad results.
    """
    coordinate_list = [1e-3, 2e-3, -3e-3, -1e-3, 2e-3, -1e-3]
    coordinates = torch.tensor(coordinate_list, dtype=torch.float64)

    p0c = torch.tensor(5.0e7, dtype=torch.float64)
    mc2 = species.mass_eV

    tau, delta, ref_energy = bmad_to_cheetah_z_pz(
        coordinates[4], coordinates[5], p0c, mc2
    )

    cheetah_coordinates = torch.ones((1, 7), dtype=torch.float64)
    cheetah_coordinates[:, :4] = coordinates[:4]
    cheetah_coordinates[:, 4] = tau
    cheetah_coordinates[:, 5] = delta

    incoming = cheetah.ParticleBeam(
        particles=cheetah_coordinates,
        energy=ref_energy,
        species=species,
        dtype=torch.float64,
    )

    # Track with Cheetah using bmadx routines
    outgoing = cheetah_element(incoming)
    # Convert to Bmad coordinates
    outgoing_bmad_coordinates, _ = cheetah_to_bmad_coords(
        outgoing.particles, ref_energy=outgoing.energy, mc2=outgoing.species.mass_eV
    )

    # Load Bmad results
    x_tao = torch.load(
        Path(__file__).parent
        / "resources"
        / "bmad"
        / f"x_tao_{species.name}_{cheetah_element.__class__.__name__}.pt"
    )

    assert torch.allclose(outgoing_bmad_coordinates, x_tao, atol=1e-14)
