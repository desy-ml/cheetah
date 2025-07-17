from pathlib import Path

import numpy as np
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


@pytest.mark.parametrize("cavity_type", ["standing_wave", "traveling_wave"])
@pytest.mark.parametrize("phase", [0.0, 30.0])
def test_cavity(cavity_type, phase):
    """
    Test that tracking a particle through a cavity in Cheetah agrees with Bmad results.
    """
    incoming = cheetah.ParameterBeam.from_twiss(
        beta_x=torch.tensor(5.91253677),
        beta_y=torch.tensor(5.91253677),
        alpha_x=torch.tensor(3.55631308),
        alpha_y=torch.tensor(3.55631308),
        energy=torch.tensor(6e6),
        species=cheetah.Species("electron"),
    )

    cavity = cheetah.Cavity(
        length=torch.tensor(1.0377),
        voltage=torch.tensor(0.01815975e9),
        phase=torch.tensor(phase),
        frequency=torch.tensor(1.3e9),
        cavity_type=cavity_type,
    )

    outgoing = cavity.track(incoming)

    if cavity_type == "standing_wave" and phase == 0.0:
        expected_beta_x = 0.238473521593469
        expected_beta_y = 0.238473521593469
        expected_alpha_x = -1.01606875938083
        expected_alpha_y = -1.01606875938083
        expected_energy = "todo"
    elif cavity_type == "traveling_wave" and phase == 0.0:
        expected_beta_x = 2.9943987369624
        expected_beta_y = 2.9943987369624
        expected_alpha_x = -5.30493404396745
        expected_alpha_y = -5.30493404396745
        expected_energy = "todo"
    else:
        pytest.fail(
            f"Unexpected combination of cavity_type={cavity_type} and phase={phase}"
        )

    assert np.isclose(outgoing.beta_x, expected_beta_x)
    assert np.isclose(outgoing.beta_y, expected_beta_y)
    assert np.isclose(outgoing.alpha_x, expected_alpha_x)
    assert np.isclose(outgoing.alpha_y, expected_alpha_y)
    assert np.isclose(outgoing.energy, expected_energy)
