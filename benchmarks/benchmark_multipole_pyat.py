#!/usr/bin/env python3
# Import pyAT components
import at
import numpy as np
import pytest
import torch

# Import Cheetah components
from cheetah.accelerator.multipole import Multipole
from cheetah.particles import ParticleBeam, Species

# Configuration parameters
ELEMENT_LENGTH = 0.5  # meters
SHORT_LENGTH = 1e-9  # meters for short dipole test
NUM_STEPS = 10  # Number of integration steps

# Magnet strengths
DRIFT_K1 = 0.0  # 1/meters^2, no focusing (drift)
QUAD_K1 = 2.0  # 1/meters^2, quadrupole strength
SEXT_K2 = 10.0  # 1/meters^3, sextupole strength
DIPOLE_K0 = 1.0  # 1/meters, dipole strength

# Fringe field flags
FRINGE_ON = 1  # Enable fringe fields
FRINGE_OFF = 0  # Disable fringe fields

# Radiation flags
RADIATION_ON = True  # Enable radiation effects
RADIATION_OFF = False  # Disable radiation effects

# Initial particle coordinates
X0 = 0.001  # meters
PX0 = 0.0  # radians
Y0 = 0.0  # meters
PY0 = 0.001  # radians
TAU0 = 0.1  # seconds
DELTA0 = -0.1  # relative energy deviation

# Tolerance for coordinate comparisons
TOLERANCE = 5e-8  # Maximum acceptable difference between PyAT and Cheetah results


def create_pyat_quadrupole(length, k1, num_steps, fringe=0, radiation=False):
    """Create a quadrupole element in pyAT"""
    pass_method = (
        "StrMPoleSymplectic4RadPass" if radiation else "StrMPoleSymplectic4Pass"
    )
    quad = at.Quadrupole(
        "Q1",
        length,
        k1,
        NumIntSteps=num_steps,
        Energy=6.0e9,
        FringeQuadEntrance=fringe,
        FringeQuadExit=fringe,
        PassMethod=pass_method,
    )
    return quad


def create_pyat_sextupole(length, k2, num_steps, radiation=False):
    """Create a sextupole element in pyAT"""
    pass_method = (
        "StrMPoleSymplectic4RadPass" if radiation else "StrMPoleSymplectic4Pass"
    )
    sext = at.Sextupole(
        "S1", length, k2, NumIntSteps=num_steps, Energy=6.0e9, PassMethod=pass_method
    )
    return sext


def create_pyat_corrector(kick_x, kick_y, length):
    """Create a corrector element in pyAT"""
    corrector = at.Corrector("C1", length, np.array([kick_x, kick_y]))
    return corrector


def create_pyat_dipole(length, k0, num_steps, radiation=False):
    """Create a dipole element in pyAT"""
    pass_method = (
        "StrMPoleSymplectic4RadPass" if radiation else "StrMPoleSymplectic4Pass"
    )

    # For a dipole, use PolynomB[0] = k0
    poly_a = []  # No skew components
    poly_b = [k0]  # Dipole component only

    dipole = at.Multipole(
        "D1",
        length,
        poly_a,
        poly_b,
        MaxOrder=0,  # Only using dipole component
        NumIntSteps=num_steps,
        Energy=6.0e9,
        PassMethod=pass_method,
    )
    return dipole


def create_cheetah_dipole(length, k0, num_steps, radiation=False):
    """Create a dipole element in Cheetah"""
    polynom_b = torch.zeros(1, dtype=torch.float64)
    polynom_b[0] = k0

    tracking_method = "symplectic4_rad" if radiation else "symplectic4"

    dipole = Multipole(
        length=torch.tensor(length, dtype=torch.float64),
        polynom_b=polynom_b,
        max_order=0,
        num_steps=num_steps,
        tracking_method=tracking_method,
        name="D1",
    )
    return dipole


def create_cheetah_quadrupole(length, k1, num_steps, fringe=0, radiation=False):
    """Create a quadrupole element in Cheetah"""
    polynom_b = torch.zeros(2, dtype=torch.float64)
    polynom_b[1] = k1

    tracking_method = "symplectic4_rad" if radiation else "symplectic4"

    quad = Multipole(
        length=torch.tensor(length, dtype=torch.float64),
        polynom_b=polynom_b,
        max_order=1,
        num_steps=num_steps,
        fringe_quad_entrance=fringe,
        fringe_quad_exit=fringe,
        tracking_method=tracking_method,
        name="Q1",
    )
    return quad


def create_cheetah_sextupole(length, k2, num_steps, radiation=False):
    """Create a sextupole element in Cheetah"""
    polynom_b = torch.zeros(3, dtype=torch.float64)
    polynom_b[2] = k2

    tracking_method = "symplectic4_rad" if radiation else "symplectic4"

    sext = Multipole(
        length=torch.tensor(length, dtype=torch.float64),
        polynom_b=polynom_b,
        max_order=2,
        num_steps=num_steps,
        tracking_method=tracking_method,
        name="S1",
    )
    return sext


def track_pyat(element, particle):
    """Track particle through pyAT element"""
    # Make a copy to avoid modifying the original
    particle_out = element.track(particle.copy())
    return particle_out


def track_cheetah(element, particle):
    """Track particle through Cheetah element"""
    electron = Species("electron")
    beam = ParticleBeam(
        particles=particle,
        energy=torch.tensor(6.0e9, dtype=torch.float64),
        species=electron,
    )

    beam_out = element.track(beam)
    return beam_out


def get_aligned_coordinates(pyat_final):
    """
    Align coordinates from pyAT and Cheetah for comparison

    PyAT: [0]=x, [1]=px, [2]=y, [3]=py, [4]=delta, [5]=tau
    Cheetah: [0]=x, [1]=px, [2]=y, [3]=py, [4]=tau, [5]=delta
    """
    aligned_pyat = np.array(
        [
            pyat_final[0],  # x
            pyat_final[1],  # px
            pyat_final[2],  # y
            pyat_final[3],  # py
            pyat_final[5],  # tau (at[5] -> cheetah[4])
            pyat_final[4],  # delta (at[4] -> cheetah[5])
        ]
    )
    return aligned_pyat


def run_element_tracking(
    element_type, length, strength, num_steps, fringe=0, radiation=False
):
    """Run tracking for a specific element type and return results for comparison"""
    # Create elements based on type
    if element_type == "drift":
        pyat_element = create_pyat_quadrupole(
            length, 0.0, num_steps, fringe, radiation
        )  # Drift is quad with k1=0
        cheetah_element = create_cheetah_quadrupole(
            length, 0.0, num_steps, fringe, radiation
        )
    elif element_type == "quadrupole":
        pyat_element = create_pyat_quadrupole(
            length, strength, num_steps, fringe, radiation
        )
        cheetah_element = create_cheetah_quadrupole(
            length, strength, num_steps, fringe, radiation
        )
    elif element_type == "sextupole":
        pyat_element = create_pyat_sextupole(length, strength, num_steps, radiation)
        cheetah_element = create_cheetah_sextupole(
            length, strength, num_steps, radiation
        )
    elif element_type == "dipole":
        pyat_element = create_pyat_dipole(length, strength, num_steps, radiation)
        cheetah_element = create_cheetah_dipole(length, strength, num_steps, radiation)
    elif element_type == "corrector":
        # For corrector, convert integrated dipole strength to kick angle
        # kick = B0 * L / (Bρ), where Bρ = p/q ≈ E/c for ultrarelativistic particles
        # For E = 6 GeV, Bρ ≈ 20 T⋅m
        brho = 20.0  # T⋅m for 6 GeV electrons
        kick_angle = strength * length / brho  # Convert B0*L to angle in radians

        pyat_element = create_pyat_corrector(kick_angle, 0.0, length)
        # Corrector doesn't support radiation, so force it off for Cheetah comparison
        cheetah_element = create_cheetah_dipole(
            length, strength, num_steps, radiation=False
        )
    else:
        raise ValueError(f"Unknown element type: {element_type}")

    # Create particles
    # PyAT coordinates: [0]=x, [1]=px, [2]=y, [3]=py, [4]=delta, [5]=tau
    pyat_particle = np.array([[X0, PX0, Y0, PY0, DELTA0, TAU0]]).T

    # Cheetah coordinates: [0]=x, [1]=px, [2]=y, [3]=py, [4]=tau, [5]=delta, [6]=1.0
    # (particle flag)
    cheetah_particle = torch.tensor(
        [[X0, PX0, Y0, PY0, TAU0, DELTA0, 1.0]], dtype=torch.float64
    )

    # Track particles
    pyat_out = track_pyat(pyat_element, pyat_particle)
    cheetah_out = track_cheetah(cheetah_element, cheetah_particle)

    # Get final coordinates
    pyat_final = pyat_out.flatten()
    cheetah_final = cheetah_out.particles[0, :6].cpu().numpy()

    return pyat_final, cheetah_final


# Helper function to assert coordinates are close (for pytest)
def assert_coordinates_close(pyat_final, cheetah_final, test_name):
    """Assert that pyAT and Cheetah coordinates are within tolerance"""
    aligned_pyat = get_aligned_coordinates(pyat_final)
    max_diff = np.max(np.abs(aligned_pyat - cheetah_final))

    assert (
        max_diff <= TOLERANCE
    ), f"Coordinates differ by {max_diff} in {test_name} test"


# Converted tests to pytest style
def test_drift():
    """Test drift space tracking"""
    pyat_final, cheetah_final = run_element_tracking(
        "drift", ELEMENT_LENGTH, DRIFT_K1, NUM_STEPS
    )
    assert_coordinates_close(pyat_final, cheetah_final, "drift")

    # Verify that PyAT and Cheetah agree on the drift behavior by comparing their
    # coordinate changes
    aligned_pyat = get_aligned_coordinates(pyat_final)

    # For drift, check that x and y changes match between PyAT and Cheetah
    assert (
        abs((aligned_pyat[0] - X0) - (cheetah_final[0] - X0)) <= TOLERANCE
    ), "Drift x coordinate change doesn't match between PyAT and Cheetah"
    assert (
        abs((aligned_pyat[2] - Y0) - (cheetah_final[2] - Y0)) <= TOLERANCE
    ), "Drift y coordinate change doesn't match between PyAT and Cheetah"


def test_quadrupole():
    """Test quadrupole tracking"""
    pyat_final, cheetah_final = run_element_tracking(
        "quadrupole", ELEMENT_LENGTH, QUAD_K1, NUM_STEPS
    )
    assert_coordinates_close(pyat_final, cheetah_final, "quadrupole")


def test_sextupole():
    """Test sextupole tracking"""
    pyat_final, cheetah_final = run_element_tracking(
        "sextupole", ELEMENT_LENGTH, SEXT_K2, NUM_STEPS
    )
    assert_coordinates_close(pyat_final, cheetah_final, "sextupole")


def test_dipole_long():
    """Test long dipole tracking"""
    pyat_final, cheetah_final = run_element_tracking(
        "dipole", ELEMENT_LENGTH, DIPOLE_K0, NUM_STEPS
    )
    assert_coordinates_close(pyat_final, cheetah_final, "dipole_long")


def test_dipole_short():
    """Test short dipole tracking"""
    pyat_final, cheetah_final = run_element_tracking(
        "corrector", SHORT_LENGTH, DIPOLE_K0, NUM_STEPS
    )
    assert_coordinates_close(pyat_final, cheetah_final, "corrector")


def test_dipole_long_radiation():
    """Test long dipole tracking with radiation"""
    pyat_final, cheetah_final = run_element_tracking(
        "dipole", ELEMENT_LENGTH, DIPOLE_K0, NUM_STEPS, radiation=RADIATION_ON
    )
    assert_coordinates_close(pyat_final, cheetah_final, "dipole_long_radiation")


def test_dipole_short_radiation():
    """Test short dipole tracking with radiation"""
    pyat_final, cheetah_final = run_element_tracking(
        "corrector", SHORT_LENGTH, DIPOLE_K0, NUM_STEPS, radiation=RADIATION_ON
    )
    assert_coordinates_close(pyat_final, cheetah_final, "corrector_radiation")


def test_quadrupole_fringe():
    """Test quadrupole with fringe fields"""
    pyat_final, cheetah_final = run_element_tracking(
        "quadrupole", ELEMENT_LENGTH, QUAD_K1, NUM_STEPS, fringe=FRINGE_ON
    )
    assert_coordinates_close(pyat_final, cheetah_final, "quadrupole_fringe")


def test_quadrupole_radiation():
    """Test quadrupole with radiation"""
    pyat_final, cheetah_final = run_element_tracking(
        "quadrupole", ELEMENT_LENGTH, QUAD_K1, NUM_STEPS, radiation=RADIATION_ON
    )
    assert_coordinates_close(pyat_final, cheetah_final, "quadrupole_radiation")


def test_sextupole_radiation():
    """Test sextupole with radiation"""
    pyat_final, cheetah_final = run_element_tracking(
        "sextupole", ELEMENT_LENGTH, SEXT_K2, NUM_STEPS, radiation=RADIATION_ON
    )
    assert_coordinates_close(pyat_final, cheetah_final, "sextupole_radiation")


def test_quadrupole_fringe_radiation():
    """Test quadrupole with fringe fields and radiation"""
    pyat_final, cheetah_final = run_element_tracking(
        "quadrupole",
        ELEMENT_LENGTH,
        QUAD_K1,
        NUM_STEPS,
        fringe=FRINGE_ON,
        radiation=RADIATION_ON,
    )
    assert_coordinates_close(pyat_final, cheetah_final, "quadrupole_fringe_radiation")


if __name__ == "__main__":
    # Provide backwards compatibility for running with python directly
    import sys

    import pytest as pytest_main

    sys.exit(pytest_main.main(["-v", __file__]))
