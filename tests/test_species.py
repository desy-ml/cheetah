import pytest
from scipy.constants import physical_constants

import cheetah


@pytest.mark.parametrize("species_name", ["proton", "electron", "deuteron"])
def test_particle_mass(species_name):
    """
    Test that the particle mass is correct for different species by comparing it to the
    value given in `scipy.constants.physical_constants`.
    """
    species = cheetah.Species(species_name)
    assert (
        species.mass_eV
        == physical_constants[f"{species_name} mass energy equivalent in MeV"][0] * 1e6
    )


def test_positron_electron_mass_equal():
    """Test that the mass of an electron and a positron are equal."""
    electron = cheetah.Species("electron")
    positron = cheetah.Species("positron")

    assert electron.mass_eV == positron.mass_eV


def test_antiproton_proton_mass_equal():
    """Test that the mass of a proton and an antiproton are equal."""
    proton = cheetah.Species("proton")
    antiproton = cheetah.Species("antiproton")

    assert proton.mass_eV == antiproton.mass_eV


def test_custom_particle_species():
    """Test that custom particle species can be defined."""
    muon = cheetah.Species(
        name="muon", num_elementary_charges=-1, mass_eV=105.6583755e6
    )
    assert (
        muon.mass_eV
        == physical_constants["muon mass energy equivalent in MeV"][0] * 1e6
    )
    assert muon.charge_coulomb == -1 * physical_constants["elementary charge"][0]


def test_error_on_missing_species_charge():
    """
    Test that an error is raised when a particle species is defined without a charge.
    """
    with pytest.raises(AssertionError):
        cheetah.Species(name="muon", mass_eV=1e6)


def test_error_on_missing_species_mass():
    """
    Test that an error is raised when a particle species is defined without a mass.
    """
    with pytest.raises(AssertionError):
        cheetah.Species(name="muon", num_elementary_charges=1)


def test_error_on_missing_species_charge_and_mass():
    """
    Test that an error is raised when a particle species is defined without a charge and
    mass.
    """
    with pytest.raises(AssertionError):
        cheetah.Species(name="muon")


def test_providing_both_charges():
    """
    Test that an error is raised when a particle species is defined with both charge
    parameters.
    """
    with pytest.raises(AssertionError):
        cheetah.Species(name="muon", num_elementary_charges=1, charge_coulomb=1.0)


def test_providing_both_masses():
    """
    Test that an error is raised when a particle species is defined with both mass
    parameters.
    """
    with pytest.raises(AssertionError):
        cheetah.Species(name="muon", mass_eV=1e6, mass_kg=1e-27)
