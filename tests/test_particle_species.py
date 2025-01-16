import pytest
from scipy.constants import physical_constants

import cheetah


def test_particle_mass():
    """Test that the particle species have the correct mass."""
    proton = cheetah.Species("proton")
    assert (
        proton.mass_eV
        == physical_constants["proton mass energy equivalent in MeV"][0] * 1e6
    )
    antiproton = cheetah.Species("antiproton")
    assert antiproton.mass_eV == proton.mass_eV
    electron = cheetah.Species("electron")
    assert (
        electron.mass_eV
        == physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
    )
    positron = cheetah.Species("positron")
    assert positron.mass_eV == electron.mass_eV

    deuteron = cheetah.Species("deuteron")
    assert (
        deuteron.mass_eV
        == physical_constants["deuteron mass energy equivalent in MeV"][0] * 1e6
    )


def test_custom_particle_species():
    """Test that custom particle species can be defined."""
    muon = cheetah.Species(name="muon", charge=-1, mass=105.6583755 * 1e6)
    assert (
        muon.mass_eV
        == physical_constants["muon mass energy equivalent in MeV"][0] * 1e6
    )
    assert muon.charge_C == -1 * physical_constants["elementary charge"][0]


def test_unknown_particle_species():
    """Test that unknown particle species raise an error."""
    with pytest.raises(ValueError):
        # no charge provided
        cheetah.Species(name="unknown", mass=1e6)
    with pytest.raises(ValueError):
        # no mass provided
        cheetah.Species(name="unknown", charge=1)
    with pytest.raises(ValueError):
        # no charge and mass provided
        cheetah.Species(name="unknown")
