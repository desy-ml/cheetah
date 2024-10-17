from typing import Optional

from scipy.constants import physical_constants

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
proton_mass_eV = physical_constants["proton mass energy equivalent in MeV"][0] * 1e6
deuteron_mass_eV = physical_constants["deuteron mass energy equivalent in MeV"][0] * 1e6
elementary_charge = physical_constants["elementary charge"][0]
eV_to_kg = physical_constants["electron volt-kilogram relationship"][0]


known_particle_species = {
    "electron": {"charge": -1, "mass": electron_mass_eV},
    "positron": {"charge": 1, "mass": electron_mass_eV},
    "proton": {"charge": 1, "mass": proton_mass_eV},
    "antiproton": {"charge": -1, "mass": proton_mass_eV},
    "deuteron": {"charge": 1, "mass": deuteron_mass_eV},
}


class Species:
    """Class to store information about a particle species.

    :param name: Name of the particle species.
        One of `['electron', 'positron', 'proton', 'antiproton', 'deuteron']`.
        Custom particle species like ions can be defined by providing charge and mass.
    :param charge: Charge of the particle species in unit of elementary charge e.
    :param mass: Mass of the particle species in eV.
    """

    def __init__(
        self, name: str, charge: Optional[int] = None, mass: Optional[float] = None
    ) -> None:
        if name in known_particle_species:
            self.name = name
            self._charge_e = known_particle_species[name]["charge"]
            self._mass = known_particle_species[name]["mass"]
        else:
            if charge is None or mass is None:
                raise ValueError(
                    f"Unknown particle species '{name}'. "
                    " Please provide charge and mass."
                )
            self.name = name
            self._charge_e = charge
            self._mass = mass

    @property
    def mass_eV(self) -> float:
        return self._mass

    @property
    def mass_kg(self) -> float:
        return self._mass * eV_to_kg

    @property
    def charge_C(self) -> float:
        """Return the charge of the particle species in Coulombs."""
        return self._charge_e * elementary_charge

    @property
    def charge_e(self) -> int:
        """Return the charge of the particle species in elementary charges."""
        return self._charge_e

    def __repr__(self) -> str:
        return (
            f"Species(name= {self.name}, charge= {self.charge_e} e,"
            f" mass= {self.mass_eV} eV)"
        )
