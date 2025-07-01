import torch
from scipy.constants import physical_constants
from torch import nn

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
proton_mass_eV = physical_constants["proton mass energy equivalent in MeV"][0] * 1e6
deuteron_mass_eV = physical_constants["deuteron mass energy equivalent in MeV"][0] * 1e6
elementary_charge = physical_constants["elementary charge"][0]
eV_to_kg = physical_constants["electron volt-kilogram relationship"][0]


class Species(nn.Module):
    """
    Named particle species defined by charge and mass.

    :param name: Name of the particle species. For species in `Species.known`, charge
        and mass are populated automatically. Custom particle species like ions can be
        defined by providing charge and mass.
    :param num_elementary_charges: Charge of the particle species in units of elementary
        charge e. NOTE: Only one of `num_elementary_charges` and `charge_coulomb`
        should be provided.
    :param charge_coulomb: Charge of the particle species in Coulombs. NOTE: Only one
        of `num_elementary_charges` and `charge_coulomb` should be provided.
    :param mass_eV: Mass of the particle species in eV. NOTE: Only one of `mass_eV` and
        `mass_kg` should be provided.
    :param mass_kg: Mass of the particle species in kg. NOTE: Only one of `mass_eV` and
        `mass_kg` should be provided.
    """

    known = {
        "electron": {"num_elementary_charges": -1, "mass_eV": electron_mass_eV},
        "positron": {"num_elementary_charges": 1, "mass_eV": electron_mass_eV},
        "proton": {"num_elementary_charges": 1, "mass_eV": proton_mass_eV},
        "antiproton": {"num_elementary_charges": -1, "mass_eV": proton_mass_eV},
        "deuteron": {"num_elementary_charges": 1, "mass_eV": deuteron_mass_eV},
    }

    def __init__(
        self,
        name: str,
        num_elementary_charges: torch.Tensor | None = None,
        charge_coulomb: torch.Tensor | None = None,
        mass_eV: torch.Tensor | None = None,
        mass_kg: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if name in self.__class__.known:  # Known particle species
            assert all(
                [
                    num_elementary_charges is None,
                    charge_coulomb is None,
                    mass_eV is None,
                    mass_kg is None,
                ]
            ), "Known particle species should not have charge and mass provided."

            self.name = name
            self.register_buffer_or_parameter(
                "num_elementary_charges",
                torch.as_tensor(
                    self.__class__.known[name]["num_elementary_charges"],
                    device=device,
                    dtype=dtype,
                ),
            )
            self.register_buffer_or_parameter(
                "mass_eV",
                torch.as_tensor(
                    self.__class__.known[name]["mass_eV"], device=device, dtype=dtype
                ),
            )
        else:  # Custom particle species
            assert any(
                [num_elementary_charges is not None, charge_coulomb is not None]
            ) and any(
                [mass_eV is not None, mass_kg is not None]
            ), "Custom particle species should have charge and mass provided."
            assert not all(
                [num_elementary_charges is not None, charge_coulomb is not None]
            ), "Only one of charge_elementary and charge_coulomb should be provided."
            assert not all(
                [mass_eV is not None, mass_kg is not None]
            ), "Only one of mass_eV and mass_kg should be provided."

            self.name = name
            self.register_buffer_or_parameter(
                "num_elementary_charges",
                torch.as_tensor(
                    num_elementary_charges or charge_coulomb / elementary_charge,
                    device=device,
                    dtype=dtype,
                ),
            )
            self.register_buffer_or_parameter(
                "mass_eV",
                torch.as_tensor(
                    mass_eV or mass_kg * eV_to_kg, device=device, dtype=dtype
                ),
            )

    @property
    def mass_kg(self) -> torch.Tensor:
        """Mass of the particle species in kg."""
        return self.mass_eV * eV_to_kg

    @mass_kg.setter
    def mass_kg(self, value: torch.Tensor) -> None:
        self.mass_eV = value / eV_to_kg

    @property
    def charge_coulomb(self) -> torch.Tensor:
        """Charge of the particle species in Coulombs."""
        return self.num_elementary_charges * elementary_charge

    @charge_coulomb.setter
    def charge_coulomb(self, value: torch.Tensor) -> None:
        self.num_elementary_charges = value / elementary_charge

    def register_buffer_or_parameter(
        self, name: str, value: torch.Tensor | nn.Parameter
    ) -> None:
        """
        Register a buffer or parameter with the given name and value. Automatically
        selects the correct method from `register_buffer` or `register_parameter` based
        on the type of `value`.

        :param name: Name of the buffer or parameter.
        :param value: Value of the buffer or parameter.
        :param default: Default value of the buffer.
        """
        if isinstance(value, nn.Parameter):
            self.register_parameter(name, value)
        else:
            self.register_buffer(name, value)

    def clone(self) -> "Species":
        """Return a copy of the species."""
        if self.name in self.__class__.known:
            # Manually passing charge and mass is not allowed for known species to avoid
            # users defining confusing species. Since no tensors are passed in that case
            # dtype and device have to be passed manually.
            return self.__class__(
                name=self.name, device=self.mass_eV.device, dtype=self.mass_eV.dtype
            )
        else:
            return self.__class__(
                name=self.name,
                num_elementary_charges=self.num_elementary_charges.clone(),
                mass_eV=self.mass_eV.clone(),
            )

    def __repr__(self) -> str:
        return (
            f"Species(name={repr(self.name)}, "
            + f"num_elementary_charges={repr(self.num_elementary_charges)}, "
            + f"mass_eV={repr(self.mass_eV)})"
        )
