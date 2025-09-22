import pytest
import torch

import cheetah


def test_assert_ei_greater_zero():
    """
    Reproduces

    ```
       1127 Ef = (energy + delta_energy) / particle_mass_eV
       1128 Ep = (Ef - Ei) / self.length  # Derivative of the energy
    -> 1129 assert Ei > 0, "Initial energy must be larger than 0"
       1131 alpha = torch.sqrt(eta / 8) / torch.cos(phi) * torch.log(Ef / Ei)
       1133 r11 = torch.cos(alpha) - torch.sqrt(2 / eta) * torch.cos(phi) * torch.sin(alpha)   # noqa: E501

    RuntimeError: Boolean value of Tensor with more than one value is ambiguous
    ```
    """
    cavity = cheetah.Cavity(
        length=torch.tensor([3.0441, 3.0441, 3.0441]),
        voltage=torch.tensor([48198468.0, 48198468.0, 48198468.0]),
        phase=torch.tensor([48198468.0, 48198468.0, 48198468.0]),
        frequency=torch.tensor([2.8560e09, 2.8560e09, 2.8560e09]),
        name="k26_2a",
    )
    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, sigma_x=torch.tensor(1e-5)
    )

    _ = cavity.track(beam)


@pytest.mark.parametrize(
    ("voltage", "phase"),
    [
        (torch.tensor(0.0), torch.tensor([-90.0, 90.0])),
        (torch.tensor([0.0, 1e6]), torch.tensor([[-90.0], [0.0], [90.0], [180.0]])),
        (torch.tensor(1e6), torch.tensor([0.0, 180.0])),
    ],
    ids=["off", "mixed", "on"],
)
@pytest.mark.parametrize("cavity_type", ["standing_wave", "traveling_wave"])
def test_vectorized_inactive_cavity(cavity_type, voltage, phase):
    """
    Tests that a vectorised cavity with zero voltage or off-crest phase does not produce
    NaNs and that switched-off cavities can be vectorized with switched-on.

    This was a bug introduced during the vectorisation of Cheetah, when the special
    case of zero was removed and the `_cavity_rmatrix` method was also used in the case
    of zero voltage or off-crest phase. The latter produced NaNs in the transfer matrix.
    """
    cavity = cheetah.Cavity(
        cavity_type=cavity_type,
        length=torch.tensor(3.0441),
        voltage=voltage,
        phase=phase,
        frequency=torch.tensor(2.8560e09),
    ).to(torch.float64)
    incoming = cheetah.ParameterBeam.from_parameters(
        sigma_x=torch.tensor(4.8492e-06),
        sigma_px=torch.tensor(1.5603e-07),
        sigma_y=torch.tensor(4.1209e-07),
        sigma_py=torch.tensor(1.1035e-08),
        sigma_tau=torch.tensor(1.0000e-10),
        sigma_p=torch.tensor(1.0000e-06),
        energy=torch.tensor(8.0000e09),
        total_charge=torch.tensor(0.0),
    ).to(torch.float64)

    outgoing = cavity.track(incoming)

    assert not torch.isnan(
        cavity.first_order_transfer_map(incoming.energy, incoming.species)
    ).any()

    assert not torch.isnan(outgoing.sigma_x).any()
    assert not torch.isnan(outgoing.sigma_y).any()
    assert not torch.isnan(outgoing.beta_x).any()
    assert not torch.isnan(outgoing.beta_y).any()


def test_multiple_cavities_preserve_species():
    """
    Test that multiple cavities preserve the incoming beam species.
    This test addresses the issue where subsequent cavities would receive
    electron species instead of the original beam species.
    
    Regression test for GitHub issue #570.
    """
    # Test with proton beam
    proton_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14),
        beta_y=torch.tensor(42.0),
        species=cheetah.Species("proton"),
        energy=torch.tensor(931.49410242 * 1.5e6),
    )

    # Create three identical cavities
    cavity1 = cheetah.Cavity(
        length=torch.tensor(0.2),
        voltage=torch.tensor(-1.0e7),
        phase=torch.tensor(-30.0),
        frequency=torch.tensor(81250000.0),
        name="CAVITY1",
    )
    cavity2 = cheetah.Cavity(
        length=torch.tensor(0.2),
        voltage=torch.tensor(-1.0e7),
        phase=torch.tensor(-30.0),
        frequency=torch.tensor(81250000.0),
        name="CAVITY2",
    )
    cavity3 = cheetah.Cavity(
        length=torch.tensor(0.2),
        voltage=torch.tensor(-1.0e7),
        phase=torch.tensor(-30.0),
        frequency=torch.tensor(81250000.0),
        name="CAVITY3",
    )

    # Track through each cavity
    beam1 = cavity1.track(proton_beam)
    beam2 = cavity2.track(beam1)
    beam3 = cavity3.track(beam2)

    # Verify species is preserved
    assert beam1.species.name == "proton"
    assert beam2.species.name == "proton"
    assert beam3.species.name == "proton"

    # Verify charge is preserved
    assert beam1.species.num_elementary_charges == 1
    assert beam2.species.num_elementary_charges == 1
    assert beam3.species.num_elementary_charges == 1

    # Verify consistent acceleration behavior (all cavities should accelerate)
    initial_energy = proton_beam.energy
    energy1 = beam1.energy
    energy2 = beam2.energy
    energy3 = beam3.energy

    # All energy changes should be positive and approximately equal
    energy_gain1 = energy1 - initial_energy
    energy_gain2 = energy2 - energy1
    energy_gain3 = energy3 - energy2

    assert energy_gain1 > 0, "First cavity should accelerate proton"
    assert energy_gain2 > 0, "Second cavity should accelerate proton"
    assert energy_gain3 > 0, "Third cavity should accelerate proton"

    # Energy gains should be approximately equal (within 1% tolerance)
    assert torch.allclose(energy_gain1, energy_gain2, rtol=0.01)
    assert torch.allclose(energy_gain2, energy_gain3, rtol=0.01)


def test_multiple_cavities_preserve_species_parameter_beam():
    """
    Test that multiple cavities preserve species for ParameterBeam as well.
    
    Regression test for GitHub issue #570.
    """
    # Test with proton ParameterBeam
    proton_beam = cheetah.ParameterBeam.from_twiss(
        beta_x=torch.tensor(3.14),
        beta_y=torch.tensor(42.0),
        species=cheetah.Species("proton"),
        energy=torch.tensor(931.49410242 * 1.5e6),
    )

    # Create cavity
    cavity = cheetah.Cavity(
        length=torch.tensor(0.2),
        voltage=torch.tensor(-1.0e7),
        phase=torch.tensor(-30.0),
        frequency=torch.tensor(81250000.0),
    )

    # Track through cavity
    outgoing_beam = cavity.track(proton_beam)

    # Verify species is preserved
    assert outgoing_beam.species.name == "proton"
    assert outgoing_beam.species.num_elementary_charges == 1


def test_multiple_cavities_custom_species():
    """
    Test that multiple cavities preserve custom species (e.g., Argon ions).
    
    Regression test for GitHub issue #570.
    """
    # Create custom Argon 8+ species
    Ar_amu = 39.962383122
    species_Ar8plus = cheetah.Species(
        name="Ar8plus",
        num_elementary_charges=torch.tensor(8),
        mass_eV=torch.tensor(Ar_amu * 931.49410242e6),
    )

    # Create beam with custom species
    argon_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14),
        beta_y=torch.tensor(42.0),
        species=species_Ar8plus,
        energy=torch.tensor((931.49410242 + 0.5) * 1.0e6 * Ar_amu),
    )

    # Create cavity
    cavity = cheetah.Cavity(
        length=torch.tensor(0.2),
        voltage=torch.tensor(-1.0e7),
        phase=torch.tensor(-30.0),
        frequency=torch.tensor(81250000.0),
    )

    # Track through cavity
    outgoing_beam = cavity.track(argon_beam)

    # Verify species is preserved
    assert outgoing_beam.species.name == "Ar8plus"
    assert outgoing_beam.species.num_elementary_charges == 8
    assert torch.allclose(
        outgoing_beam.species.mass_eV,
        torch.tensor(Ar_amu * 931.49410242e6),
        rtol=1e-6,
    )
