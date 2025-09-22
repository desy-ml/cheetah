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


def test_multiple_cavity_consistency():
    """
    Test that multiple identical cavities produce identical energy changes.
    This test validates the fix for issue #570 where subsequent cavities
    showed different behavior due to species not being preserved.
    """
    # Create proton beam
    proton_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14), 
        beta_y=torch.tensor(42.0), 
        species=cheetah.Species('proton'), 
        energy=torch.tensor(931.49410242*1.5e6),
    )
    
    # Create three identical cavities
    cavities = [
        cheetah.Cavity(
            length=torch.tensor(0.2), 
            voltage=torch.tensor(-1.0e7), 
            phase=torch.tensor(-30.), 
            frequency=torch.tensor(81250000.)
        ) for _ in range(3)
    ]
    
    current_beam = proton_beam
    energy_changes = []
    
    for cavity in cavities:
        initial_energy = current_beam.energy
        current_beam = cavity.track(current_beam)
        final_energy = current_beam.energy
        energy_change = final_energy - initial_energy
        energy_changes.append(energy_change)
        
        # Verify that species is preserved
        assert current_beam.species.num_elementary_charges == proton_beam.species.num_elementary_charges, \
            "Species charge should be preserved during tracking"
    
    # All energy changes should be identical for identical cavities
    for i in range(1, len(energy_changes)):
        assert torch.abs(energy_changes[i] - energy_changes[0]) < 1e3, \
            f"Energy change from cavity {i+1} should match cavity 1"


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
