#!/usr/bin/env python3
"""
Test to validate the fix for issue #570 - cavity element consistency.
"""

import torch
import cheetah


def test_multiple_cavity_consistency():
    """
    Test that multiple identical cavities produce identical energy changes.
    This test validates the fix for issue #570.
    """
    # Create proton beam
    proton_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14), 
        beta_y=torch.tensor(42.0), 
        species=cheetah.Species('proton'), 
        energy=torch.tensor(931.49410242*1.5e6),
    )
    
    # Test case 1: All cavities have same voltage
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
            f"Energy change from cavity {i+1} ({energy_changes[i]:.2e}) should match cavity 1 ({energy_changes[0]:.2e})"
    
    print("✓ Test case 1 passed: All identical cavities produce identical energy changes")
    
    # Test case 2: Different voltages should produce different energy changes
    cavity_pos = cheetah.Cavity(
        length=torch.tensor(0.2), 
        voltage=torch.tensor(1.0e7), 
        phase=torch.tensor(-30.), 
        frequency=torch.tensor(81250000.)
    )
    
    cavity_neg = cheetah.Cavity(
        length=torch.tensor(0.2), 
        voltage=torch.tensor(-1.0e7), 
        phase=torch.tensor(-30.), 
        frequency=torch.tensor(81250000.)
    )
    
    # Track through both cavities with same initial beam
    beam_pos = cavity_pos.track(proton_beam)
    beam_neg = cavity_neg.track(proton_beam)
    
    energy_change_pos = beam_pos.energy - proton_beam.energy
    energy_change_neg = beam_neg.energy - proton_beam.energy
    
    # Energy changes should have opposite signs
    assert energy_change_pos * energy_change_neg < 0, \
        "Opposite voltages should produce opposite energy changes"
    
    print("✓ Test case 2 passed: Opposite voltages produce opposite energy changes")
    
    # Test case 3: Test with electrons to ensure fix doesn't break electron behavior
    electron_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14), 
        beta_y=torch.tensor(42.0), 
        species=cheetah.Species('electron'), 
        energy=torch.tensor(100e6),
    )
    
    electron_out = cavity_pos.track(electron_beam)
    assert electron_out.species.num_elementary_charges == electron_beam.species.num_elementary_charges, \
        "Electron species should be preserved"
    
    print("✓ Test case 3 passed: Electron species preserved correctly")
    
    print("\n🎉 All tests passed! Issue #570 is fixed.")


if __name__ == "__main__":
    test_multiple_cavity_consistency()