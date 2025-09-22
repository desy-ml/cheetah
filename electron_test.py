#!/usr/bin/env python3
"""
Test script to understand electron cavity behavior.
"""

import torch
import cheetah


def test_electron_cavity():
    """Test behavior of a single cavity with electrons"""
    print("=== Electron Cavity Test ===")
    
    # Create electron beam
    electron_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14), beta_y=torch.tensor(42.0), 
        species=cheetah.Species('electron'), 
        energy=torch.tensor(100e6),  # 100 MeV
    )
    
    # Test different voltages
    voltages = [-1.0e7, 1.0e7]
    phase = -30.0
    
    for voltage in voltages:
        print(f"\nVoltage: {voltage:.1e} V, Phase: {phase} degrees")
        
        cavity = cheetah.Cavity(
            length=torch.tensor(0.2), 
            voltage=torch.tensor(voltage), 
            phase=torch.tensor(phase), 
            frequency=torch.tensor(81250000.)
        )
        
        initial_energy = electron_beam.energy
        outgoing_beam = cavity.track(electron_beam)
        final_energy = outgoing_beam.energy
        energy_change = final_energy - initial_energy
        
        print(f"Initial energy: {initial_energy:.2e} eV")
        print(f"Final energy: {final_energy:.2e} eV")
        print(f"Energy change: {energy_change:.2e} eV")
        
        # Calculate expected energy change
        phi_rad = torch.deg2rad(torch.tensor(phase))
        expected_change = voltage * torch.cos(phi_rad) * electron_beam.species.num_elementary_charges
        print(f"Expected change (naive): {expected_change:.2e} eV")
        print(f"Species charge: {electron_beam.species.num_elementary_charges}")
        
        # Check sign according to documentation
        if voltage > 0:
            if energy_change > 0:
                print("✓ Positive voltage gives positive energy change")
            else:
                print("✗ Positive voltage gives negative energy change")
        else:
            if energy_change < 0:
                print("✓ Negative voltage gives negative energy change")  
            else:
                print("✗ Negative voltage gives positive energy change")


if __name__ == "__main__":
    test_electron_cavity()