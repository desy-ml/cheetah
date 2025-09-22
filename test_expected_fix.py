#!/usr/bin/env python3
"""
Test script to validate the proposed fix.
"""

import torch
import cheetah


def test_fixed_cavity_behavior():
    """Test the expected behavior after the fix"""
    print("=== Test Expected Behavior After Fix ===")
    
    # Test with both electrons and protons
    species_list = [
        ('electron', cheetah.Species('electron'), 100e6),
        ('proton', cheetah.Species('proton'), 931.49410242*1.5e6)
    ]
    
    voltages = [-1.0e7, 1.0e7]
    phase = -30.0
    
    for species_name, species, energy in species_list:
        print(f"\n=== {species_name.upper()} ===")
        print(f"Species charge: {species.num_elementary_charges}")
        
        beam = cheetah.ParticleBeam.from_twiss(
            beta_x=torch.tensor(3.14), beta_y=torch.tensor(42.0), 
            species=species, 
            energy=torch.tensor(energy),
        )
        
        for voltage in voltages:
            print(f"\nVoltage: {voltage:.1e} V, Phase: {phase} degrees")
            
            cavity = cheetah.Cavity(
                length=torch.tensor(0.2), 
                voltage=torch.tensor(voltage), 
                phase=torch.tensor(phase), 
                frequency=torch.tensor(81250000.)
            )
            
            initial_energy = beam.energy
            outgoing_beam = cavity.track(beam)
            final_energy = outgoing_beam.energy
            energy_change = final_energy - initial_energy
            
            print(f"Energy change: {energy_change:.2e} eV")
            
            # Calculate what the energy change should be after the fix
            phi_rad = torch.deg2rad(torch.tensor(phase))
            effective_voltage = -voltage * species.num_elementary_charges
            expected_change = effective_voltage * torch.cos(phi_rad)
            
            print(f"Expected after fix: {expected_change:.2e} eV")
            
            # Check if positive voltage gives positive energy change
            if voltage > 0:
                if expected_change > 0:
                    print("✓ After fix: Positive voltage should give positive energy change")
                else:
                    print("✗ After fix: Positive voltage should give positive energy change")


if __name__ == "__main__":
    test_fixed_cavity_behavior()