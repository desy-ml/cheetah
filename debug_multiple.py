#!/usr/bin/env python3
"""
Debug script to check multiple cavities individually.
"""

import torch
import cheetah


def debug_multiple_cavities():
    """Debug multiple cavities individually"""
    print("=== Debug Multiple Cavities ===")
    
    # Create the exact same setup as in the reproduction test
    parameter_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14), beta_y=torch.tensor(42.0), 
        species=cheetah.Species('proton'), 
        energy=torch.tensor(931.49410242*1.5e6),
    )
    
    # Create three identical cavities
    cavities = [
        cheetah.Cavity(length=torch.tensor(0.2), voltage=torch.tensor(-1.0e7), phase=torch.tensor(-30.), frequency=torch.tensor(81250000.), name="AREACAVM1"),
        cheetah.Cavity(length=torch.tensor(0.2), voltage=torch.tensor(-1.0e7), phase=torch.tensor(-30.), frequency=torch.tensor(81250000.), name="AREACAVM2"),
        cheetah.Cavity(length=torch.tensor(0.2), voltage=torch.tensor(-1.0e7), phase=torch.tensor(-30.), frequency=torch.tensor(81250000.), name="AREACAVM3"),
    ]
    
    print(f"Initial energy: {parameter_beam.energy:.2e} eV")
    
    current_beam = parameter_beam
    for i, cavity in enumerate(cavities):
        print(f"\n--- Cavity {i+1} ({cavity.name}) ---")
        initial_energy = current_beam.energy
        print(f"Input energy: {initial_energy:.2e} eV")
        
        # Debug the transfer map computation
        tm = cavity.first_order_transfer_map(current_beam.energy, current_beam.species)
        print(f"R[5,5] (energy ratio): {tm[..., 5, 5]:.6f}")
        
        # Calculate expected delta_energy
        phi_rad = torch.deg2rad(cavity.phase)
        effective_voltage = -cavity.voltage * current_beam.species.num_elementary_charges
        expected_delta_energy = effective_voltage * torch.cos(phi_rad)
        print(f"cavity.voltage: {cavity.voltage:.2e} V")
        print(f"species.num_elementary_charges: {current_beam.species.num_elementary_charges}")
        print(f"effective_voltage: {effective_voltage:.2e} V")
        print(f"cos(phi): {torch.cos(phi_rad):.6f}")
        print(f"Expected delta_energy: {expected_delta_energy:.2e} eV")
        
        # Track through cavity
        current_beam = cavity.track(current_beam)
        final_energy = current_beam.energy
        actual_energy_change = final_energy - initial_energy
        
        print(f"Output energy: {final_energy:.2e} eV")
        print(f"Actual energy change: {actual_energy_change:.2e} eV")
        
        # Check consistency
        if torch.abs(actual_energy_change - expected_delta_energy) < 1e3:
            print("✓ Energy change matches expectation")
        else:
            print(f"✗ Energy change doesn't match! Diff: {actual_energy_change - expected_delta_energy:.2e} eV")


if __name__ == "__main__":
    debug_multiple_cavities()