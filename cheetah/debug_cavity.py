#!/usr/bin/env python3
"""
Debug script to understand the transfer map vs energy calculation.
"""

import torch
import cheetah


def debug_cavity_calculation():
    """Debug the cavity calculation step by step"""
    print("=== Debug Cavity Calculation ===")
    
    # Create proton beam
    proton_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14), beta_y=torch.tensor(42.0), 
        species=cheetah.Species('proton'), 
        energy=torch.tensor(931.49410242*1.5e6),
    )
    
    voltage = -1.0e7
    phase = -30.0
    
    cavity = cheetah.Cavity(
        length=torch.tensor(0.2), 
        voltage=torch.tensor(voltage), 
        phase=torch.tensor(phase), 
        frequency=torch.tensor(81250000.)
    )
    
    print(f"Voltage: {voltage:.1e} V, Phase: {phase} degrees")
    print(f"Initial energy: {proton_beam.energy:.2e} eV")
    print(f"Species charge: {proton_beam.species.num_elementary_charges}")
    
    # Calculate what the transfer map should give
    tm = cavity.first_order_transfer_map(proton_beam.energy, proton_beam.species)
    print(f"Transfer map shape: {tm.shape}")
    print(f"Transfer map R[5,5] (energy scaling): {tm[..., 5, 5]}")
    
    # Manually calculate the expected energy change using the same formula as in _cavity_rmatrix
    phi_rad = torch.deg2rad(torch.tensor(phase))
    effective_voltage = -voltage * proton_beam.species.num_elementary_charges
    delta_energy_expected = effective_voltage * torch.cos(phi_rad)
    print(f"Expected delta_energy from _cavity_rmatrix formula: {delta_energy_expected:.2e} eV")
    
    # Calculate what the track method computes
    outgoing_beam = cavity.track(proton_beam)
    actual_energy_change = outgoing_beam.energy - proton_beam.energy
    print(f"Actual energy change from track(): {actual_energy_change:.2e} eV")
    
    # The issue might be that the transfer map is computed with initial energy, 
    # but applied to a beam that has already been modified
    

if __name__ == "__main__":
    debug_cavity_calculation()