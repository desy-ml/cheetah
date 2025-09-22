#!/usr/bin/env python3
"""
Test script to reproduce the cavity issue described in issue #570.
"""

import torch
import cheetah


def test_proton_beam_case1():
    """Case 1: Proton beam, all three cavities set to +1e7 V"""
    print("=== Case 1: Proton beam, all three cavities set to +1e7 V ===")
    
    parameter_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14), beta_y=torch.tensor(42.0), 
        species=cheetah.Species('proton'), 
        energy=torch.tensor(931.49410242*1.5e6),
    )
    
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.204)),
            cheetah.Drift(length=torch.tensor(0.122)),
            cheetah.Drift(length=torch.tensor(0.179)),
            cheetah.Cavity(length=torch.tensor(0.2), voltage=torch.tensor(-1.0e7), phase=torch.tensor(-30.), frequency=torch.tensor(81250000.), name="AREACAVM1"),
            cheetah.Drift(length=torch.tensor(0.45)),
            cheetah.Cavity(length=torch.tensor(0.2), voltage=torch.tensor(-1.0e7), phase=torch.tensor(-30.), frequency=torch.tensor(81250000.), name="AREACAVM2"),
            cheetah.Drift(length=torch.tensor(0.45)),
            cheetah.Cavity(length=torch.tensor(0.2), voltage=torch.tensor(-1.0e7), phase=torch.tensor(-30.), frequency=torch.tensor(81250000.), name="AREACAVM3"),
            cheetah.Drift(length=torch.tensor(0.45)),
            cheetah.Screen(name="AREABSCR1"),
        ]
    )
    
    print("Initial energy:", parameter_beam.energy)
    
    # Track through each cavity individually
    current_beam = parameter_beam
    for i, element in enumerate(segment.elements):
        if isinstance(element, cheetah.Cavity):
            previous_energy = current_beam.energy
            current_beam = element.track(current_beam)
            energy_change = current_beam.energy - previous_energy
            print(f"Cavity {element.name}: Energy change = {energy_change:.2e} eV (from {previous_energy:.2e} to {current_beam.energy:.2e})")
        else:
            current_beam = element.track(current_beam)
    
    return current_beam


def test_proton_beam_case2():
    """Case 2: Proton beam, first cavity +1e7 V, second and third -1e7 V"""
    print("\n=== Case 2: Proton beam, first cavity +1e7 V, second and third -1e7 V ===")
    
    parameter_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(3.14), beta_y=torch.tensor(42.0), 
        species=cheetah.Species('proton'), 
        energy=torch.tensor(931.49410242*1.5e6),
    )
    
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.204)),
            cheetah.Drift(length=torch.tensor(0.122)),
            cheetah.Drift(length=torch.tensor(0.179)),
            cheetah.Cavity(length=torch.tensor(0.2), voltage=torch.tensor(-1.0e7), phase=torch.tensor(-30.), frequency=torch.tensor(81250000.), name="AREACAVM1"),
            cheetah.Drift(length=torch.tensor(0.45)),
            cheetah.Cavity(length=torch.tensor(0.2), voltage=torch.tensor(1.0e7), phase=torch.tensor(-30.), frequency=torch.tensor(81250000.), name="AREACAVM2"),
            cheetah.Drift(length=torch.tensor(0.45)),
            cheetah.Cavity(length=torch.tensor(0.2), voltage=torch.tensor(1.0e7), phase=torch.tensor(-30.), frequency=torch.tensor(81250000.), name="AREACAVM3"),
            cheetah.Drift(length=torch.tensor(0.45)),
            cheetah.Screen(name="AREABSCR1"),
        ]
    )
    
    print("Initial energy:", parameter_beam.energy)
    
    # Track through each cavity individually
    current_beam = parameter_beam
    for i, element in enumerate(segment.elements):
        if isinstance(element, cheetah.Cavity):
            previous_energy = current_beam.energy
            current_beam = element.track(current_beam)
            energy_change = current_beam.energy - previous_energy
            print(f"Cavity {element.name}: Energy change = {energy_change:.2e} eV (from {previous_energy:.2e} to {current_beam.energy:.2e})")
        else:
            current_beam = element.track(current_beam)
    
    return current_beam


if __name__ == "__main__":
    test_proton_beam_case1()
    test_proton_beam_case2()