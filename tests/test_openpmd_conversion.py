import torch

from cheetah import ParticleBeam


def test_particlebeam_to_and_from_particlegroup():

    ref_energy = torch.tensor(1e6)
    beam = ParticleBeam.from_parameters(
        num_particles=10000,
        mu_x=torch.tensor(1e-4),
        sigma_x=torch.tensor(2e-5),
        mu_y=torch.tensor(1e-4),
        sigma_y=torch.tensor(2e-5),
        sigma_p=torch.tensor(1e-4),
        energy=ref_energy,
        total_charge=torch.tensor(1e-9),
        dtype=torch.float64,
    )

    particle_group = beam.to_openpmd_particlegroup()

    # Test that the loaded particle group is the same as the original beam
    beam_loaded = ParticleBeam.from_openpmd_particlegroup(
        particle_group, energy=ref_energy, dtype=torch.float64
    )

    assert beam.num_particles == beam_loaded.num_particles
    assert torch.allclose(beam.particles, beam_loaded.particles)
    assert torch.allclose(beam.particle_charges, beam_loaded.particle_charges)


def test_particlebeam_to_and_from_openpmd_h5():

    ref_energy = torch.tensor(1e6)
    beam = ParticleBeam.from_parameters(
        num_particles=10000,
        mu_x=torch.tensor(1e-4),
        sigma_x=torch.tensor(2e-5),
        mu_y=torch.tensor(1e-4),
        sigma_y=torch.tensor(2e-5),
        sigma_p=torch.tensor(1e-4),
        energy=ref_energy,
        total_charge=torch.tensor(1e-9),
        dtype=torch.float64,
    )

    filename = "tests/resources/test_particlebeam_to_and_from_openpmd_h5.h5"
    beam.save_as_openpmd_h5(filename)

    # Test that the loaded particle group is the same as the original beam
    beam_loaded = ParticleBeam.from_openpmd_file(
        filename, energy=ref_energy, dtype=torch.float64
    )

    assert beam.num_particles == beam_loaded.num_particles
    assert torch.allclose(beam.particles, beam_loaded.particles)
    assert torch.allclose(beam.particle_charges, beam_loaded.particle_charges)

    # Clean up
    import os

    os.remove(filename)
