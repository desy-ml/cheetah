import torch

import cheetah


def test_particlebeam_to_and_from_particlegroup():
    """
    Test that a `ParticleBeam` can be converted to an OpenPMD `ParticleGroup` and back,
    checking that the loaded `ParticleBeam` is the same as the original.
    """
    reference_energy = torch.tensor(1e6)

    original_cheetah_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        mu_x=torch.tensor(1e-4),
        sigma_x=torch.tensor(2e-5),
        mu_y=torch.tensor(1e-4),
        sigma_y=torch.tensor(2e-5),
        sigma_p=torch.tensor(1e-4),
        energy=reference_energy,
        total_charge=torch.tensor(1e-9),
        dtype=torch.float64,
    )
    openpmd_particle_group = original_cheetah_beam.to_openpmd_particlegroup()
    loaded_cheetah_beam = cheetah.ParticleBeam.from_openpmd_particlegroup(
        openpmd_particle_group, energy=reference_energy, dtype=torch.float64
    )

    assert original_cheetah_beam.num_particles == loaded_cheetah_beam.num_particles
    assert torch.allclose(
        original_cheetah_beam.particles, loaded_cheetah_beam.particles
    )
    assert torch.allclose(
        original_cheetah_beam.particle_charges, loaded_cheetah_beam.particle_charges
    )


def test_particlebeam_to_and_from_openpmd_h5(tmp_path):
    """
    Test that a `ParticleBeam` can be saved to an OpenPMD HDF5 file and loaded back,
    checking that the loaded `ParticleBeam` is the same as the original.
    """
    reference_energy = torch.tensor(1e6)

    original_cheetah_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        mu_x=torch.tensor(1e-4),
        sigma_x=torch.tensor(2e-5),
        mu_y=torch.tensor(1e-4),
        sigma_y=torch.tensor(2e-5),
        sigma_p=torch.tensor(1e-4),
        energy=reference_energy,
        total_charge=torch.tensor(1e-9),
        dtype=torch.float64,
    )
    original_cheetah_beam.save_as_openpmd_h5(tmp_path / "particlegroup.h5")
    loaded_cheetah_beam = cheetah.ParticleBeam.from_openpmd_file(
        tmp_path / "particlegroup.h5", energy=reference_energy, dtype=torch.float64
    )

    assert original_cheetah_beam.num_particles == loaded_cheetah_beam.num_particles
    assert torch.allclose(
        original_cheetah_beam.particles, loaded_cheetah_beam.particles
    )
    assert torch.allclose(
        original_cheetah_beam.particle_charges, loaded_cheetah_beam.particle_charges
    )
