import torch

from cheetah import ParticleBeam


def test_generate_uniform_ellipsoid():
    radius_x = torch.tensor([1e-3])
    radius_y = torch.tensor([1e-4])
    radius_s = torch.tensor([1e-5])

    num_particles = torch.tensor(1_000_000)
    parray = ParticleBeam.uniform_3d_ellispoid(
        num_particles=num_particles,
        radius_x=radius_x,
        radius_y=radius_y,
        radius_s=radius_s,
    )

    particles = parray.particles

    assert particles.shape == (1, num_particles, 7)
    assert torch.all(particles[0, :, 6] == 1)

    assert torch.max(torch.abs(particles[0, :, 0])) <= radius_x
    assert torch.max(torch.abs(particles[0, :, 2])) <= radius_y
    assert torch.max(torch.abs(particles[0, :, 4])) <= radius_s


def test_generate_uniform_ellipsoid_batched():
    radius_x = torch.tensor([1e-3, 2e-3])
    radius_y = torch.tensor([1e-4, 2e-4])
    radius_s = torch.tensor([1e-5, 2e-5])

    num_particles = torch.tensor(1_000_000)
    parray = ParticleBeam.uniform_3d_ellispoid(
        num_particles=num_particles,
        radius_x=radius_x,
        radius_y=radius_y,
        radius_s=radius_s,
    )

    particles = parray.particles

    assert particles.shape == (2, 1_000_000, 7)
    assert torch.all(particles[:, :, 6] == 1)

    assert torch.all(
        torch.max(torch.abs(particles[:, :, 0]), dim=-1).values <= radius_x
    )
    assert torch.all(
        torch.max(torch.abs(particles[:, :, 2]), dim=-1).values <= radius_y
    )
    assert torch.all(
        torch.max(torch.abs(particles[:, :, 4]), dim=-1).values <= radius_s
    )
