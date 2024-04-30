import torch
from icecream import ic

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
    sigma_xp = torch.tensor([2e-7, 1e-7])
    sigma_yp = torch.tensor([3e-7, 2e-7])
    sigma_p = torch.tensor([0.000001, 0.000002])
    energy = torch.tensor([1e7, 2e7])
    total_charge = torch.tensor([1e-9, 3e-9])

    num_particles = torch.tensor(1_000_000)
    beam = ParticleBeam.uniform_3d_ellispoid(
        num_particles=num_particles,
        radius_x=radius_x,
        radius_y=radius_y,
        radius_s=radius_s,
        sigma_xp=sigma_xp,
        sigma_yp=sigma_yp,
        sigma_p=sigma_p,
        energy=energy,
        total_charge=total_charge,
    )

    assert beam.num_particles == num_particles
    assert torch.all(beam.xs.abs().transpose(0, 1) <= radius_x)
    assert torch.all(beam.ys.abs().transpose(0, 1) <= radius_y)
    assert torch.all(beam.ss.abs().transpose(0, 1) <= radius_s)
    assert torch.allclose(beam.sigma_xp, sigma_xp)
    assert torch.allclose(beam.sigma_yp, sigma_yp)
    assert torch.allclose(beam.sigma_p, sigma_p)
    assert torch.allclose(beam.energy, energy)
    assert torch.allclose(beam.total_charge, total_charge)
