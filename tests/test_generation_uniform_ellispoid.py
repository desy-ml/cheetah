import torch

from cheetah.utils import generate_3d_uniform_ellipsoid


def test_generate_uniform_ellipsoid():
    radius_x = torch.tensor(1e-3)
    radius_y = torch.tensor(1e-4)
    radius_s = torch.tensor(1e-5)

    num_particles = 1_000_000
    particles = generate_3d_uniform_ellipsoid(
        num_particles, radius_x, radius_y, radius_s
    )

    assert particles.shape == (1, num_particles, 7)
    assert torch.all(particles[0, :, 6] == 1)

    assert torch.max(torch.abs(particles[0, :, 0])) <= radius_x
    assert torch.max(torch.abs(particles[0, :, 2])) <= radius_y
    assert torch.max(torch.abs(particles[0, :, 4])) <= radius_s
