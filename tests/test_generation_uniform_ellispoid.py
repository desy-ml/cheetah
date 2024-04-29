import torch

from cheetah.utils import generate_3d_uniform_ellipsoid


def test_generate_uniform_ellipsoid():
    r_x = torch.tensor(1e-3)
    r_y = torch.tensor(1e-4)
    r_z = torch.tensor(1e-5)

    num_particles = 1_000_000
    particles = generate_3d_uniform_ellipsoid(num_particles, r_x, r_y, r_z)

    assert particles.shape == (1, num_particles, 7)
    assert torch.all(particles[0, :, 6] == 1)

    assert torch.max(torch.abs(particles[0, :, 0])) <= r_x
    assert torch.max(torch.abs(particles[0, :, 2])) <= r_y
    assert torch.max(torch.abs(particles[0, :, 4])) <= r_z
