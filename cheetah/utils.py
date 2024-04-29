import torch


class UniqueNameGenerator:
    """Generates a unique name given a prefix."""

    def __init__(self, prefix: str):
        self._prefix = prefix
        self._counter = 0

    def __call__(self):
        name = f"{self._prefix}_{self._counter}"
        self._counter += 1
        return name


def generate_3d_uniform_ellipsoid(
    num_particles: torch.Tensor, r_x: torch.Tensor, r_y: torch.Tensor, r_z: torch.Tensor
) -> torch.Tensor:
    """Generate the marcroparticles filling a 3D uniform ellipsoid.
    Note: Now we only consider non-batched version,
    e.g. num_particles, r_x, r_y, r_z are all length-1 tensors.

    :param num_particles: The number of macroparticles to be generated.
    :param r_x: The radius of the ellipsoid along the x-axis.
    :param r_y: The radius of the ellipsoid along the y-axis.
    :param r_z: The radius of the ellipsoid along the z-axis.
    :return: Phase-space coordinates of the particles
        with the shape of (1, num_particles, 7).
    """
    particles = torch.zeros((1, num_particles, 7))
    particles[0, :, 6] = 1

    num_generated = 0

    while num_generated < num_particles:
        Xs = (torch.rand(num_particles) - 0.5) * 2 * r_x
        Ys = (torch.rand(num_particles) - 0.5) * 2 * r_y
        Zs = (torch.rand(num_particles) - 0.5) * 2 * r_z

        # Rejection sampling to get the points inside the ellipsoid.
        indices = (Xs**2 / r_x**2 + Ys**2 / r_y**2 + Zs**2 / r_z**2) <= 1

        num_new_generated = Xs[indices].shape[0]
        num_to_add = min(num_new_generated, int(num_particles - num_generated))

        particles[0, num_generated : num_generated + num_to_add, 0] = Xs[indices][
            :num_to_add
        ]
        particles[0, num_generated : num_generated + num_to_add, 2] = Ys[indices][
            :num_to_add
        ]
        particles[0, num_generated : num_generated + num_to_add, 4] = Zs[indices][
            :num_to_add
        ]
        num_generated += num_to_add

    return particles
