
import torch
from cheetah.accelerator.patch import Patch
from cheetah.particles.particle_beam import ParticleBeam

def test_patch_rotation_matrix():
    """
    Test the Patch element functionality.
    """

    pitches = [
        torch.tensor((0.0, 0.0)),
        torch.tensor((torch.pi / 2.0, 0.0)),
        torch.tensor((0.0, torch.pi / 2.0)),
        torch.tensor((0.0, 0.0)),
    ]
    tilts = [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(torch.pi / 2.0)]

    expected_rotation_matrices = [
        torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]), torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]),torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]), torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])]

    for pitch, tilt, expected_matrix in zip(pitches, tilts, expected_rotation_matrices):
        patch = Patch(
            length=torch.tensor(1.0),
            offset=torch.tensor([0.1, 0.2, 0.3]),
            time_offset=torch.tensor(0.5),
            pitch=pitch,
            tilt=tilt,
            E_tot_offset=torch.tensor(0.01),
            E_tot_set=torch.tensor(0.02),
        )
        rotation_matrix = patch.rotation_matrix()
        assert torch.allclose(rotation_matrix, expected_matrix, atol=1e-6), "Rotation matrix is incorrect"


def test_patch_transform_particles():
    """
    Test the Patch element's transform_particles method.
    """
    patch = Patch(
        length=torch.tensor(1.0),
        offset=torch.tensor([0.1, 0.2, 0.3]),
        time_offset=torch.tensor(0.5),
        pitch=torch.tensor((0.0, 0.0)),
        tilt=torch.tensor(0.0),
        E_tot_offset=torch.tensor(0.01),
        E_tot_set=torch.tensor(0.02),
    )

    beam = ParticleBeam(
        particles = torch.rand(10, 7),  # 10 particles with 3D position and 3D momentum
        energy = torch.tensor(1.0)
    )

    transformed_beam = patch.transform_particles(beam)
    assert torch.allclose(transformed_beam.particles[...,-1], beam.particles[...,-1] + 0.1, atol=1e-6), "Particle transformation is incorrect"

