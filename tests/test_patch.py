import torch

from cheetah.accelerator import Segment, Drift
from cheetah.accelerator.patch import Patch
from cheetah.particles.particle_beam import ParticleBeam


def test_tracking_in_segment():
    beam = ParticleBeam(torch.zeros(10, 7), energy=torch.tensor(1.0e9))
    patch = Patch(
        offset=torch.tensor([0.1, 0.2, 0.3]),
        time_offset=torch.tensor(0.0),
        pitch=torch.tensor((0.0, 1.0)),
        tilt=torch.tensor(0.0),
        E_tot_offset=torch.tensor(0.0),
    )

    segment = Segment(
        elements=[
            Drift(length=torch.tensor(1.0)),
            patch,
            Drift(length=torch.tensor(1.0)),
        ],
    )
    segment.track(beam)

    # test tracking with defaults
    patch = Patch()

    segment = Segment(
        elements=[
            Drift(length=torch.tensor(1.0)),
            patch,
            Drift(length=torch.tensor(1.0)),
        ],
    )
    segment.track(beam)

def test_patch_with_vectorization():
    """ test that patch works with vectorized beams """
    beam = ParticleBeam(torch.zeros(4, 10, 7), energy=torch.tensor(1.0e9))
    patch = Patch(
        offset=torch.tensor([0.1, 0.2, 0.3]),
        time_offset=torch.tensor(0.0),
        pitch=torch.tensor((0.0, 1.0)),
        tilt=torch.tensor(0.0),
        E_tot_offset=torch.tensor(0.0),
    )

    segment = Segment(
        elements=[
            Drift(length=torch.tensor(1.0)),
            patch,
            Drift(length=torch.tensor(1.0)),
        ],
    )
    segment.track(beam)

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
    tilts = [
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(torch.pi / 2.0),
    ]

    expected_rotation_matrices = [
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    ]

    for pitch, tilt, expected_matrix in zip(pitches, tilts, expected_rotation_matrices):
        patch = Patch(
            offset=torch.tensor([0.1, 0.2, 0.3]),
            time_offset=torch.tensor(0.5),
            pitch=pitch,
            tilt=tilt,
            E_tot_offset=torch.tensor(0.01),
            E_tot_set=torch.tensor(0.02),
        )
        rotation_matrix = patch.rotation_matrix()
        assert torch.allclose(
            rotation_matrix, expected_matrix, atol=1e-6
        ), "Rotation matrix is incorrect"


def test_patch_length_property():
    """
    Test the Patch element's length property.
    """
    patch = Patch(
        offset=torch.tensor([0.1, 0.2, 0.3]),
        time_offset=torch.tensor(0.0),
        pitch=torch.tensor((0.0, 0.0)),
        tilt=torch.tensor(0.0),
        E_tot_offset=torch.tensor(0.0),
        E_tot_set=torch.tensor(0.0),
    )
    expected_length = 0.3  # since no rotation, length should be z-offset
    assert torch.isclose(
        patch.length, torch.tensor(expected_length), atol=1e-6
    ), "Length property is incorrect"


def test_patch_transform_particles():
    """
    Test the Patch element's transform_particles method.
    """
    patch = Patch(
        offset=torch.tensor([0.1, 0.2, 0.3]),
        time_offset=torch.tensor(0.0),
        pitch=torch.tensor((0.0, 0.0)),
        tilt=torch.tensor(0.0),
        E_tot_offset=torch.tensor(0.0),
        E_tot_set=torch.tensor(0.0),
    )

    beam = ParticleBeam(
        particles=torch.zeros(
            10, 7
        ),  # 10 particles with 3D position and 3D momentum at the origin
        energy=torch.tensor(1.0),
    )

    transformed_beam = patch.transform_particles(beam)
    assert torch.allclose(
        transformed_beam.particles[..., 0], beam.particles[..., 0] - 0.1, atol=1e-6
    ), "Particle transformation is incorrect"
    assert torch.allclose(
        transformed_beam.particles[..., 2], beam.particles[..., 2] - 0.2, atol=1e-6
    ), "Particle transformation is incorrect"

    for i in [1, 3, 4, 5, 6]:
        assert torch.allclose(
            transformed_beam.particles[..., i], beam.particles[..., i], atol=1e-6
        ), "Particle transformation is incorrect"


def test_patch_transform_particles_with_angles():
    # test with angles (no tilt)
    patch_with_angles = Patch(
        offset=torch.tensor([0.1, 0.2, 0.0]),
        time_offset=torch.tensor(0.0),
        pitch=torch.tensor((0.5, -0.5)),
        tilt=torch.tensor(0.0),
        E_tot_offset=torch.tensor(0.0),
        E_tot_set=torch.tensor(0.0),
    )

    beam = ParticleBeam(
        particles=torch.zeros(
            10, 7
        ),  # 10 particles with 3D position and 3D momentum at the origin
        energy=torch.tensor(1.0e9),
    )

    transformed_beam = patch_with_angles.transform_particles(beam)

    # expected offsets from Bmad - note potential issue with the change in energy here
    # TODO: fix potential energy issue
    bmad_offsets = torch.tensor(
        [
            -5.426011e-02,
            -4.794255e-01,
            -2.278988e-01,
            4.207355e-01,
            -1.605987e-02,
            -0.229800e00,
        ]
    )
    for i, offset in zip(range(5), bmad_offsets):
        assert torch.allclose(
            transformed_beam.particles[..., i],
            beam.particles[..., i] + offset,
            atol=1e-6,
        ), "Particle transformation is incorrect"

    patch_with_angles = Patch(
        offset=torch.tensor([0.1, 0.2, 0.0]),
        time_offset=torch.tensor(0.0),
        pitch=torch.tensor((0.5, -0.5)),
        tilt=torch.tensor(0.75),
        E_tot_offset=torch.tensor(0.0),
        E_tot_set=torch.tensor(0.0),
    )

    beam = ParticleBeam(
        particles=torch.zeros(
            10, 7
        ),  # 10 particles with 3D position and 3D momentum at the origin
        energy=torch.tensor(1.0e9),
    )

    transformed_beam = patch_with_angles.transform_particles(beam)
    bmad_offsets = torch.tensor(
        [-1.950462e-01, -6.400071e-02, -1.297652e-01, 6.346425e-01, -1.605987e-02]
    )
    for i, offset in zip(range(5), bmad_offsets):
        assert torch.allclose(
            transformed_beam.particles[..., i],
            beam.particles[..., i] + offset,
            atol=1e-6,
        ), "Particle transformation is incorrect"
