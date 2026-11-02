import pytest
import torch

import cheetah


@pytest.mark.parametrize(
    "pitch, tilt, correct_rotation_matrix",
    [
        (torch.tensor((0.0, 0.0)), torch.tensor(0.0), torch.eye(3)),
        (
            torch.tensor((torch.pi / 2.0, 0.0)),
            torch.tensor(0.0),
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        ),
        (
            torch.tensor((0.0, torch.pi / 2.0)),
            torch.tensor(0.0),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
        ),
        (
            torch.tensor((0.0, 0.0)),
            torch.tensor(torch.pi / 2.0),
            torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ),
    ],
    ids=["no_rotation", "pitch_90_degrees_x", "pitch_90_degrees_y", "tilt_90_degrees"],
)
def test_rotation_matrix(pitch, tilt, correct_rotation_matrix):
    """Test that the `Patch` element's rotation matrix is computed correctly."""
    patch = cheetah.Patch(pitch=pitch, tilt=tilt)
    assert torch.allclose(patch._rotation_matrix(), correct_rotation_matrix, atol=1e-6)


def test_length():
    """
    Test that the Patch element's length property is computed correctly using a simple
    case with no rotation, where the length should equal the z-offset.
    """
    correct_length = 0.3
    patch = cheetah.Patch(offset=torch.tensor([0.1, 0.2, correct_length]))

    assert torch.isclose(patch.length, torch.tensor(correct_length))


def test_transform_particles():
    """
    Test that tracking a beam through a Patch element correctly applies the offset to
    particles placed at the origin.
    """
    # Beam with 10 particles placed at the origin
    incoming = cheetah.ParticleBeam(
        particles=torch.zeros(10, 7), energy=torch.tensor(1.0)
    )

    patch = cheetah.Patch(offset=torch.tensor([0.1, 0.2, 0.3]))

    outgoing = patch.track(incoming)

    assert (outgoing.x == -0.1).all()
    assert (outgoing.px == incoming.px).all()
    assert (outgoing.y == -0.2).all()
    assert (outgoing.py == incoming.py).all()
    assert (outgoing.tau == incoming.tau).all()
    assert (outgoing.p == incoming.p).all()
    assert outgoing.s == 0.3


def test_transform_particles_with_angles():
    # test with angles (no tilt)
    patch_with_angles = cheetah.Patch(
        offset=torch.tensor([0.1, 0.2, 0.3]),
        pitch=torch.tensor((0.5, -0.5)),
        tilt=torch.tensor(0.0),
    )

    assert torch.allclose(
        patch_with_angles.length, torch.tensor(1.7723379e-01), atol=1e-6
    )

    incoming = cheetah.ParticleBeam(
        particles=torch.zeros(
            10, 7
        ),  # 10 particles with 3D position and 3D momentum at the origin
        energy=torch.tensor(1.0e7),
    )

    outgoing = patch_with_angles.track(incoming)

    # Expected offsets from Bmad - note potential issue with the change in energy here
    # TODO: fix potential energy issue
    bmad_offsets = torch.tensor(
        [-5.426011e-02, -4.794255e-01, -2.278988e-01, 4.207355e-01, -5.28948e-02]
    )
    for i, offset in zip(range(5), bmad_offsets):
        assert torch.allclose(
            outgoing.particles[..., i],
            incoming.particles[..., i] + offset,
            atol=1e-6,
        )

    patch_with_angles = cheetah.Patch(
        offset=torch.tensor([0.1, 0.2, 0.0]),
        pitch=torch.tensor((0.5, -0.5)),
        tilt=torch.tensor(0.75),
    )

    incoming = cheetah.ParticleBeam(
        particles=torch.zeros(
            10, 7
        ),  # 10 particles with 3D position and 3D momentum at the origin
        energy=torch.tensor(1.0e9),
    )

    outgoing = patch_with_angles.track(incoming)
    bmad_offsets = torch.tensor(
        [-1.950462e-01, -6.400071e-02, -1.297652e-01, 6.346425e-01, 1.605987e-02]
    )
    for i, offset in zip(range(5), bmad_offsets):
        assert torch.allclose(
            outgoing.particles[..., i],
            incoming.particles[..., i] + offset,
            atol=1e-6,
        )


def test_jacobian():
    patch_with_angles = cheetah.Patch(
        offset=torch.tensor([0.1, 0.2, 0.3]),
        time_offset=torch.tensor(0.0),
        pitch=torch.tensor((0.5, -0.5)),
        tilt=torch.tensor(0.75),
        energy_offset=torch.tensor(0.0),
        energy_setpoint=torch.tensor(0.0),
    )
    energy = torch.tensor(1.0e9)

    def f(x):
        return patch_with_angles.track(
            cheetah.ParticleBeam(particles=x, energy=energy)
        ).particles

    with torch.autograd.set_detect_anomaly(True):
        J = torch.autograd.functional.jacobian(f, torch.zeros((1, 7))).squeeze()

    gt_J = torch.tensor(
        [
            [0.8337550, 0.1918709, 0.5583533, 0.1284931, 0.0000000, 0.0000000],
            [0.0000000, 0.7987913, 0.0000000, 0.5981943, 0.0000000, -0.0640007],
            [-0.7767232, -0.1787462, 1.0371877, 0.2386865, 0.0000000, 0.0000000],
            [0.0000000, -0.4300164, 0.0000000, 0.6421174, 0.0000000, 0.6346425],
            [0.5463025, 0.1257198, -0.6225084, -0.1432570, 1.0000000, 0.0004628],
            [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000],
        ]
    )

    assert torch.allclose(J[:6, :6], gt_J, atol=5e-4, rtol=1e-4)


def test_transform_parameters():
    """
    Test that tracking a `ParameterBeam` through a `Patch` matches the mean and
    covariance of a tracked `ParticleBeam`.
    """
    patch = cheetah.Patch(
        offset=torch.tensor([0.1, 0.2, 0.3]),
        pitch=torch.tensor([0.5, -0.5]),
        tilt=torch.tensor(0.75),
    )
    energy = torch.tensor(1.0e9)

    particles = torch.randn(100_000, 7) * 1.0e-4
    particles[:, 6] = 1.0
    p_beam = cheetah.ParticleBeam(particles=particles, energy=energy)
    param_beam = p_beam.as_parameter_beam()

    p_out = patch.track(p_beam)
    param_out = patch.track(param_beam)
    p_out_as_param = p_out.as_parameter_beam()

    assert torch.allclose(param_out.mu, p_out_as_param.mu, atol=1.0e-5)
    assert torch.allclose(
        param_out.cov[:6, :6], p_out_as_param.cov[:6, :6], atol=1.0e-7, rtol=5.0e-2
    )


def test_patch_clone():
    """Test that `Patch.clone` preserves all defining features."""
    patch = cheetah.Patch(
        offset=torch.tensor([0.1, 0.2, 0.3]),
        time_offset=torch.tensor(1.0e-9),
        pitch=torch.tensor([0.05, -0.05]),
        tilt=torch.tensor(0.2),
        energy_offset=torch.tensor(1.0e6),
        energy_setpoint=torch.tensor(2.0e6),
        drift_to_exit=False,
        name="test_patch",
    )
    cloned = patch.clone()

    assert cloned.name == patch.name
    assert torch.equal(cloned.offset, patch.offset)
    assert torch.equal(cloned.time_offset, patch.time_offset)
    assert torch.equal(cloned.pitch, patch.pitch)
    assert torch.equal(cloned.tilt, patch.tilt)
    assert torch.equal(cloned.energy_offset, patch.energy_offset)
    assert torch.equal(cloned.energy_setpoint, patch.energy_setpoint)
    assert cloned.drift_to_exit == patch.drift_to_exit


def test_patch_lattice_json_roundtrip(tmp_path):
    """Test serialising a `Patch` to LatticeJSON and reloading it."""
    patch = cheetah.Patch(
        offset=torch.tensor([0.1, 0.2, 0.3]),
        time_offset=torch.tensor(1.0e-9),
        pitch=torch.tensor([0.05, -0.05]),
        tilt=torch.tensor(0.2),
        energy_offset=torch.tensor(1.0e6),
        energy_setpoint=torch.tensor(2.0e6),
        drift_to_exit=False,
        name="my_patch",
    )
    segment = cheetah.Segment([patch], name="patch_segment")
    json_path = tmp_path / "patch_lattice.json"
    segment.to_lattice_json(json_path)

    reloaded = cheetah.Segment.from_lattice_json(json_path)
    reloaded_patch = reloaded.my_patch

    assert isinstance(reloaded_patch, cheetah.Patch)
    assert torch.allclose(reloaded_patch.offset, patch.offset)
    assert torch.allclose(reloaded_patch.time_offset, patch.time_offset)
    assert torch.allclose(reloaded_patch.pitch, patch.pitch)
    assert torch.allclose(reloaded_patch.tilt, patch.tilt)
    assert torch.allclose(reloaded_patch.energy_offset, patch.energy_offset)
    assert torch.allclose(reloaded_patch.energy_setpoint, patch.energy_setpoint)
    assert reloaded_patch.drift_to_exit == patch.drift_to_exit


def test_vectorized_patch():
    """Test tracking with a batched `Patch` element."""
    batched_patch = cheetah.Patch(
        offset=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        pitch=torch.tensor([[0.1, -0.1], [0.2, -0.2]]),
        tilt=torch.tensor([0.1, 0.2]),
    )
    incoming = cheetah.ParticleBeam(
        particles=torch.zeros(20, 7), energy=torch.tensor(1.0e9)
    )
    outgoing = batched_patch.track(incoming)
    assert outgoing.particles.shape == (2, 20, 7)
    assert outgoing.s.shape == (2,)


def test_patch_energy_offset_and_setpoint():
    """Test that energy offsets and setpoints scale momentum deviations properly."""
    patch_offset = cheetah.Patch(energy_offset=torch.tensor(2.0e8))
    patch_setpoint = cheetah.Patch(energy_setpoint=torch.tensor(1.2e9))

    incoming = cheetah.ParticleBeam(
        particles=torch.zeros(10, 7), energy=torch.tensor(1.0e9)
    )
    # Give initial delta of 0.05
    incoming.particles[:, 5] = 0.05

    out_offset = patch_offset.track(incoming)
    out_setpoint = patch_setpoint.track(incoming)

    assert torch.isclose(out_offset.energy, torch.tensor(1.2e9))
    assert torch.isclose(out_setpoint.energy, torch.tensor(1.2e9))
    assert torch.allclose(out_offset.particles, out_setpoint.particles)
