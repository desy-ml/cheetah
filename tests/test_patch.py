import torch

import cheetah


def test_transform_particle_positions():
    """
    Test that tracking a beam through a `Patch` element correctly applies the offset to
    particles placed at the origin, and that the length equals the longitudinal offset.
    """
    # Beam with 10 particles placed at the origin
    incoming = cheetah.ParticleBeam(
        particles=torch.zeros(10, 7), energy=torch.tensor(1.0e9)
    )

    patch = cheetah.Patch(position_offset=torch.tensor([0.1, 0.2, 0.3]))

    torch.testing.assert_close(patch.length, torch.tensor(0.3))

    outgoing = patch.track(incoming)

    torch.testing.assert_close(outgoing.x, torch.full_like(outgoing.x, -0.1))
    torch.testing.assert_close(outgoing.px, incoming.px)
    torch.testing.assert_close(outgoing.y, torch.full_like(outgoing.y, -0.2))
    torch.testing.assert_close(outgoing.py, incoming.py)
    torch.testing.assert_close(outgoing.tau, incoming.tau)
    torch.testing.assert_close(outgoing.p, incoming.p)
    torch.testing.assert_close(outgoing.s, torch.tensor(0.3))


def test_transform_particles_with_angles():
    """
    Test tracking through a `Patch` with non-zero pitch angles and tilt against expected
    Bmad reference coordinates.
    """
    # Beam with 10 particles placed at the origin
    incoming = cheetah.ParticleBeam(
        particles=torch.zeros(10, 7), energy=torch.tensor(1.0e9)
    )

    patch = cheetah.Patch(
        position_offset=torch.tensor([0.1, 0.2, 0.0]),
        pitch=torch.tensor((0.5, -0.5)),
        tilt=torch.tensor(0.75),
    )

    outgoing = patch.track(incoming)

    expected_offsets_from_bmad = torch.tensor(
        [-1.950462e-01, -6.400071e-02, -1.297652e-01, 6.346425e-01, 1.605987e-02, 0.0]
    )
    expected_offset_particles = incoming.particles[..., :6] + expected_offsets_from_bmad

    torch.testing.assert_close(
        outgoing.particles[..., :6], expected_offset_particles, atol=1e-6, rtol=1e-5
    )


def test_transform_parameters():
    """
    Test that tracking a `ParameterBeam` through a `Patch` matches the mean and
    covariance of a tracked `ParticleBeam`.
    """
    incoming_particle_beam = cheetah.ParticleBeam(
        particles=torch.randn(100_000, 7) * 1.0e-4, energy=torch.tensor(1.0e9)
    )
    incoming_parameter_beam = incoming_particle_beam.as_parameter_beam()

    patch = cheetah.Patch(
        position_offset=torch.tensor([0.1, 0.2, 0.3]),
        pitch=torch.tensor([0.5, -0.5]),
        tilt=torch.tensor(0.75),
    )

    outgoing_parameter_beam = patch.track(incoming_parameter_beam)
    expected_outgoing_parameter_beam = patch.track(
        incoming_particle_beam
    ).as_parameter_beam()

    torch.testing.assert_close(
        outgoing_parameter_beam.mu,
        expected_outgoing_parameter_beam.mu,
        atol=1.0e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        outgoing_parameter_beam.cov,
        expected_outgoing_parameter_beam.cov,
        atol=1.0e-7,
        rtol=5.0e-2,
    )


def test_energy_offset():
    """Test that energy offset scales momentum deviations properly."""
    incoming = cheetah.ParticleBeam(
        particles=torch.zeros(10, 7), energy=torch.tensor(1.0e9)
    )
    incoming.particles[..., 5] = 0.05  # Give initial delta of 0.05

    patch = cheetah.Patch(energy_offset=torch.tensor(2.0e8))

    outgoing = patch.track(incoming)

    momentum_ratio = (incoming.energy * incoming.relativistic_beta) / (
        outgoing.energy * outgoing.relativistic_beta
    )
    expected_delta = 0.05 * momentum_ratio + (momentum_ratio - 1.0)

    torch.testing.assert_close(outgoing.energy, torch.tensor(1.2e9))
    torch.testing.assert_close(
        outgoing.particles[..., 5], expected_delta.expand_as(outgoing.particles[..., 5])
    )


def test_jacobian_and_transfer_map():
    """Test that the autograd Jacobian matches the analytical transfer map."""
    patch_with_angles = cheetah.Patch(
        position_offset=torch.tensor([0.1, 0.2, 0.3]),
        time_offset=torch.tensor(0.0),
        pitch=torch.tensor((0.5, -0.5)),
        tilt=torch.tensor(0.75),
        energy_offset=torch.tensor(0.0),
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

    transfer_map = patch_with_angles.first_order_transfer_map(
        energy, cheetah.Species("electron")
    )
    torch.testing.assert_close(transfer_map[:6, :6], J[:6, :6], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(J[:6, :6], gt_J, atol=5e-4, rtol=1e-4)


def test_vectorized():
    """Test tracking with a vectorized `Patch` element."""
    vectorized_patch = cheetah.Patch(
        position_offset=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        pitch=torch.tensor([[0.1, -0.1], [0.2, -0.2]]),
        tilt=torch.tensor([0.1, 0.2]),
        time_offset=torch.tensor([0.0, 1.0e-9]),
        energy_offset=torch.tensor([0.0, 1.0e6]),
    )
    incoming = cheetah.ParticleBeam(
        particles=torch.zeros(20, 7), energy=torch.tensor(1.0e9)
    )
    outgoing = vectorized_patch.track(incoming)
    assert outgoing.particles.shape == (2, 20, 7)
    assert outgoing.s.shape == (2,)

    transfer_map = vectorized_patch.first_order_transfer_map(
        incoming.energy, incoming.species
    )
    assert transfer_map.shape == (2, 7, 7)

    outgoing_param = vectorized_patch.track(incoming.as_parameter_beam())
    assert outgoing_param.mu.shape == (2, 7)
    assert outgoing_param.cov.shape == (2, 7, 7)
