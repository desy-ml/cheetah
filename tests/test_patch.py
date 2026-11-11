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
    incoming_particle_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000,
        energy=torch.tensor(1.0e9),
        sigma_x=torch.tensor(1.0e-4),
        sigma_px=torch.tensor(1.0e-4),
        sigma_y=torch.tensor(1.0e-4),
        sigma_py=torch.tensor(1.0e-4),
        sigma_tau=torch.tensor(1.0e-4),
        sigma_p=torch.tensor(1.0e-4),
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
        outgoing_parameter_beam.cov[:6, :6],
        expected_outgoing_parameter_beam.cov[:6, :6],
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


def test_vectorized():
    """
    Test tracking that tracking through a vectorised `Patch` element results in the
    correct shapes on the outputs.
    """
    incoming_particle_beam = cheetah.ParticleBeam(
        particles=torch.zeros(20, 7), energy=torch.tensor(1.0e9)
    )
    incoming_parameter_beam = incoming_particle_beam.as_parameter_beam()

    patch = cheetah.Patch(
        position_offset=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        pitch=torch.tensor([[0.1, -0.1], [0.2, -0.2]]),
        tilt=torch.tensor([0.1, 0.2]),
        time_offset=torch.tensor([0.0, 1.0e-9]),
        energy_offset=torch.tensor([0.0, 1.0e6]),
    )

    outgoing_particle_beam = patch.track(incoming_particle_beam)
    outgoing_parameter_beam = patch.track(incoming_parameter_beam)

    assert outgoing_particle_beam.particles.shape == (2, 20, 7)
    assert outgoing_particle_beam.s.shape == (2,)

    assert outgoing_parameter_beam.mu.shape == (2, 7)
    assert outgoing_parameter_beam.cov.shape == (2, 7, 7)
