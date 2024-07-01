import torch

from cheetah import Drift, ParameterBeam, ParticleBeam, Quadrupole, Segment


def test_quadrupole_off():
    """
    Test that a quadrupole with k1=0 behaves still like a drift.
    """
    quadrupole = Quadrupole(length=torch.tensor([1.0]), k1=torch.tensor([0.0]))
    drift = Drift(length=torch.tensor([1.0]))
    incoming_beam = ParameterBeam.from_parameters(
        sigma_xp=torch.tensor([2e-7]), sigma_yp=torch.tensor([2e-7])
    )
    outbeam_quad = quadrupole(incoming_beam)
    outbeam_drift = drift(incoming_beam)

    quadrupole.k1 = torch.tensor([1.0], device=quadrupole.k1.device)
    outbeam_quad_on = quadrupole(incoming_beam)

    assert torch.allclose(outbeam_quad.sigma_x, outbeam_drift.sigma_x)
    assert not torch.allclose(outbeam_quad_on.sigma_x, outbeam_drift.sigma_x)


def test_quadrupole_with_misalignments_batched():
    """
    Test that a quadrupole with misalignments behaves as expected.
    """

    quad_with_misalignment = Quadrupole(
        length=torch.tensor([1.0]),
        k1=torch.tensor([1.0]),
        misalignment=torch.tensor([[0.1, 0.1]]),
    )

    quad_without_misalignment = Quadrupole(
        length=torch.tensor([1.0]), k1=torch.tensor([1.0])
    )
    incoming_beam = ParameterBeam.from_parameters(
        sigma_xp=torch.tensor([2e-7]), sigma_yp=torch.tensor([2e-7])
    )
    outbeam_quad_with_misalignment = quad_with_misalignment(incoming_beam)
    outbeam_quad_without_misalignment = quad_without_misalignment(incoming_beam)

    assert not torch.allclose(
        outbeam_quad_with_misalignment.mu_x,
        outbeam_quad_without_misalignment.mu_x,
    )


def test_quadrupole_with_misalignments_multiple_batch_dimension():
    """
    Test that a quadrupole with misalignments with multiple batch dimension.
    """
    batch_shape = torch.Size([4, 3])
    quad_with_misalignment = Quadrupole(
        length=torch.tensor([1.0]),
        k1=torch.tensor([1.0]),
        misalignment=torch.tensor([[0.1, 0.1]]),
    ).broadcast(batch_shape)

    quad_without_misalignment = Quadrupole(
        length=torch.tensor([1.0]), k1=torch.tensor([1.0])
    ).broadcast(batch_shape)
    incoming_beam = ParameterBeam.from_parameters(
        sigma_xp=torch.tensor([2e-7]), sigma_yp=torch.tensor([2e-7])
    ).broadcast(batch_shape)
    outbeam_quad_with_misalignment = quad_with_misalignment(incoming_beam)
    outbeam_quad_without_misalignment = quad_without_misalignment(incoming_beam)

    # Check that the misalignment has an effect
    assert not torch.allclose(
        outbeam_quad_with_misalignment.mu_x,
        outbeam_quad_without_misalignment.mu_x,
    )

    # Check that the output shape is correct
    assert outbeam_quad_with_misalignment.mu_x.shape == batch_shape


def test_tilted_quadrupole_batch():
    batch_shape = torch.Size([3])
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1000000),
        energy=torch.tensor([1e9]),
        mu_x=torch.tensor([1e-5]),
    ).broadcast(batch_shape)
    segment = Segment(
        [
            Quadrupole(
                length=torch.tensor([0.5, 0.5, 0.5]),
                k1=torch.tensor([1.0, 1.0, 1.0]),
                tilt=torch.tensor([torch.pi / 4, torch.pi / 2, torch.pi * 5 / 4]),
            ),
            Drift(length=torch.tensor([0.5])).broadcast(batch_shape),
        ]
    )
    outgoing = segment(incoming)

    # Check pi/4 and 5/4*pi rotations is the same for quadrupole
    assert torch.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check pi/2 rotation is different
    assert not torch.allclose(outgoing.particles[0], outgoing.particles[1])


def test_tilted_quadrupole_multiple_batch_dimension():
    batch_shape = torch.Size([3, 2])
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(10000),
        energy=torch.tensor([1e9]),
        mu_x=torch.tensor([1e-5]),
    ).broadcast(batch_shape)
    segment = Segment(
        [
            Quadrupole(
                length=torch.tensor([0.5]),
                k1=torch.tensor([1.0]),
                tilt=torch.tensor([torch.pi / 4]),
            ),
            Drift(length=torch.tensor([0.5])),
        ]
    ).broadcast(batch_shape)
    outgoing = segment(incoming)

    assert torch.allclose(outgoing.particles[0, 0], outgoing.particles[0, 1])


def test_quadrupole_bmadx_tracking():
    incoming_beam = torch.load("tests/resources/quadrupole_bmadx/incoming_beam.pt")
    # quadrupole params
    l_quad = torch.tensor([1.0])
    k1_quad = torch.tensor([10.0])
    tilt = torch.tensor([0.5])
    offsets = torch.tensor([0.01, -0.02])
    # bmadx tracking method
    cheetah_quad_bmadx = Quadrupole(
        l_quad,
        k1_quad,
        misalignment=offsets,
        tilt=tilt,
        num_steps=10,
        tracking_method="bmadx",
        dtype=torch.double,
    )
    cheetah_segment_bmadx = Segment(elements=[cheetah_quad_bmadx])
    cheetah_bmadx_outgoing_beam = cheetah_segment_bmadx.track(incoming_beam)
    # load result from bmadx library
    bmadx_out_with_cheetah_coords = torch.load(
        "tests/resources/quadrupole_bmadx/bmadx_out_with_cheetah_coords.pt"
    )

    assert torch.allclose(
        bmadx_out_with_cheetah_coords,
        cheetah_bmadx_outgoing_beam.particles,
        atol=1e-7,
        rtol=1e-7,
    )
