import pytest
import torch

from cheetah import Drift, ParameterBeam, ParticleBeam, Quadrupole, Segment


def test_quadrupole_off():
    """
    Test that a quadrupole with k1=0 behaves still like a drift.
    """
    quadrupole = Quadrupole(length=torch.tensor([1.0]), k1=torch.tensor([0.0]))
    drift = Drift(length=torch.tensor([1.0]))
    incoming_beam = ParameterBeam.from_parameters(
        sigma_px=torch.tensor([2e-7]), sigma_py=torch.tensor([2e-7])
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
        sigma_px=torch.tensor([2e-7]), sigma_py=torch.tensor([2e-7])
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
        sigma_px=torch.tensor([2e-7]), sigma_py=torch.tensor([2e-7])
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
    """
    Test that a quadrupole with a tilt behaves as expected in vectorised mode.
    """
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
    """
    Test that a quadrupole with a tilt behaves as expected in vectorised mode with
    multiple vectorisation dimensions.
    """
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
    """
    Test that the results of tracking through a quadrupole with the `"bmadx"` tracking
    method match the results from Bmad-X.
    """
    incoming = torch.load("tests/resources/bmadx/incoming_beam.pt")
    quadrupole = Quadrupole(
        length=torch.tensor([1.0]),
        k1=torch.tensor([10.0]),
        misalignment=torch.tensor([[0.01, -0.02]]),
        tilt=torch.tensor([0.5]),
        num_steps=10,
        tracking_method="bmadx",
        dtype=torch.double,
    )
    segment = Segment(elements=[quadrupole])

    # Run tracking
    outgoing = segment.track(incoming)

    # Load reference result computed with Bmad-X
    bmadx_out_with_cheetah_coords = torch.load(
        "tests/resources/bmadx/quadrupole_bmadx_out_with_cheetah_coords.pt"
    )

    assert torch.allclose(
        outgoing.particles, bmadx_out_with_cheetah_coords, atol=1e-7, rtol=1e-7
    )


@pytest.mark.parametrize("tracking_method", ["cheetah", "bmadx"])
def test_tracking_method_vectorization(tracking_method):
    """
    Test that the quadruople vectorisation works correctly with both tracking methods.
    Only checks the shapes, not the physical correctness of the results.
    """
    quadrupole = Quadrupole(
        length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.4, 0.45]]),
        k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.4, 4.4]]),
        misalignment=torch.zeros((3, 2, 2)),
        tilt=torch.zeros((3, 2)),
        tracking_method=tracking_method,
    )
    incoming = ParticleBeam.from_parameters(
        sigma_x=torch.tensor([[1e-5, 2e-5], [2e-5, 3e-5], [3e-5, 4e-5]])
    )

    outgoing = quadrupole.track(incoming)

    assert outgoing.mu_x.shape == (3, 2)
    assert outgoing.mu_px.shape == (3, 2)
    assert outgoing.mu_y.shape == (3, 2)
    assert outgoing.mu_py.shape == (3, 2)
    assert outgoing.sigma_x.shape == (3, 2)
    assert outgoing.sigma_px.shape == (3, 2)
    assert outgoing.sigma_y.shape == (3, 2)
    assert outgoing.sigma_py.shape == (3, 2)
    assert outgoing.sigma_tau.shape == (3, 2)
    assert outgoing.sigma_p.shape == (3, 2)
    assert outgoing.energy.shape == (3, 2)
    assert outgoing.total_charge.shape == (3, 2)
