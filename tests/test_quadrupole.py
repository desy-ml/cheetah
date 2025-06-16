import pytest
import torch

import cheetah


def test_quadrupole_off():
    """
    Test that a quadrupole with k1=0 behaves still like a drift.
    """
    quadrupole = cheetah.Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(0.0))
    drift = cheetah.Drift(length=torch.tensor(1.0))
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
    )
    outbeam_quad = quadrupole(incoming_beam)
    outbeam_drift = drift(incoming_beam)

    quadrupole.k1 = torch.tensor(1.0, device=quadrupole.k1.device)
    outbeam_quad_on = quadrupole(incoming_beam)

    assert torch.allclose(outbeam_quad.sigma_x, outbeam_drift.sigma_x)
    assert not torch.allclose(outbeam_quad_on.sigma_x, outbeam_drift.sigma_x)


def test_quadrupole_with_misalignments_vectorized():
    """
    Test that a quadrupole with misalignments behaves as expected.
    """
    quad_with_misalignment = cheetah.Quadrupole(
        length=torch.tensor(1.0),
        k1=torch.tensor(1.0),
        misalignment=torch.tensor([0.1, 0.1]).unsqueeze(0),
    )

    quad_without_misalignment = cheetah.Quadrupole(
        length=torch.tensor(1.0), k1=torch.tensor(1.0)
    )
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
    )
    outbeam_quad_with_misalignment = quad_with_misalignment(incoming_beam)
    outbeam_quad_without_misalignment = quad_without_misalignment(incoming_beam)

    assert not torch.allclose(
        outbeam_quad_with_misalignment.mu_x,
        outbeam_quad_without_misalignment.mu_x,
    )


def test_quadrupole_with_misalignments_multiple_vector_dimensions():
    """
    Test that a quadrupole with misalignments that have multiple vector dimensions does
    not raise an error and behaves as expected.
    """
    quad_with_misalignment = cheetah.Quadrupole(
        length=torch.tensor(1.0),
        k1=torch.tensor(1.0),
        misalignment=torch.randn((4, 3, 2)) * 5e-4,
    )
    quad_without_misalignment = cheetah.Quadrupole(
        length=torch.tensor(1.0), k1=torch.tensor(1.0)
    )

    incoming = cheetah.ParameterBeam.from_parameters(
        sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
    )

    outgoing_with_misalignment = quad_with_misalignment(incoming)
    outgoing_without_misalignment = quad_without_misalignment(incoming)

    # Check that the misalignment has an effect
    assert not torch.allclose(
        outgoing_with_misalignment.mu_x, outgoing_without_misalignment.mu_x
    )

    # Check that the output shape is correct
    assert outgoing_with_misalignment.mu_x.shape == (4, 3)


def test_tilted_quadrupole_vectorized():
    """
    Test that a quadrupole with a tilt behaves as expected in vectorised mode.
    """
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=1_000_000, energy=torch.tensor(1e9), mu_x=torch.tensor(1e-5)
    )
    segment = cheetah.Segment(
        [
            cheetah.Quadrupole(
                length=torch.tensor(0.5),
                k1=torch.tensor(1.0),
                tilt=torch.tensor([torch.pi / 4, torch.pi / 2, torch.pi * 5 / 4]),
            ),
            cheetah.Drift(length=torch.tensor(0.5)),
        ]
    )
    outgoing = segment(incoming)

    # Check that pi/4 and 5/4*pi rotations is the same for quadrupole
    assert torch.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check that pi/2 rotation is different
    assert not torch.allclose(outgoing.particles[0], outgoing.particles[1])


def test_tilted_quadrupole_multiple_vector_dimensions():
    """
    Test that a quadrupole with tilts that have multiple vectorisation dimensions does
    not raise an error and behaves as expected.
    """
    segment = cheetah.Segment(
        [
            cheetah.Quadrupole(
                length=torch.tensor(0.5),
                k1=torch.tensor(1.0),
                tilt=torch.tensor(
                    [
                        [torch.pi / 4, torch.pi / 2, torch.pi * 5 / 4],
                        [torch.pi * 5 / 4, torch.pi / 2, torch.pi / 4],
                    ]
                ),
            ),
            cheetah.Drift(length=torch.tensor(0.5)),
        ]
    )

    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000, energy=torch.tensor(1e9), mu_x=torch.tensor(1e-5)
    )

    outgoing = segment(incoming)

    # Test that shape is correct
    assert outgoing.particles.shape == (2, 3, 10_000, 7)

    # Check that same tilts give same results
    assert torch.allclose(outgoing.particles[0, 0], outgoing.particles[1, 2])
    assert torch.allclose(outgoing.particles[0, 1], outgoing.particles[1, 1])
    assert torch.allclose(outgoing.particles[0, 2], outgoing.particles[1, 0])


def test_quadrupole_length_multiple_vector_dimensions():
    """
    Test that a quadrupole with lengths that have multiple vectorisation dimensions does
    not raise an error and behaves as expected.
    """
    lengths = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.4, 0.7]])
    segment = cheetah.Segment(
        [
            cheetah.Quadrupole(length=lengths, k1=torch.tensor(4.2)),
            cheetah.Drift(length=lengths * 2),
        ]
    )

    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000, energy=torch.tensor(1e9), mu_x=torch.tensor(1e-5)
    )

    outgoing = segment(incoming)

    assert outgoing.particles.shape == (2, 3, 10_000, 7)
    assert torch.allclose(outgoing.particles[0, 2], outgoing.particles[1, 1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quadrupole_bmadx_tracking(dtype):
    """
    Test that the results of tracking through a quadrupole with the `"bmadx"` tracking
    method match the results from Bmad-X.
    """
    incoming = torch.load("tests/resources/bmadx/incoming.pt", weights_only=False).to(
        dtype
    )
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor(1.0),
        k1=torch.tensor(10.0),
        misalignment=torch.tensor([0.01, -0.02], dtype=dtype),
        tilt=torch.tensor(0.5),
        num_steps=10,
        tracking_method="bmadx",
        dtype=dtype,
    )
    segment = cheetah.Segment(elements=[quadrupole])

    # Run tracking
    outgoing = segment.track(incoming)

    # Load reference result computed with Bmad-X
    outgoing_bmadx = torch.load(
        "tests/resources/bmadx/outgoing_quadrupole.pt", weights_only=False
    )

    assert torch.allclose(
        outgoing.particles,
        outgoing_bmadx.to(dtype),
        atol=1e-14 if dtype == torch.float64 else 1e-5,
        rtol=1e-14 if dtype == torch.float64 else 1e-6,
    )


@pytest.mark.parametrize("tracking_method", ["cheetah", "bmadx"])
def test_tracking_method_vectorization(tracking_method):
    """
    Test that the quadruople vectorisation works correctly with both tracking methods.
    Only checks the shapes, not the physical correctness of the results.
    """
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.4, 0.45]]),
        k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.4, 4.4]]),
        misalignment=torch.zeros(2),
        tilt=torch.tensor(0.0),
        tracking_method=tracking_method,
    )
    incoming = cheetah.ParticleBeam.from_parameters(
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
    assert outgoing.energy.shape == torch.Size([])
    assert outgoing.total_charge.shape == torch.Size([])


@pytest.mark.parametrize("tracking_method", ["cheetah", "bmadx"])
def test_quadrupole_clone_tracking_method(tracking_method):
    """
    Test that the tracking_method is preserved when cloning a Quadrupole.
    """
    # Create a quadrupole with bmadx tracking method
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor(1.0), k1=torch.tensor(1.0), tracking_method=tracking_method
    )

    # Clone the quadrupole
    cloned = quadrupole.clone()

    # Verify that tracking_method is preserved
    assert cloned.tracking_method == quadrupole.tracking_method
    assert cloned.tracking_method == tracking_method


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tilted_quad_transfer_matrix_precision(dtype):
    """
    Test that the transfer matrix for a tilted quadrupole element with k1=0 matches the
    transfer matrices of a normal quadrupole and a drift element to the precision of the
    used dtype.
    """
    # Create three elements that should have the same transfer matrix
    length = torch.tensor(0.5, dtype=dtype)
    k1 = torch.tensor(0.0, dtype=dtype)
    tilt = torch.tensor(torch.pi / 4, dtype=dtype)

    quad = cheetah.Quadrupole(length=length, k1=k1)
    skew_quad = cheetah.Quadrupole(length=length, k1=k1, tilt=tilt)
    drift = cheetah.Drift(length=length)

    # Compute the transfer matrices
    energy = torch.tensor(1e9, dtype=dtype)
    spiecies = cheetah.Species("electron")

    tm_quad = quad.transfer_map(energy, spiecies)
    tm_skew_quad = skew_quad.transfer_map(energy, spiecies)
    tm_drift = drift.transfer_map(energy, spiecies)

    # Check that the transfer matrices are equal of the dtype
    # NOTE: The `==` is used here over `torch.allclose` on purpose
    assert (tm_quad == tm_drift).all()
    assert (tm_skew_quad == tm_drift).all()
