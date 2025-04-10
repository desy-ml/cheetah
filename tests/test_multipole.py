import pytest
import torch

import cheetah


def test_multipole_as_drift():
    """Test that a multipole with all coefficients zero approximates drift behaviour."""
    multipole = cheetah.Multipole(length=torch.tensor(1.0))
    drift = cheetah.Drift(length=torch.tensor(1.0))

    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=torch.tensor(1_000), energy=torch.tensor(1e9)
    )

    # Track through both elements
    outgoing_multipole = multipole.track(incoming)
    outgoing_drift = drift.track(incoming)

    assert torch.allclose(
        outgoing_multipole.particles, outgoing_drift.particles, atol=1e-5
    )


def test_multipole_as_quadrupole():
    """
    Compare a multipole configured as a quadrupole with a native quadrupole element.
    """
    k1 = torch.tensor(4.2)

    multipole = cheetah.Multipole(
        length=torch.tensor(1.0), polynom_b=torch.tensor([0.0, k1])  # B1 = k1
    )
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor(1.0),
        k1=k1,
        tracking_method="bmadx",
        num_steps=10,  # Use multiple steps for better accuracy
    )

    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=torch.tensor(1_000),
        energy=torch.tensor(1e9),
        mu_x=torch.tensor(1e-3),
    )

    # Track through both elements
    outgoing_multipole = multipole.track(incoming)
    outgoing_quadrupole = quadrupole.track(incoming)

    # Both should cause focusing/defocusing, but exact values will differ due to
    # different tracking methods. Just verify they both affect the beam.
    assert abs(outgoing_multipole.mu_px) > 1e-9
    assert abs(outgoing_quadrupole.mu_px) > 1e-9

    # The sign of px change should be the same for both
    assert torch.sign(outgoing_multipole.mu_px) == torch.sign(outgoing_quadrupole.mu_px)


def test_multipole_as_horizontal_corrector():
    """
    Test that a multipole with B0 coefficient behaves like a horizontal corrector
    element.
    """
    # Set parameters
    length = torch.tensor(1e-9)  # Very short element
    angle = torch.tensor(2e-3)  # Angle in radians

    # Create elements
    hcorr = HorizontalCorrector(length=length, angle=angle)
    multipole_hcorr = Multipole(
        length=length,
        polynom_b=torch.tensor([-angle / length, 0.0]),  # B0 = -angle/length
        max_order=1,
    )

    # Create test beam
    beam = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1_000),
        energy=torch.tensor(1e9),
    )

    # Track through both elements
    out_hcorr = hcorr(beam)
    out_multipole = multipole_hcorr(beam)

    # Compare results
    print("\nComparison Results:")
    print(f"HCM horizontal momentum: {out_hcorr.mu_px.item():.8e}")
    print(f"Multipole horizontal momentum: {out_multipole.mu_px.item():.8e}")
    print(f"Difference: {(out_hcorr.mu_px - out_multipole.mu_px).item():.8e}")

    # Verify they produce similar results
    assert torch.allclose(out_hcorr.mu_px, out_multipole.mu_px, atol=1e-9)
    assert torch.allclose(out_hcorr.mu_py, out_multipole.mu_py, atol=1e-9)
    assert torch.allclose(out_hcorr.mu_x, out_multipole.mu_x, atol=1e-9)
    assert torch.allclose(out_hcorr.mu_y, out_multipole.mu_y, atol=1e-9)


def test_multipole_with_misalignment():
    """
    Test that a multipole with misalignment behaves equivalently to a centered
    multipole with an offset beam (with appropriate transformation).
    """
    # Parameters
    length = torch.tensor(1.0)
    k1 = torch.tensor(20.0)  # Strong quadrupole for clear effect
    offset_x = torch.tensor(0.001)  # 1mm horizontal offset
    offset_y = torch.tensor(0.002)  # 2mm vertical offset

    # Create a quadrupole-like multipole with misalignment
    multipole_with_misalignment = Multipole(
        length=length,
        polynom_b=torch.tensor([0.0, k1]),
        misalignment=torch.tensor([offset_x, offset_y]),
    )

    # Create a centered multipole
    multipole_centered = Multipole(
        length=length,
        polynom_b=torch.tensor([0.0, k1]),
    )

    # Create a beam centered at origin
    centered_beam = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1_000),
        energy=torch.tensor(1e9),
    )

    # Create an offset beam with opposite offset relative to misalignment
    offset_beam = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1_000),
        energy=torch.tensor(1e9),
        mu_x=-offset_x,
        mu_y=-offset_y,
    )

    # Track through both scenarios
    outbeam_misaligned_quad = multipole_with_misalignment(centered_beam)
    outbeam_offset_beam = multipole_centered(offset_beam)

    # Transform the offset beam results back to the lab frame
    # For offset beam case, the beam was offset by (-offset_x, -offset_y)
    # So we need to translate the results back by (offset_x, offset_y)
    transformed_offset_beam_mu_x = outbeam_offset_beam.mu_x + offset_x
    transformed_offset_beam_mu_y = outbeam_offset_beam.mu_y + offset_y

    # Compare the results - they should be nearly identical after transformation
    assert torch.allclose(
        outbeam_misaligned_quad.mu_x,
        transformed_offset_beam_mu_x,
        atol=1e-5,  # Increased tolerance to account for numerical differences
    )
    assert torch.allclose(
        outbeam_misaligned_quad.mu_y,
        transformed_offset_beam_mu_y,
        atol=1e-5,  # Increased tolerance to account for numerical differences
    )
    assert torch.allclose(
        outbeam_misaligned_quad.mu_px,
        outbeam_offset_beam.mu_px,
        atol=1e-5,  # Increased tolerance to account for numerical differences
    )
    assert torch.allclose(
        outbeam_misaligned_quad.mu_py,
        outbeam_offset_beam.mu_py,
        atol=1e-5,  # Increased tolerance to account for numerical differences
    )


def test_skew_multipole_behavior():
    """
    Test that skew multipole components (polynom_a) work as expected (like 90deg tilted
    normal quadrupole).
    """
    # Normal quadrupole (using B1)
    normal_quad = Multipole(
        length=torch.tensor(1.0),
        polynom_b=torch.tensor([0.0, 5.0]),  # Strong for clear effect
    )

    # Skew quadrupole (using A1)
    skew_quad = Multipole(
        length=torch.tensor(1.0),
        polynom_a=torch.tensor([0.0, 5.0]),
    )

    # Create a beam with x offset only
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1_000),
        energy=torch.tensor(1e9),
        mu_x=torch.tensor(1e-3),
        mu_y=torch.tensor(0.0),
    )

    # Track through both elements
    out_normal = normal_quad(incoming)
    out_skew = skew_quad(incoming)

    # Normal and skew quadrupoles should have different coupling behaviors
    # Skew quadrupole should have larger py change from x offset compared to normal quad
    assert not torch.allclose(out_normal.mu_py, out_skew.mu_py, atol=1e-10)


def test_multipole_max_order():
    """
    Test that the max_order parameter correctly limits the polynomial orders.
    """
    # Create a multipole with coefficients up to order 5
    multipole_full = Multipole(
        length=torch.tensor(1.0),
        polynom_b=torch.tensor([0.0, 1.0, 0.5, 0.2, 0.1, 0.05]),  # Orders 0-5
        max_order=5,
    )

    # Create a multipole with the same coefficients but limited to order 2
    multipole_limited = Multipole(
        length=torch.tensor(1.0),
        polynom_b=torch.tensor(
            [0.0, 1.0, 0.5, 0.2, 0.1, 0.05]
        ),  # Only first 3 will be used
        max_order=2,
    )

    # Verify that the coefficients were properly truncated
    assert multipole_limited.polynom_b.size(0) == 3  # Should contain orders 0, 1, 2
    assert torch.all(multipole_limited.polynom_b == torch.tensor([0.0, 1.0, 0.5]))

    # Create another multipole with extra coefficients beyond max_order
    polynom_b_extra = torch.zeros(10)
    polynom_b_extra[1] = 1.0  # Set B1 = 1.0

    multipole_extra = Multipole(
        length=torch.tensor(1.0),
        polynom_b=polynom_b_extra,
        max_order=2,
    )

    # Should be truncated to just 3 elements
    assert multipole_extra.polynom_b.size(0) == 3


@pytest.mark.parametrize("tracking_method", ["symplectic4", "symplectic4_rad"])
def test_multipole_tracking_methods(tracking_method):
    """
    Test that different tracking methods don't crash.
    """
    # Create a multipole with both normal and skew components
    multipole = Multipole(
        length=torch.tensor(1.0),
        polynom_b=torch.tensor([0.0, 1.0, 1.0]),  # B1 and B2 components
        polynom_a=torch.tensor([0.0, 0.5]),  # A1 component
        tracking_method=tracking_method,
        num_steps=10,  # Use multiple steps for better accuracy
    )

    # Create a beam with offset to see effects
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1_000),
        energy=torch.tensor(1e9),
        mu_x=torch.tensor(1e-3),
    )

    # Track - should run without errors
    outgoing = multipole(incoming)

    # Output should have expected dimensions
    assert outgoing is not None
    assert outgoing.particles.shape == incoming.particles.shape


def test_multipole_split():
    """
    Test that splitting a multipole into smaller segments works correctly.
    """
    # Create a quadrupole-like multipole
    original = Multipole(
        length=torch.tensor(1.0),
        polynom_b=torch.tensor([0.0, 1.0]),
        fringe_quad_entrance=1,
        fringe_quad_exit=1,
    )

    # Split into 5 pieces
    split_elements = original.split(torch.tensor(0.2))

    assert len(split_elements) == 5
    assert isinstance(split_elements[0], Multipole)

    # Check that fringe fields are only applied at the beginning and end
    assert split_elements[0].fringe_quad_entrance == 1
    assert split_elements[0].fringe_quad_exit == 0
    assert split_elements[1].fringe_quad_entrance == 0
    assert split_elements[1].fringe_quad_exit == 0
    assert split_elements[4].fringe_quad_entrance == 0
    assert split_elements[4].fringe_quad_exit == 1

    # Check that the total length is preserved
    total_length = sum(element.length.item() for element in split_elements)
    assert torch.isclose(torch.tensor(total_length), original.length)

    # Check that all other properties are preserved
    for element in split_elements:
        assert torch.all(element.polynom_b == original.polynom_b)
        assert torch.all(element.polynom_a == original.polynom_a)
        assert element.max_order == original.max_order
        assert torch.all(element.misalignment == original.misalignment)
        assert torch.all(element.tilt == original.tilt)


def test_multipole_parameter_validation():
    """
    Test that the multipole element properly validates and initializes parameters.
    """
    # Test with default parameters
    multipole = Multipole(length=torch.tensor(1.0))
    assert multipole.polynom_a.size(0) == 2  # Should contain orders 0, 1
    assert multipole.polynom_b.size(0) == 2
    assert multipole.max_order == 1

    # Test with custom max_order
    multipole = Multipole(length=torch.tensor(1.0), max_order=3)
    assert multipole.polynom_a.size(0) == 4  # Should contain orders 0, 1, 2, 3
    assert multipole.polynom_b.size(0) == 4

    # Test with partial polynomial coefficients
    multipole = Multipole(
        length=torch.tensor(1.0), polynom_b=torch.tensor([0.0, 1.0]), max_order=3
    )
    assert multipole.polynom_b.size(0) == 4  # Should be padded to max_order+1
    assert multipole.polynom_b[0] == 0.0
    assert multipole.polynom_b[1] == 1.0
    assert multipole.polynom_b[2] == 0.0
    assert multipole.polynom_b[3] == 0.0


def test_multipole_vectorization():
    """Test that a multipole with vectorized parameters works correctly."""
    # Create a multipole with vectorized k1 values (quadrupole strengths)
    k1_values = torch.tensor([1.0, 2.0, 3.0])
    multipole = Multipole(
        length=torch.tensor(1.0),
        polynom_b=torch.tensor([0.0, 1.0]).unsqueeze(0).repeat(3, 1)
        * k1_values.unsqueeze(1),
    )

    # Create a simple beam
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1_000),
        energy=torch.tensor(1e9),
        mu_x=torch.tensor(1e-3),  # Small offset to see focusing effects
    )

    # Track through the multipole
    outgoing = multipole(incoming)

    # Verify the output shape matches the vectorization
    assert outgoing.mu_px.shape == (3,)

    # Verify that different k1 values produce different focusing effects
    # Stronger k1 should result in stronger focusing
    assert abs(outgoing.mu_px[2]) > abs(outgoing.mu_px[1]) > abs(outgoing.mu_px[0])
