import pytest
import torch

from cheetah.utils.cloud_in_cell import (
    cloud_in_cell_charge_deposition,
    cloud_in_cell_charge_deposition_1d,
    cloud_in_cell_charge_deposition_2d,
    cloud_in_cell_charge_deposition_3d,
)


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_deposit_charge_cic_2d_basic(device):
    """Test basic functionality of 2D CIC charge deposition."""
    # Simple test case with known expected result
    x1 = torch.tensor([0.5, 1.5], device=device, dtype=torch.float32)
    x2 = torch.tensor([0.5, 1.5], device=device, dtype=torch.float32)

    bins1 = torch.tensor([0.0, 1.0, 2.0], device=device)
    bins2 = torch.tensor([0.0, 1.0, 2.0], device=device)

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2)

    # Expected result: each particle should be deposited equally among 4 grid points
    # Particle at (0.5, 0.5) -> corners at (0,0), (1,0), (0,1), (1,1) with weight 0.25
    # Particle at (1.5, 1.5) -> corners at (1,1), (2,1), (1,2), (2,2) with weight 0.25
    expected = torch.tensor(
        [
            [0.25, 0.25, 0.0],
            [0.25, 0.5, 0.25],  # (1,1) gets contribution from both particles
            [0.0, 0.25, 0.25],
        ],
        device=device,
    )

    assert result.shape == (3, 3)
    assert torch.allclose(result, expected, atol=1e-6)


def test_deposit_charge_cic_2d_with_weights():
    """Test CIC charge deposition with custom weights."""
    x1 = torch.tensor([0.5, 1.5])
    x2 = torch.tensor([0.5, 1.5])
    weights = torch.tensor([2.0, 3.0])

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2, weights)

    # Particle 1 (weight=2.0) at (0.5, 0.5) -> 0.5 to each corner
    # Particle 2 (weight=3.0) at (1.5, 1.5) -> 0.75 to each corner
    expected = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [0.5, 1.25, 0.75],  # (1,1) gets 0.5 + 0.75 = 1.25
            [0.0, 0.75, 0.75],
        ]
    )

    assert torch.allclose(result, expected, atol=1e-6)


def test_deposit_charge_cic_2d_batched():
    """Test batched CIC charge deposition."""
    batch_size = 2

    x1 = torch.tensor([[0.5, 1.5, 0.8], [1.2, 0.3, 1.7]])
    x2 = torch.tensor([[0.5, 1.5, 0.8], [1.2, 0.3, 1.7]])

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2)

    assert result.shape == (2, 3, 3)

    # Test that each batch is processed correctly
    for b in range(batch_size):
        batch_result = cloud_in_cell_charge_deposition_2d(x1[b], x2[b], bins1, bins2)
        assert torch.allclose(result[b], batch_result, atol=1e-6)


def test_deposit_charge_cic_2d_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test with particles exactly on grid points (excluding upper boundary)
    x1 = torch.tensor([0.0, 1.0])  # Removed 2.0 as it's outside [0.0, 2.0)
    x2 = torch.tensor([0.0, 1.0])  # Removed 2.0 as it's outside [0.0, 2.0)

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2)

    # Particles exactly on grid points should deposit all charge to that point
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # No particle at (2.0, 2.0) anymore
        ]
    )

    assert torch.allclose(result, expected, atol=1e-6)


def test_deposit_charge_cic_2d_outside_bounds():
    """Test behavior with particles outside grid bounds."""
    # Particles outside the grid should have their weights set to zero
    x1 = torch.tensor([-0.5, 1.0, 2.5])  # First and last outside grid bounds [0, 2]
    x2 = torch.tensor([0.5, 1.0, 1.5])
    weights = torch.tensor([1.0, 2.0, 3.0])

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2, weights)

    # Should not raise errors
    assert result.shape == (3, 3)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()

    # Only the middle particle (inside bounds) should contribute
    # Total deposited charge should be less than total input charge
    assert result.sum() == 2.0  # Only the middle particle with weight 2.0
    # Charge conservation violated due to outside particles
    assert result.sum() < weights.sum()


def test_deposit_charge_cic_2d_charge_zeroing():
    """Test that particles outside grid bounds have their charge set to zero."""
    # Create particles both inside and outside bounds
    x1 = torch.tensor([0.5, 1.5, -0.5, 2.5, 1.0])  # 3rd and 4th outside
    x2 = torch.tensor([0.5, 1.5, 0.5, 1.5, -0.5])  # 5th outside
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2, weights)

    # Only particles 1, 2 should contribute (weights 1.0, 2.0)
    expected_charge = 1.0 + 2.0  # 3.0
    assert torch.allclose(result.sum(), torch.tensor(expected_charge), atol=1e-6)

    # Verify specific outside particles contribute nothing
    x1_outside_only = torch.tensor([-1.0, 3.0])
    x2_outside_only = torch.tensor([0.5, 1.5])
    weights_outside = torch.tensor([10.0, 20.0])

    result_outside = cloud_in_cell_charge_deposition_2d(
        x1_outside_only, x2_outside_only, bins1, bins2, weights_outside
    )
    assert result_outside.sum() == 0.0  # All charge should be lost


def test_deposit_charge_cic_2d_conservation():
    """Test charge conservation."""
    x1 = torch.tensor([0.3, 1.7, 0.9, 1.1])
    x2 = torch.tensor([0.4, 1.6, 0.8, 1.2])
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5])

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2, weights)

    # Total deposited charge should equal sum of weights
    total_charge = result.sum()
    expected_total = weights.sum()

    assert torch.allclose(total_charge, expected_total, atol=1e-6)


def test_deposit_charge_cic_2d_dtype_consistency():
    """Test that output dtype matches input dtype."""
    for dtype in [torch.float32, torch.float64]:
        x1 = torch.tensor([0.5, 1.5], dtype=dtype)
        x2 = torch.tensor([0.5, 1.5], dtype=dtype)

        bins1 = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        bins2 = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)

        result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2)

        assert result.dtype == dtype


def test_deposit_charge_cic_2d_empty_input():
    """Test behavior with empty particle arrays."""
    x1 = torch.empty((0,))
    x2 = torch.empty((0,))

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2)

    expected = torch.zeros(3, 3)
    assert torch.allclose(result, expected)


def test_deposit_charge_cic_2d_non_uniform_spacing_error():
    """Test that non-uniform bin spacing raises an error."""
    x1 = torch.tensor([0.5, 1.5])
    x2 = torch.tensor([0.5, 1.5])

    # Non-uniform spacing should raise ValueError
    bins1_non_uniform = torch.tensor([0.0, 1.0, 2.5])  # Non-uniform: 1.0, 1.5
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="bins\\[0\\] must have uniform spacing"):
        cloud_in_cell_charge_deposition_2d(x1, x2, bins1_non_uniform, bins2)

    # Test non-uniform bins2
    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2_non_uniform = torch.tensor([0.0, 1.0, 2.3])  # Non-uniform: 1.0, 1.3

    with pytest.raises(ValueError, match="bins\\[1\\] must have uniform spacing"):
        cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2_non_uniform)


def test_deposit_charge_cic_2d_insufficient_bins():
    """Test that insufficient number of bins raises an error."""
    x1 = torch.tensor([0.5])
    x2 = torch.tensor([0.5])

    # Single bin should raise error
    bins1_single = torch.tensor([0.0])
    bins2 = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="bins\\[0\\] must have at least 2 elements"):
        cloud_in_cell_charge_deposition_2d(x1, x2, bins1_single, bins2)

    # Test single bin for bins2
    bins1 = torch.tensor([0.0, 1.0])
    bins2_single = torch.tensor([0.0])

    with pytest.raises(ValueError, match="bins\\[1\\] must have at least 2 elements"):
        cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2_single)


def test_deposit_charge_cic_2d_single_particle():
    """Test with a single particle at various positions."""
    bins1 = torch.tensor([0.0, 1.0, 2.0, 3.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0, 3.0])

    # Test particle at center of a cell
    x1 = torch.tensor([1.5])
    x2 = torch.tensor([1.5])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2)

    # Particle at cell center should deposit 0.25 to each of 4 surrounding corners
    expected = torch.zeros(4, 4)
    expected[1, 1] = 0.25  # (1, 1)
    expected[1, 2] = 0.25  # (1, 2)
    expected[2, 1] = 0.25  # (2, 1)
    expected[2, 2] = 0.25  # (2, 2)

    assert torch.allclose(result, expected, atol=1e-6)


def test_deposit_charge_cic_2d_multidimensional_batch():
    """Test with multi-dimensional batch shapes."""
    # Test with 2D batch dimensions
    x1 = torch.tensor([[[0.5, 1.5], [1.0, 1.2]]])  # shape (1, 2, 2)
    x2 = torch.tensor([[[0.5, 1.5], [1.0, 1.2]]])  # shape (1, 2, 2)

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2)

    assert result.shape == (1, 2, 3, 3)

    # Test that charge is conserved
    total_charge = result.sum(dim=(-1, -2))  # Sum over grid dimensions
    expected_total = torch.ones(1, 2) * 2  # 2 particles per batch element

    assert torch.allclose(total_charge, expected_total, atol=1e-6)


# =============== Tests for Generalized CIC Functions ===============


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_deposit_charge_cic_1d_basic(device):
    """Test basic 1D CIC charge deposition."""
    x = torch.tensor([0.5, 1.5], device=device, dtype=torch.float32)
    bins = torch.tensor([0.0, 1.0, 2.0], device=device)

    result = cloud_in_cell_charge_deposition_1d(x, bins)

    # Particle at 0.5 -> 0.5 to bin 0, 0.5 to bin 1
    # Particle at 1.5 -> 0.5 to bin 1, 0.5 to bin 2
    expected = torch.tensor([0.5, 1.0, 0.5], device=device)

    assert result.shape == (3,)
    assert torch.allclose(result, expected, atol=1e-6)


def test_deposit_charge_cic_1d_wrapper():
    """Test that 1D wrapper matches generalized function."""
    x = torch.tensor([0.3, 1.7, 0.9])
    bins = torch.tensor([0.0, 1.0, 2.0])
    weights = torch.tensor([1.0, 2.0, 0.5])

    result_wrapper = cloud_in_cell_charge_deposition_1d(x, bins, weights)
    result_general = cloud_in_cell_charge_deposition([x], [bins], weights)

    assert torch.allclose(result_wrapper, result_general)


def test_deposit_charge_cic_3d_basic():
    """Test basic 3D CIC charge deposition."""
    x1 = torch.tensor([0.5])
    x2 = torch.tensor([0.5])
    x3 = torch.tensor([0.5])

    bins1 = torch.tensor([0.0, 1.0])
    bins2 = torch.tensor([0.0, 1.0])
    bins3 = torch.tensor([0.0, 1.0])

    result = cloud_in_cell_charge_deposition_3d(x1, x2, x3, bins1, bins2, bins3)

    # Particle at (0.5, 0.5, 0.5) should contribute equally to all 8 corners
    expected = torch.ones(2, 2, 2) * 0.125

    assert result.shape == (2, 2, 2)
    assert torch.allclose(result, expected, atol=1e-6)


def test_deposit_charge_cic_3d_wrapper():
    """Test that 3D wrapper matches generalized function."""
    x1 = torch.tensor([0.3, 1.7])
    x2 = torch.tensor([0.4, 1.6])
    x3 = torch.tensor([0.6, 1.4])

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])
    bins3 = torch.tensor([0.0, 1.0, 2.0])

    weights = torch.tensor([1.5, 0.8])

    result_wrapper = cloud_in_cell_charge_deposition_3d(
        x1, x2, x3, bins1, bins2, bins3, weights
    )
    result_general = cloud_in_cell_charge_deposition(
        [x1, x2, x3], [bins1, bins2, bins3], weights
    )

    assert torch.allclose(result_wrapper, result_general)


def test_deposit_charge_cic_general_validation():
    """Test input validation for generalized function."""
    x = torch.tensor([0.5])
    bins = torch.tensor([0.0, 1.0])

    # Test empty positions
    with pytest.raises(
        ValueError, match="positions must contain at least one dimension"
    ):
        cloud_in_cell_charge_deposition([], [])

    # Test mismatched lengths
    with pytest.raises(
        ValueError, match="positions and bins must have the same length"
    ):
        cloud_in_cell_charge_deposition([x], [bins, bins])

    # Test too many dimensions
    with pytest.raises(
        ValueError, match="Only 1D, 2D, and 3D CIC deposition are supported"
    ):
        cloud_in_cell_charge_deposition([x, x, x, x], [bins, bins, bins, bins])

    # Test mismatched shapes
    x1 = torch.tensor([0.5])
    x2 = torch.tensor([0.5, 1.5])  # Different shape
    with pytest.raises(
        ValueError, match="All position tensors must have the same shape"
    ):
        cloud_in_cell_charge_deposition([x1, x2], [bins, bins])

    # Test mismatched devices
    if torch.cuda.is_available():
        x_cpu = torch.tensor([0.5])
        x_gpu = torch.tensor([0.5], device="cuda")
        with pytest.raises(ValueError, match="All tensors must be on the same device"):
            cloud_in_cell_charge_deposition([x_cpu, x_gpu], [bins, bins])

    # Test mismatched dtypes
    x_float = torch.tensor([0.5], dtype=torch.float32)
    x_double = torch.tensor([0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="All tensors must have the same dtype"):
        cloud_in_cell_charge_deposition([x_float, x_double], [bins, bins])


def test_deposit_charge_cic_charge_conservation():
    """Test charge conservation in all dimensions for particles within bounds."""
    weights = torch.tensor([1.0, 2.0, 0.5])

    # 1D case
    x = torch.tensor([0.3, 1.7, 0.9])
    bins = torch.tensor([0.0, 1.0, 2.0])
    result_1d = cloud_in_cell_charge_deposition([x], [bins], weights)
    assert torch.allclose(result_1d.sum(), weights.sum())

    # 2D case
    x1, x2 = torch.tensor([0.3, 1.7, 0.9]), torch.tensor([0.4, 1.6, 0.8])
    result_2d = cloud_in_cell_charge_deposition([x1, x2], [bins, bins], weights)
    assert torch.allclose(result_2d.sum(), weights.sum())

    # 3D case
    x3 = torch.tensor([0.6, 1.4, 1.2])
    result_3d = cloud_in_cell_charge_deposition(
        [x1, x2, x3], [bins, bins, bins], weights
    )
    assert torch.allclose(result_3d.sum(), weights.sum())


def test_deposit_charge_cic_charge_loss_outside_bounds():
    """Test charge loss for particles outside bounds in all dimensions."""

    # 1D case - particle outside bounds
    x_1d = torch.tensor([0.5, 2.5, 1.0])  # 2nd particle outside [0, 2]
    weights_1d = torch.tensor([1.0, 2.0, 3.0])
    bins_1d = torch.tensor([0.0, 1.0, 2.0])
    result_1d = cloud_in_cell_charge_deposition([x_1d], [bins_1d], weights_1d)
    expected_1d = 1.0 + 3.0  # Only 1st and 3rd particles contribute
    assert torch.allclose(result_1d.sum(), torch.tensor(expected_1d))

    # 2D case - particles outside in different dimensions
    x1_2d = torch.tensor([0.5, 2.5, 1.0])  # 2nd outside x1 bounds
    x2_2d = torch.tensor([3.0, 1.0, 1.0])  # 1st outside x2 bounds
    weights_2d = torch.tensor([1.0, 2.0, 3.0])
    result_2d = cloud_in_cell_charge_deposition(
        [x1_2d, x2_2d], [bins_1d, bins_1d], weights_2d
    )
    expected_2d = 3.0  # Only 3rd particle contributes (both coordinates inside)
    assert torch.allclose(result_2d.sum(), torch.tensor(expected_2d))

    # 3D case - particle outside multiple dimensions
    x3_2d = torch.tensor([0.5, 1.0, 1.0])
    result_3d = cloud_in_cell_charge_deposition(
        [x1_2d, x2_2d, x3_2d], [bins_1d, bins_1d, bins_1d], weights_2d
    )
    assert torch.allclose(result_3d.sum(), torch.tensor(expected_2d))  # Same as 2D case


def test_deposit_charge_cic_batched():
    """Test batched processing in generalized function."""
    batch_size = 2

    # 2D batched case
    x1 = torch.tensor([[0.5, 1.5, 0.8], [1.2, 0.3, 1.7]])
    x2 = torch.tensor([[0.5, 1.5, 0.8], [1.2, 0.3, 1.7]])

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition([x1, x2], [bins1, bins2])

    assert result.shape == (2, 3, 3)

    # Test against individual batch processing
    for b in range(batch_size):
        batch_result = cloud_in_cell_charge_deposition([x1[b], x2[b]], [bins1, bins2])
        assert torch.allclose(result[b], batch_result, atol=1e-6)


def test_deposit_charge_cic_uniform_spacing_validation():
    """Test uniform spacing validation in generalized function."""
    x = torch.tensor([0.5, 1.5])
    bins_uniform = torch.tensor([0.0, 1.0, 2.0])
    bins_non_uniform = torch.tensor([0.0, 1.0, 2.5])

    # Should work with uniform spacing
    result = cloud_in_cell_charge_deposition([x], [bins_uniform])
    assert result.shape == (3,)

    # Should fail with non-uniform spacing
    with pytest.raises(ValueError, match="bins\\[0\\] must have uniform spacing"):
        cloud_in_cell_charge_deposition([x], [bins_non_uniform])

    # Test for 2D case
    with pytest.raises(ValueError, match="bins\\[1\\] must have uniform spacing"):
        cloud_in_cell_charge_deposition([x, x], [bins_uniform, bins_non_uniform])


def test_deposit_charge_cic_2d_backward_compatibility():
    """Test that 2D function still works as before through wrapper."""
    x1 = torch.tensor([0.5, 1.5])
    x2 = torch.tensor([0.5, 1.5])
    weights = torch.tensor([2.0, 3.0])

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    # Compare old-style call with generalized function
    result_2d = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2, weights)
    result_general = cloud_in_cell_charge_deposition([x1, x2], [bins1, bins2], weights)

    assert torch.allclose(result_2d, result_general)


def test_deposit_charge_cic_edge_particles_all_dims():
    """Test particles exactly on grid points in all dimensions."""

    # 1D case - particles within bounds
    x = torch.tensor([0.0, 1.0])  # Both are within [0.0, 2.0) since 1.0 < 2.0
    bins = torch.tensor([0.0, 1.0, 2.0])
    result_1d = cloud_in_cell_charge_deposition([x], [bins])
    # Particles at grid points contribute fully to their grid points
    expected_1d = torch.tensor([1.0, 1.0, 0.0])
    assert torch.allclose(result_1d, expected_1d)

    # 2D case - particles within bounds
    x1 = torch.tensor([0.0, 0.5])  # First at origin, second at center
    x2 = torch.tensor([0.0, 0.5])  # First at origin, second at center
    bins_2d = torch.tensor([0.0, 1.0])  # 2-element bins [0.0, 1.0)
    result_2d = cloud_in_cell_charge_deposition([x1, x2], [bins_2d, bins_2d])
    # Particle 1 at (0.0, 0.0): all charge goes to grid point (0,0)
    # Particle 2 at (0.5, 0.5): charge distributed to all 4 corners (0.25 each)
    expected_2d = torch.tensor([[1.25, 0.25], [0.25, 0.25]])
    assert torch.allclose(result_2d, expected_2d)

    # 3D case - particle at origin
    x1 = torch.tensor([0.0])
    x2 = torch.tensor([0.0])
    x3 = torch.tensor([0.0])
    result_3d = cloud_in_cell_charge_deposition(
        [x1, x2, x3], [bins_2d, bins_2d, bins_2d]
    )
    expected_3d = torch.zeros(2, 2, 2)
    expected_3d[0, 0, 0] = 1.0
    assert torch.allclose(result_3d, expected_3d)
