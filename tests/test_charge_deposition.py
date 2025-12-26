import pytest
import torch

from cheetah.utils.charge_deposition import deposit_charge_cic_2d


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_deposit_charge_cic_2d_basic(device):
    """Test basic functionality of 2D CIC charge deposition."""
    # Simple test case with known expected result
    x1 = torch.tensor([0.5, 1.5], device=device, dtype=torch.float32)
    x2 = torch.tensor([0.5, 1.5], device=device, dtype=torch.float32)
    
    bins1 = torch.tensor([0.0, 1.0, 2.0], device=device)
    bins2 = torch.tensor([0.0, 1.0, 2.0], device=device)
    
    result = deposit_charge_cic_2d(x1, x2, bins1, bins2)
    
    # Expected result: each particle should be deposited equally among 4 grid points
    # Particle at (0.5, 0.5) -> corners at (0,0), (1,0), (0,1), (1,1) with weight 0.25 each
    # Particle at (1.5, 1.5) -> corners at (1,1), (2,1), (1,2), (2,2) with weight 0.25 each
    expected = torch.tensor([
        [0.25, 0.25, 0.0],
        [0.25, 0.5, 0.25],  # (1,1) gets contribution from both particles
        [0.0, 0.25, 0.25]
    ], device=device)
    
    assert result.shape == (3, 3)
    assert torch.allclose(result, expected, atol=1e-6)


def test_deposit_charge_cic_2d_with_weights():
    """Test CIC charge deposition with custom weights."""
    x1 = torch.tensor([0.5, 1.5])
    x2 = torch.tensor([0.5, 1.5])
    weights = torch.tensor([2.0, 3.0])
    
    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])
    
    result = deposit_charge_cic_2d(x1, x2, bins1, bins2, weights)
    
    # Particle 1 (weight=2.0) at (0.5, 0.5) -> 0.5 to each corner
    # Particle 2 (weight=3.0) at (1.5, 1.5) -> 0.75 to each corner
    expected = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.5, 1.25, 0.75],  # (1,1) gets 0.5 + 0.75 = 1.25
        [0.0, 0.75, 0.75]
    ])
    
    assert torch.allclose(result, expected, atol=1e-6)


def test_deposit_charge_cic_2d_batched():
    """Test batched CIC charge deposition."""
    batch_size = 2
    n_particles = 3
    
    x1 = torch.tensor([[0.5, 1.5, 0.8], [1.2, 0.3, 1.7]])
    x2 = torch.tensor([[0.5, 1.5, 0.8], [1.2, 0.3, 1.7]])
    
    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])
    
    result = deposit_charge_cic_2d(x1, x2, bins1, bins2)
    
    assert result.shape == (2, 3, 3)
    
    # Test that each batch is processed correctly
    for b in range(batch_size):
        batch_result = deposit_charge_cic_2d(x1[b], x2[b], bins1, bins2)
        assert torch.allclose(result[b], batch_result, atol=1e-6)


def test_deposit_charge_cic_2d_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test with particles exactly on grid points
    x1 = torch.tensor([0.0, 1.0, 2.0])
    x2 = torch.tensor([0.0, 1.0, 2.0])
    
    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])
    
    result = deposit_charge_cic_2d(x1, x2, bins1, bins2)
    
    # Particles exactly on grid points should deposit all charge to that point
    expected = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    assert torch.allclose(result, expected, atol=1e-6)


def test_deposit_charge_cic_2d_outside_bounds():
    """Test behavior with particles outside grid bounds."""
    # Particles outside the grid should be clamped to boundary cells
    x1 = torch.tensor([-0.5, 2.5])  # Outside grid bounds [0, 2]
    x2 = torch.tensor([0.5, 1.5])
    
    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])
    
    result = deposit_charge_cic_2d(x1, x2, bins1, bins2)
    
    # Should not raise errors and should clamp to boundary cells
    assert result.shape == (3, 3)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_deposit_charge_cic_2d_conservation():
    """Test charge conservation."""
    x1 = torch.tensor([0.3, 1.7, 0.9, 1.1])
    x2 = torch.tensor([0.4, 1.6, 0.8, 1.2])
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5])
    
    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])
    
    result = deposit_charge_cic_2d(x1, x2, bins1, bins2, weights)
    
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
        
        result = deposit_charge_cic_2d(x1, x2, bins1, bins2)
        
        assert result.dtype == dtype


def test_deposit_charge_cic_2d_empty_input():
    """Test behavior with empty particle arrays."""
    x1 = torch.empty((0,))
    x2 = torch.empty((0,))
    
    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])
    
    result = deposit_charge_cic_2d(x1, x2, bins1, bins2)
    
    expected = torch.zeros(3, 3)
    assert torch.allclose(result, expected)


def test_deposit_charge_cic_2d_non_uniform_spacing_error():
    """Test that non-uniform bin spacing raises an error."""
    x1 = torch.tensor([0.5, 1.5])
    x2 = torch.tensor([0.5, 1.5])
    
    # Non-uniform spacing should raise ValueError
    bins1_non_uniform = torch.tensor([0.0, 1.0, 2.5])  # Non-uniform: 1.0, 1.5
    bins2 = torch.tensor([0.0, 1.0, 2.0])
    
    with pytest.raises(ValueError, match="bins1 must have uniform spacing"):
        deposit_charge_cic_2d(x1, x2, bins1_non_uniform, bins2)
    
    # Test non-uniform bins2
    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2_non_uniform = torch.tensor([0.0, 1.0, 2.3])  # Non-uniform: 1.0, 1.3
    
    with pytest.raises(ValueError, match="bins2 must have uniform spacing"):
        deposit_charge_cic_2d(x1, x2, bins1, bins2_non_uniform)


def test_deposit_charge_cic_2d_insufficient_bins():
    """Test that insufficient number of bins raises an error."""
    x1 = torch.tensor([0.5])
    x2 = torch.tensor([0.5])
    
    # Single bin should raise error
    bins1_single = torch.tensor([0.0])
    bins2 = torch.tensor([0.0, 1.0])
    
    with pytest.raises(ValueError, match="bins1 and bins2 must have at least 2 elements"):
        deposit_charge_cic_2d(x1, x2, bins1_single, bins2)
    
    # Test single bin for bins2
    bins1 = torch.tensor([0.0, 1.0])
    bins2_single = torch.tensor([0.0])
    
    with pytest.raises(ValueError, match="bins1 and bins2 must have at least 2 elements"):
        deposit_charge_cic_2d(x1, x2, bins1, bins2_single)


def test_deposit_charge_cic_2d_single_particle():
    """Test with a single particle at various positions."""
    bins1 = torch.tensor([0.0, 1.0, 2.0, 3.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0, 3.0])
    
    # Test particle at center of a cell
    x1 = torch.tensor([1.5])
    x2 = torch.tensor([1.5])
    
    result = deposit_charge_cic_2d(x1, x2, bins1, bins2)
    
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
    
    result = deposit_charge_cic_2d(x1, x2, bins1, bins2)
    
    assert result.shape == (1, 2, 3, 3)
    
    # Test that charge is conserved
    total_charge = result.sum(dim=(-1, -2))  # Sum over grid dimensions
    expected_total = torch.ones(1, 2) * 2  # 2 particles per batch element
    
    assert torch.allclose(total_charge, expected_total, atol=1e-6)
