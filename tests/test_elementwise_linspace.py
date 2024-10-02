import torch

from cheetah.utils import elementwise_linspace


def test_example():
    """ "Tests an example case with two 2D tensors."""
    start = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    end = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    steps = 5

    result = elementwise_linspace(start, end, steps)

    # Check shape
    assert result.shape == (2, 3, 5)

    # Check that edges are correct
    assert torch.allclose(result[:, :, 0], start)
    assert torch.allclose(result[:, :, -1], end)

    # Check that the values are linearly interpolated for each linspace
    assert torch.allclose(result[0, 0, :], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.allclose(result[0, 1, :], torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    assert torch.allclose(result[0, 2, :], torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0]))
    assert torch.allclose(result[1, 0, :], torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
    assert torch.allclose(result[1, 1, :], torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0]))
    assert torch.allclose(result[1, 2, :], torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0]))
