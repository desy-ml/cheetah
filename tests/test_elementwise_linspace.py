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
    torch.testing.assert_close(result[:, :, 0], start)
    torch.testing.assert_close(result[:, :, -1], end)

    # Check that the values are linearly interpolated for each linspace
    torch.testing.assert_close(result[0, 0, :], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    torch.testing.assert_close(result[0, 1, :], torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    torch.testing.assert_close(result[0, 2, :], torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0]))
    torch.testing.assert_close(result[1, 0, :], torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
    torch.testing.assert_close(result[1, 1, :], torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0]))
    torch.testing.assert_close(
        result[1, 2, :], torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0])
    )
