import torch

import cheetah

double_precision_epsilon = torch.finfo(torch.float64).eps


def test_tilted_quad_transfer_matrix_precision():
    """
    Test the precision of the transfer matrix for a tilted Quadrupole element with k1=0.
    The transfer matrix should be close to the expected values for double precision.
    """
    length = torch.tensor(0.5, dtype=torch.float64)
    k1 = torch.tensor(0.0, dtype=torch.float64)
    tilt = torch.tensor(torch.pi / 4, dtype=torch.float64)
    energy = torch.tensor(1e9, dtype=torch.float64)
    spiecies = cheetah.Species("electron")

    # Create the normal Quadrupole element
    quad = cheetah.Quadrupole(
        length=length,
        k1=k1,
        name="test_quad",
    )

    skew_quad = cheetah.Quadrupole(
        length=length,
        k1=k1,
        tilt=tilt,
        name="test_skew_quad",
    )

    # Create a drift section with the same length
    drift = cheetah.Drift(
        length=length,
        name="test_drift",
    )

    # Define the incoming beam
    tm_quad = quad.transfer_map(energy, spiecies)
    tm_skew_quad = skew_quad.transfer_map(energy, spiecies)
    tm_drift = drift.transfer_map(energy, spiecies)

    assert torch.allclose(
        tm_quad, tm_skew_quad, atol=double_precision_epsilon
    ), "Transfer matrices for normal and skew quadrupole should be close."

    assert torch.allclose(
        tm_quad, tm_drift, atol=double_precision_epsilon
    ), "Transfer matrix for drift should be close to that of the quadrupole."

    assert torch.allclose(
        tm_skew_quad, tm_drift, atol=double_precision_epsilon
    ), "Transfer matrix for drift should be close to that of the skew quadrupole."
