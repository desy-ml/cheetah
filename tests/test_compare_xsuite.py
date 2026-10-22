from pathlib import Path

import numpy as np
import pytest
import torch

import cheetah
from cheetah.track_methods import base_ttensor


@pytest.mark.parametrize(
    "case, length, k1, k2, hx",
    [
        ("combined_short", 0.7, 0.3, 0.8, 0.5),
        ("combined_long", 1.3, 0.3, 0.8, 0.5),
        ("defocusing", 1.3, -0.8, 0.8, 0.5),
        ("zero_horizontal_focusing", 1.3, -0.25, 0.8, 0.5),
        ("resonance", 1.3, -0.2, 0.8, 1.0),
        ("near_resonance", 1.3, -0.196, 0.8, 1.0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_second_order_body_map(case, length, k1, k2, hx, dtype):
    """
    Compare explicitly every body-map coefficient to an Xsuite map.
    """

    # Body map only: no entrance/exit edges, matching the Xsuite reference bend.
    actual = base_ttensor(
        length=torch.tensor(length, dtype=dtype),  # m
        k1=torch.tensor(k1, dtype=dtype),  # quadrupole strength [m^-2]
        k2=torch.tensor(k2, dtype=dtype),  # sextupole strength [m^-3]
        hx=torch.tensor(hx, dtype=dtype),  # curvature [m^-1]
        species=cheetah.Species("electron", dtype=dtype),
        energy=torch.tensor(1e9, dtype=dtype),  # total reference energy [eV]
    )
    path = Path(__file__).parent / "resources" / "xsuite" / "second_order_maps.npz"
    with np.load(path) as maps:
        expected = torch.tensor(maps[case], dtype=dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5 if dtype == torch.float32 else 1e-9,
        atol=1e-6 if dtype == torch.float32 else 1e-10,
    )
