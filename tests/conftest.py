import pytest
import torch

SEED = 42


@pytest.fixture(autouse=True)
def seed_random_generators():
    """
    Manually seed all torch random generators. This ensures that test failures are
    determinstic and not appearing randomly between runs.
    """

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.mps.is_available():
        torch.mps.manual_seed(SEED)
