import random

import pytest
import torch

from cheetah.utils import is_mps_available_and_functional


def pytest_addoption(parser):
    """Add --seed option to pytest command line interface."""
    parser.addoption(
        "--seed", action="store", type=int, default=random.Random().getrandbits(32)
    )


def pytest_report_header(config) -> str:
    """Report chosen seed to command line for reproducability."""
    return f"seed: {config.getoption('--seed')}"


@pytest.fixture(autouse=True)
def seed_random_generators(request):
    """
    Manually seed all torch random generators. This ensures that test failures are
    determinstic and not appearing randomly between runs.
    """

    # Determine seed from command line option
    seed = request.config.getoption("--seed")

    # Manually seed all torch PRNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if is_mps_available_and_functional():
        torch.mps.manual_seed(seed)


@pytest.fixture
def default_dtype(request):
    """Test fixture for parametrizing the default torch dtype."""
    stored_dtype = torch.get_default_dtype()

    # Set desired default dtype
    torch.set_default_dtype(request.param)

    # Execute test
    yield

    # Restore original default dtype
    torch.set_default_dtype(stored_dtype)
