import random

import pytest
import torch

import cheetah
from cheetah.utils import is_mps_available_and_functional

ELEMENT_CLASS_DEFAULT_ARGS = {
    cheetah.Aperture: {},
    cheetah.BPM: {},
    cheetah.Cavity: {"length": torch.tensor(1.0)},
    cheetah.CustomTransferMap: {"predefined_transfer_map": torch.eye(7)},
    cheetah.Dipole: {"length": torch.tensor(1.0)},
    cheetah.Drift: {"length": torch.tensor(1.0)},
    cheetah.HorizontalCorrector: {"length": torch.tensor(1.0)},
    cheetah.Marker: {},
    cheetah.Quadrupole: {"length": torch.tensor(1.0)},
    cheetah.Screen: {},
    cheetah.Segment: {"elements": [cheetah.Drift(length=torch.tensor(1.0))]},
    cheetah.Sextupole: {"length": torch.tensor(1.0)},
    cheetah.Solenoid: {"length": torch.tensor(1.0)},
    cheetah.SpaceChargeKick: {"effect_length": torch.tensor(1.0)},
    cheetah.TransverseDeflectingCavity: {"length": torch.tensor(1.0)},
    cheetah.Undulator: {"length": torch.tensor(1.0)},
    cheetah.VerticalCorrector: {"length": torch.tensor(1.0)},
}


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
def default_torch_dtype(request):
    """Temporarily set the default torch dtype for the test to a requested value."""
    tmp_dtype_for_test = request.param
    previous_dtype = torch.get_default_dtype()

    torch.set_default_dtype(tmp_dtype_for_test)

    # Return the requested dtype for the test and let the test run
    yield tmp_dtype_for_test

    torch.set_default_dtype(previous_dtype)


@pytest.fixture(params=cheetah.Element.__subclasses__())
def mwe_cheetah_element(request):
    """
    Run a test with a minimum working example of every cheetah Element. A default
    value is used for elements with mandatory arguments.
    """
    ElementClass = request.param

    test_marker = request.node.get_closest_marker("test_all_elements")
    if test_marker is None:
        pytest.fail(
            "Using 'mwe_cheetah_element' without mandatory marker 'test_all_elements'"
        )
    else:
        for argument, value in test_marker.kwargs.items():
            match (argument):
                case "except_for":
                    if ElementClass in value:
                        pytest.skip(f"skipped for class '{ElementClass}'")

    if ElementClass in ELEMENT_CLASS_DEFAULT_ARGS:
        # Clone to prevent global state between test calls
        return ElementClass(**ELEMENT_CLASS_DEFAULT_ARGS[ElementClass]).clone()
    else:
        pytest.fail("No default arguments for element class '{ElementClass}'")
