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


def pytest_addoption(parser) -> None:
    """Add --seed option to pytest command line interface."""
    parser.addoption(
        "--seed", action="store", type=int, default=random.Random().getrandbits(32)
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    Add a `pytest.mark.parametrize`-like marker that runs a test with an MWE of each
    Cheetah `Element` subclass.

    This marker can be used by adding the `@pytest.mark.for_every_mwe_element` marker to
    a test function. The user must specify the argument name in the marker via the
    `arg_name` keyword argument, and define an argument of that name in the test
    function's signature. Element classes can be excluded from the test by using the
    `except_for` keyword argument, which takes a list of element classes.
    """
    for_every_mwe_element_marker = metafunc.definition.get_closest_marker(
        "for_every_mwe_element"
    )

    if for_every_mwe_element_marker is None:
        # No marker found, return early
        return

    arg_name = (
        for_every_mwe_element_marker.args[0]
        if for_every_mwe_element_marker.args[0] is not None  # TODO
        else for_every_mwe_element_marker.kwargs["arg_name"]
    )
    except_for = (
        for_every_mwe_element_marker.args[1]
        if len(for_every_mwe_element_marker.args) > 1
        else for_every_mwe_element_marker.kwargs.get("except_for", [])
    )

    all_element_subclasses = cheetah.Element.__subclasses__()
    element_subclasses_to_test = [
        subclass for subclass in all_element_subclasses if subclass not in except_for
    ]

    # Generate minimal working examples
    element_mwe_dict = {
        subclass.__name__: (
            subclass
            if subclass in ELEMENT_CLASS_DEFAULT_ARGS
            else pytest.param(None, marks=pytest.mark.no_mwe)
        )
        for subclass in element_subclasses_to_test
    }

    metafunc.parametrize(
        arg_name, element_mwe_dict.values(), ids=element_mwe_dict.keys()
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


@pytest.fixture(autouse=True)
def fail_no_mwe(request):
    """Ensure that test with the 'no_mwe' marker fail."""
    if request.node.get_closest_marker("no_mwe") is not None:
        pytest.fail("There is no minimal working example for the given element class!")
