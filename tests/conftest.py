import random
from typing import Callable

import pytest
import torch

import cheetah
from cheetah.utils import is_mps_available_and_functional

ELEMENT_SUBCLASSES_ARGS = {
    cheetah.Aperture: [{}],
    cheetah.BPM: [{}],
    cheetah.Cavity: [{"length": torch.tensor(1.0)}],
    cheetah.CustomTransferMap: [{"predefined_transfer_map": torch.eye(7)}],
    cheetah.Dipole: [{"length": torch.tensor(1.0), "angle": torch.tensor([1.0, -2.0])}],
    cheetah.Drift: [{"length": torch.tensor([1.0, -1.0])}],
    cheetah.HorizontalCorrector: [
        {
            "length": torch.tensor(1.0),
            "angle": torch.tensor([1.0, -2.0]),
        }
    ],
    cheetah.Marker: [{}],
    cheetah.Quadrupole: [
        {
            "length": torch.tensor(1.0),
            "k1": torch.tensor([1.0, -2.0]),
        }
    ],
    cheetah.RBend: [{"length": torch.tensor(1.0), "angle": torch.tensor([1.0, -2.0])}],
    cheetah.Screen: [{}],
    cheetah.Segment: [{"elements": [cheetah.Drift(length=torch.tensor(1.0))]}],
    cheetah.Sextupole: [{"length": torch.tensor(1.0), "k2": torch.tensor([1.0, -2.0])}],
    cheetah.Solenoid: [{"length": torch.tensor(1.0), "k": torch.tensor([1.0, -2.0])}],
    cheetah.SpaceChargeKick: [{"effect_length": torch.tensor(1.0)}],
    cheetah.TransverseDeflectingCavity: [{"length": torch.tensor(1.0)}],
    cheetah.Undulator: [{"length": torch.tensor(1.0)}],
    cheetah.VerticalCorrector: [
        {
            "length": torch.tensor(1.0),
            "angle": torch.tensor([1.0, -2.0]),
        }
    ],
}


def pytest_addoption(parser) -> None:
    """Add --seed option to pytest command line interface."""
    parser.addoption(
        "--seed", action="store", type=int, default=random.Random().getrandbits(32)
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    Add a `pytest.mark.parametrize`-like marker that runs a test with an testcases of
    each Cheetah `Element` subclass.
    """
    for_every_element_marker = metafunc.definition.get_closest_marker(
        "for_every_element"
    )

    if for_every_element_marker is None:
        # No marker found, return early
        return

    arg_name = (
        for_every_element_marker.args[0]
        if for_every_element_marker.args[0] is not None
        else for_every_element_marker.kwargs["arg_name"]
    )

    for_every_element(
        metafunc,
        arg_name,
        only_if=for_every_element_marker.kwargs.get("only_if"),
        except_if=for_every_element_marker.kwargs.get("except_if"),
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

    # Prevent torch from using non-deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Manually seed all torch PRNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if is_mps_available_and_functional():
        torch.mps.manual_seed(seed)


@pytest.fixture(autouse=True)
def fail_because_no_testcase_defined(request):
    """
    Mark a test to fail because no testcases are defined for the given `Element`
    subclass.
    """
    if request.node.get_closest_marker("fail_because_no_testcase_defined") is not None:
        pytest.fail("No Testcases are defined for this Element subclass.")


@pytest.fixture
def default_torch_dtype(request):
    """Temporarily set the default torch dtype for the test to a requested value."""
    tmp_dtype_for_test = request.param
    previous_dtype = torch.get_default_dtype()

    torch.set_default_dtype(tmp_dtype_for_test)

    # Return the requested dtype for the test and let the test run
    yield tmp_dtype_for_test

    torch.set_default_dtype(previous_dtype)


def for_every_element(
    metafunc: pytest.Metafunc,
    arg_name: str,
    only_if: Callable[[cheetah.Element], bool] | None = None,
    except_if: Callable[[cheetah.Element], bool] | None = None,
) -> None:
    """
    This marker can be used by adding the `@pytest.mark.for_every_element` marker to a
    test function. The user must specify the argument name in the marker via the first
    positional argument or the `arg_name` keyword argument, and define an argument of
    that name in the test function's signature.

    The `only_if` and `except_if` keyword arguments provide means to filter which
    testcases are executed. They expect single-argument lambda expressions that evaluate
    to a bool and are passed the testcase elements one-by-one.
    """
    only_if = only_if if only_if is not None else lambda _: True
    except_if = except_if if except_if is not None else lambda _: False

    # Recursively discover all subclasses of `Element`
    def get_all_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from get_all_subclasses(subclass)
            yield subclass

    all_element_subclasses = get_all_subclasses(cheetah.Element)

    # Generate test cases for all element subclasses
    testcase_dict = {
        subclass: (
            subclass(**testcase).clone()
            if testcase is not None
            else pytest.param(None, marks=pytest.mark.fail_because_no_mwe_args_defined)
        )
        for subclass in all_element_subclasses
        for testcase in ELEMENT_SUBCLASSES_ARGS.get(subclass, [None])
    }

    # Remove test cases according to `only_if` and `except_if`
    filtered_dict = {
        label: testcase
        for label, testcase in testcase_dict.items()
        if only_if(testcase) and not except_if(testcase)
    }

    metafunc.parametrize(arg_name, filtered_dict.values(), ids=filtered_dict.keys())
