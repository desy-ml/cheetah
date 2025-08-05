import random
from typing import Callable

import pytest
import torch

import cheetah
from cheetah.utils import is_mps_available_and_functional

ELEMENT_SUBCLASSES_ARGS = {
    cheetah.Aperture: {"inactive": {"is_active": False}, "active": {"is_active": True}},
    cheetah.BPM: {"inactive": {"is_active": False}, "active": {"is_active": True}},
    cheetah.Cavity: {"default": {"length": torch.tensor(1.0)}},
    cheetah.CustomTransferMap: {"identity": {"predefined_transfer_map": torch.eye(7)}},
    cheetah.Dipole: {
        "linear": {
            "length": torch.tensor(1.0),
            "angle": torch.tensor([1.0, -2.0]),
            "tracking_method": "linear",
        },
        "second_order": {
            "length": torch.tensor(1.0),
            "angle": torch.tensor([1.0, -2.0]),
            "tracking_method": "second_order",
        },
        "drift_kick_drift": {
            "length": torch.tensor(1.0),
            "angle": torch.tensor([1.0, -2.0]),
            "tracking_method": "drift_kick_drift",
        },
    },
    cheetah.Drift: {
        "linear": {"length": torch.tensor([1.0, -1.0]), "tracking_method": "linear"},
        "second_order": {
            "length": torch.tensor([1.0, -1.0]),
            "tracking_method": "second_order",
        },
        "drift_kick_drift": {
            "length": torch.tensor([1.0, -1.0]),
            "tracking_method": "drift_kick_drift",
        },
    },
    cheetah.HorizontalCorrector: {
        "default": {"length": torch.tensor(1.0), "angle": torch.tensor([1.0, -2.0])}
    },
    cheetah.Marker: {"default": {}},
    cheetah.Quadrupole: {
        "linear": {
            "length": torch.tensor(1.0),
            "k1": torch.tensor([1.0, -2.0]),
            "tracking_method": "linear",
        },
        "second_order": {
            "length": torch.tensor(1.0),
            "k1": torch.tensor([1.0, -2.0]),
            "tracking_method": "second_order",
        },
        "drift_kick_drift": {
            "length": torch.tensor(1.0),
            "k1": torch.tensor([1.0, -2.0]),
            "tracking_method": "drift_kick_drift",
        },
    },
    cheetah.RBend: {
        "linear": {
            "length": torch.tensor(1.0),
            "angle": torch.tensor([1.0, -2.0]),
            "tracking_method": "linear",
        },
        "second_order": {
            "length": torch.tensor(1.0),
            "angle": torch.tensor([1.0, -2.0]),
            "tracking_method": "second_order",
        },
        "drift_kick_drift": {
            "length": torch.tensor(1.0),
            "angle": torch.tensor([1.0, -2.0]),
            "tracking_method": "drift_kick_drift",
        },
    },
    cheetah.Screen: {"default": {}},
    cheetah.Segment: {
        "default": {"elements": [cheetah.Drift(length=torch.tensor(1.0))]}
    },
    cheetah.Sextupole: {
        "linear": {
            "length": torch.tensor(1.0),
            "k2": torch.tensor([1.0, -2.0]),
            "tracking_method": "linear",
        },
        "second_order": {
            "length": torch.tensor(1.0),
            "k2": torch.tensor([1.0, -2.0]),
            "tracking_method": "second_order",
        },
    },
    cheetah.Solenoid: {
        "default": {"length": torch.tensor(1.0), "k": torch.tensor([1.0, -2.0])}
    },
    cheetah.SpaceChargeKick: {"default": {"effect_length": torch.tensor(1.0)}},
    cheetah.TransverseDeflectingCavity: {
        "inactive": {"length": torch.tensor(1.0), "voltage": torch.tensor(0.0)},
        "active": {"length": torch.tensor(1.0), "voltage": torch.tensor(1e6)},
    },
    cheetah.Undulator: {"default": {"length": torch.tensor(1.0)}},
    cheetah.VerticalCorrector: {
        "default": {"length": torch.tensor(1.0), "angle": torch.tensor([1.0, -2.0])}
    },
}


def pytest_addoption(parser) -> None:
    """Add --seed option to pytest command line interface."""
    parser.addoption(
        "--seed", action="store", type=int, default=random.Random().getrandbits(32)
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    Add a `pytest.mark.parametrize`-like marker that runs a test with a test case for
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
        except_if=for_every_element_marker.kwargs.get("except_if"),
        xfail_if=for_every_element_marker.kwargs.get("xfail_if"),
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
def fail_because_no_test_case_defined(request):
    """
    Mark a test as failing because no test cases are defined for the given `Element`
    subclass.
    """
    if request.node.get_closest_marker("fail_because_no_test_case_defined") is not None:
        pytest.fail("No test cases are defined for this Element subclass.")


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
    except_if: Callable[[cheetah.Element], bool] | None = None,
    xfail_if: Callable[[cheetah.Element], bool] | None = None,
) -> None:
    """
    This marker can be used by adding the `@pytest.mark.for_every_element` marker to a
    test function. The user must specify the argument name in the marker via the first
    positional argument or the `arg_name` keyword argument, and define an argument of
    that name in the test function's signature.

    The `except_if` keyword argument provides means to filter which test cases are
    executed. It expects a single-argument lambda expression that evaluates to a bool
    and is passed to the `Element` under test one-by-one.

    The `xfail_if` keyword argument can be used to mark test cases are are supposed to
    fail. It takes a lambda expression with the same signature as `except_if`.
    """
    except_if = except_if if except_if is not None else lambda _: False
    xfail_if = xfail_if if xfail_if is not None else lambda _: False

    # Recursively discover all subclasses of `Element`
    def get_all_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from get_all_subclasses(subclass)
            yield subclass

    all_element_subclasses = get_all_subclasses(cheetah.Element)

    # Generate test cases for all element subclasses
    test_cases = []
    for subclass in all_element_subclasses:
        if subclass in ELEMENT_SUBCLASSES_ARGS:
            subclass_test_cases = ELEMENT_SUBCLASSES_ARGS[subclass]
            for label, test_case_args in subclass_test_cases.items():
                # The clone prevents tests from modifying the test cases, which is
                # especially relevant for `Segment`. This is necessary since the
                # subclass constructors reference their arguments instead of copying.
                test_case = subclass(**test_case_args).clone()

                if not except_if(test_case):
                    test_cases.append(
                        pytest.param(
                            test_case,
                            id=(
                                f"{subclass.__name__}-{label}"
                                if len(subclass_test_cases) > 1
                                else subclass.__name__
                            ),
                            marks=pytest.mark.xfail if xfail_if(test_case) else (),
                        )
                    )
        else:
            test_cases.append(
                pytest.param(
                    None,
                    id=subclass.__name__,
                    marks=pytest.mark.fail_because_no_test_case_defined,
                )
            )

    metafunc.parametrize(arg_name, test_cases)
