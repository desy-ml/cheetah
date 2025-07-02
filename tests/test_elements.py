import pytest
import torch

import cheetah
from cheetah.utils import is_mps_available_and_functional


@pytest.mark.for_every_element("element")
def test_element_subclasses_is_active_boolean(element):
    """
    Test that the `is_active` property of all `Element` subclasses returns a boolean if
    the element class has an `is_active` property.
    """
    assert not hasattr(element, "is_active") or isinstance(element.is_active, bool)


@pytest.mark.for_every_element("element")
def test_all_element_subclasses_is_skippable_boolean(element):
    """
    Test that the `is_skippable` property of all `Element` subclasses returns a boolean.
    """
    assert isinstance(element.is_skippable, bool)


@pytest.mark.for_every_element("element")
def test_defining_features_dtype(element):
    """
    Test that all defining features of `Element` subclasses that are `torch.Tensor`are
    properly converted between different dtypes. This transitively tests if all defining
    features are registered as pytorch buffers.
    """

    # Ensure all features have the same dtype initially
    for feature in element.defining_tensors:
        assert getattr(element, feature).dtype == torch.float32

    element.to(torch.float64)

    # Ensure all features have been converted to float64
    for feature in element.defining_tensors:
        assert getattr(element, feature).dtype == torch.float64


@pytest.mark.for_every_element("element")
@pytest.mark.parametrize(
    "device, dtype",
    [
        pytest.param(
            torch.device("cpu"),
            torch.float16,
            marks=pytest.mark.skip(reason="Float16 not currently supported"),
        ),
        (torch.device("cpu"), torch.float32),
        (torch.device("cpu"), torch.float64),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=pytest.mark.skip(reason="Float16 not currently supported"),
        ),
        pytest.param(
            torch.device("cuda"),
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            torch.float16,
            marks=pytest.mark.skip(reason="Float16 not currently supported"),
        ),
        pytest.param(
            torch.device("mps"),
            torch.float32,
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
    ids=[
        "cpu-float16",
        "cpu-float32",
        "cpu-float64",
        "cuda-float16",
        "cuda-float32",
        "mps-float16",
        "mps-float32",
    ],
)
def test_particle_beam_tracking_with_device_and_dtype(element, device, dtype):
    """
    Test that element subclasses work correctly on various devices and with various
    dtypes if tracked with a `ParticleBeam`
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.25)),
            element,
            cheetah.Drift(length=torch.tensor(0.25)),
        ]
    ).to(device=device, dtype=dtype)
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        total_charge=torch.tensor(1e-9),
        mu_x=torch.tensor(5e-5),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
    ).to(device=device, dtype=dtype)

    # Run in part to see if errors are raised
    outgoing_beam = segment.track(incoming_beam)

    # Check device and dtype of the output
    for attribute in outgoing_beam.UNVECTORIZED_NUM_ATTR_DIMS.keys():
        assert getattr(outgoing_beam, attribute).device.type == device.type
        assert getattr(outgoing_beam, attribute).dtype == dtype


@pytest.mark.for_every_element(
    "element",
    xfail_if=lambda element: isinstance(
        element, (cheetah.SpaceChargeKick, cheetah.TransverseDeflectingCavity)
    )
    or (isinstance(element, cheetah.Aperture) and element.is_active)
    or (
        isinstance(
            element, (cheetah.Dipole, cheetah.Drift, cheetah.Quadrupole, cheetah.RBend)
        )
        and element.tracking_method == "bmadx"
    ),
)
@pytest.mark.parametrize(
    "device, dtype",
    [
        pytest.param(
            torch.device("cpu"),
            torch.float16,
            marks=pytest.mark.skip(reason="Float16 not currently supported"),
        ),
        (torch.device("cpu"), torch.float32),
        (torch.device("cpu"), torch.float64),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=pytest.mark.skip(reason="Float16 not currently supported"),
        ),
        pytest.param(
            torch.device("cuda"),
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            torch.float16,
            marks=pytest.mark.skip(reason="Float16 not currently supported"),
        ),
        pytest.param(
            torch.device("mps"),
            torch.float32,
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
    ids=[
        "cpu-float16",
        "cpu-float32",
        "cpu-float64",
        "cuda-float16",
        "cuda-float32",
        "mps-float16",
        "mps-float32",
    ],
)
def test_parameter_beam_tracking_with_device_and_dtype(element, device, dtype):
    """
    Test that element subclasses work correctly on various devices and with various
    dtypes if tracked with a `ParticleBeam`
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.25)),
            element,
            cheetah.Drift(length=torch.tensor(0.25)),
        ]
    ).to(device=device, dtype=dtype)
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        total_charge=torch.tensor(1e-9),
        mu_x=torch.tensor(5e-5),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
    ).to(device=device, dtype=dtype)

    # Run in part to see if errors are raised
    outgoing_beam = segment.track(incoming_beam)

    # Check device and dtype of the output
    for attribute in outgoing_beam.UNVECTORIZED_NUM_ATTR_DIMS.keys():
        assert getattr(outgoing_beam, attribute).device.type == device.type
        assert getattr(outgoing_beam, attribute).dtype == dtype
