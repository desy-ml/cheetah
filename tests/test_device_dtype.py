import inspect

import pytest
import torch

import cheetah
from cheetah.utils import is_mps_available_and_functional


@pytest.mark.initialize_elements
@pytest.mark.parametrize(
    "target_device",
    [
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_move_element_to_device(mwe_element, target_device: torch.device):
    """Test that elements can be successfully moved to a different device."""

    # Test that by default the element is on the CPU
    for buffer in mwe_element.buffers():
        assert buffer.dtype == "cpu"

    # Move the element to the target device
    mwe_element.to(target_device)

    # Test that the element is now on the target device
    for buffer in mwe_element.buffers():
        assert buffer.dtype == target_device.type


@pytest.mark.initialize_elements(except_for=[cheetah.Marker, cheetah.Segment])
def test_forced_element_dtype(mwe_element):
    """
    Test that the dtype is properly overridden for all element classes.
    """
    # Ensure everything is float32 by default
    for buffer in mwe_element.buffers():
        assert buffer.dtype == torch.float32

    # Create new element with overriden dtype
    double_element = mwe_element.double()
    half_element = double_element.__class__(
        **{
            feature: getattr(double_element, feature)
            for feature in double_element.defining_features
        },
        dtype=torch.float16,
    )

    for buffer in half_element.buffers():
        assert buffer.dtype == torch.float16


@pytest.mark.initialize_elements(
    except_for=[cheetah.BPM, cheetah.Marker, cheetah.Segment]
)
def test_infer_element_dtype(mwe_element):
    """
    Test that the dtype is properly inferred for all element classes.
    """
    # Ensure everything is float32 by default
    for buffer in mwe_element.buffers():
        assert buffer.dtype == torch.float32

    # Create new element and infer the dtype from passed buffers
    double_element = mwe_element.double()
    inferred_element = double_element.__class__(
        **{
            feature: getattr(double_element, feature)
            for feature in double_element.defining_features
        }
    )

    for buffer in inferred_element.buffers():
        assert buffer.dtype == torch.float64


@pytest.mark.initialize_elements
def test_conflicting_element_dtype(mwe_element):
    """Test that creating elements with conflicting argument dtypes fails."""
    arguments = inspect.signature(mwe_element.__init__).parameters

    # Extract required arguments for this element class
    required_arguments = {
        name: getattr(mwe_element, name)
        for name, properties in arguments.items()
        if name != "self" and properties.default is inspect._empty
    }

    # Generate list of optional tensor arguments whose dtype can be varied
    optional_tensor_arguments = {
        feature: getattr(mwe_element, feature)
        for feature in mwe_element.defining_features
        if isinstance(getattr(mwe_element, feature), torch.Tensor)
        and feature not in required_arguments
    }

    # Ensure that at least one tensor is part of the arguments that are passed each call
    if (
        not any(
            isinstance(value, torch.Tensor) for value in required_arguments.values()
        )
        and len(optional_tensor_arguments) > 0
    ):
        name, value = optional_tensor_arguments.popitem()
        required_arguments[name] = value

    # Vary individual optional arguments dtype
    for name, value in optional_tensor_arguments.items():
        with pytest.raises(AssertionError):
            # Contains conflicting dtype
            mwe_element.__class__(**{name: value.double()}, **required_arguments)

        # Conflict can be overriden by manual dtype selection
        mwe_element.__class__(
            **{name: value.double()}, **required_arguments, dtype=torch.float16
        )


@pytest.mark.initialize_elements
def test_change_element_dtype(mwe_element):
    """Test that elements can be successfully changed to a different dtype."""

    # Test that by default the element is of dtype float32
    for buffer in mwe_element.buffers():
        assert buffer.dtype == torch.float32

    # Change the dtype of the element
    mwe_element.to(torch.float64)

    # Test that the element is now of dtype float64
    for buffer in mwe_element.buffers():
        assert buffer.dtype == torch.float64


@pytest.mark.parametrize(
    "BeamClass",
    [
        cheetah.ParameterBeam,
        cheetah.ParticleBeam,
    ],
)
@pytest.mark.parametrize(
    "target_device",
    [
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_move_beam_to_device(BeamClass, target_device: torch.device):
    """Test that a particle beam can be successfully moved to a different device."""
    beam = BeamClass.from_parameters()

    # Test that by default the beam is on the CPU
    for buffer in beam.buffers():
        assert buffer.device.type == "cpu"

    beam.to(target_device)

    # Test that the beam is now on the target device
    for buffer in beam.buffers():
        assert buffer.device.type == target_device.type


@pytest.mark.parametrize(
    "BeamClass",
    [
        cheetah.ParameterBeam,
        cheetah.ParticleBeam,
    ],
)
def test_forced_beam_dtype(BeamClass):
    """
    Test that the dtype is properly overriden on beam creation.
    """
    beam = BeamClass.from_parameters(
        mu_x=torch.tensor(1e-5, dtype=torch.float32), dtype=torch.float64
    )
    for buffer in beam.buffers():
        assert buffer.dtype == torch.float64

    beam = BeamClass.from_twiss(
        beta_x=torch.tensor(1.0, dtype=torch.float16),
        beta_y=torch.tensor(2.0, dtype=torch.float64),
        dtype=torch.float32,
    )
    for buffer in beam.buffers():
        assert buffer.dtype == torch.float32


@pytest.mark.parametrize(
    "BeamClass",
    [
        cheetah.ParameterBeam,
        cheetah.ParticleBeam,
    ],
)
def test_infer_beam_dtype(BeamClass):
    """
    Test that the dtype is properly inferred on beam creation.
    """
    beam = BeamClass.from_parameters(mu_x=torch.tensor(1e-5, dtype=torch.float64))
    for buffer in beam.buffers():
        assert buffer.dtype == torch.float64

    beam = BeamClass.from_twiss(
        beta_x=torch.tensor(1.0, dtype=torch.float64),
        beta_y=torch.tensor(2.0, dtype=torch.float64),
    )
    for buffer in beam.buffers():
        assert buffer.dtype == torch.float64


@pytest.mark.parametrize(
    "BeamClass",
    [
        cheetah.ParameterBeam,
        cheetah.ParticleBeam,
    ],
)
def test_conflicting_beam_dtype(BeamClass):
    """Test if creating a beam with conflicting argument dtypes fails."""
    with pytest.raises(AssertionError):
        BeamClass.from_twiss(
            beta_x=torch.tensor(1.0, dtype=torch.float32),
            beta_y=torch.tensor(2.0, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "BeamClass",
    [
        cheetah.ParameterBeam,
        cheetah.ParticleBeam,
    ],
)
def test_change_beam_dtype(BeamClass):
    """Test that beams can be successfully changed to a different dtype."""
    beam = BeamClass.from_parameters(dtype=torch.float64)
    beam.to(torch.float16)

    # Test that the beam is now of dtype float16
    for buffer in beam.buffers():
        assert buffer.dtype == torch.float16


@pytest.mark.parametrize(
    "BeamClass",
    [
        cheetah.ParameterBeam,
        cheetah.ParticleBeam,
    ],
)
def test_transformed_beam_dtype(BeamClass):
    """
    Test that Beam.transformed_to keeps the dtype by default.
    """
    beam = BeamClass.from_parameters(mu_x=torch.tensor(1e-5), dtype=torch.float64)

    # Verify the dtype is kept by default
    transformed_beam = beam.transformed_to(mu_x=torch.tensor(-2e-5))
    for buffer in transformed_beam.buffers():
        assert buffer.dtype == torch.float64

    # Check that the manual dtype selection works
    transformed_beam = beam.transformed_to(
        mu_x=torch.tensor(-2e-5), dtype=torch.float32
    )
    for buffer in transformed_beam.buffers():
        assert buffer.dtype == torch.float32
