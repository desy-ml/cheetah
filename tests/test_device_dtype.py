import inspect

import pytest
import torch

import cheetah


@pytest.mark.for_every_element(
    "single_element", except_for=[cheetah.Marker, cheetah.Segment]
)
def test_forced_element_dtype(single_element):
    """
    Test that the dtype is properly overridden for all element classes.
    """
    # Ensure everything is float32 by default
    for feature in single_element.defining_tensors:
        assert getattr(single_element, feature).dtype == torch.float32

    # Create new element with overriden dtype
    double_element = single_element.double()
    half_element = double_element.__class__(
        **{
            feature: getattr(double_element, feature)
            for feature in double_element.defining_features
        },
        dtype=torch.float16,
    )

    for feature in half_element.defining_tensors:
        assert getattr(half_element, feature).dtype == torch.float16


@pytest.mark.for_every_element(
    "single_element", except_for=[cheetah.BPM, cheetah.Marker, cheetah.Segment]
)
def test_infer_element_dtype(single_element):
    """
    Test that the dtype is properly inferred for all element classes.
    """
    # Ensure everything is float32 by default
    for feature in single_element.defining_tensors:
        assert getattr(single_element, feature).dtype == torch.float32

    # Create new element and infer the dtype from passed buffers
    double_element = single_element.double()
    inferred_element = double_element.__class__(
        **{
            feature: getattr(double_element, feature)
            for feature in double_element.defining_features
        }
    )

    for feature in inferred_element.defining_tensors:
        assert getattr(inferred_element, feature).dtype == torch.float64


@pytest.mark.for_every_element("element")
def test_conflicting_element_dtype(element):
    """Test that creating elements with conflicting argument dtypes fails."""
    arguments = inspect.signature(element.__init__).parameters

    # Extract required arguments for this element class
    required_arguments = {
        name: getattr(element, name)
        for name, properties in arguments.items()
        if name != "self" and properties.default is inspect._empty
    }

    # Generate list of optional tensor arguments whose dtype can be varied
    optional_tensor_arguments = {
        feature: getattr(element, feature)
        for feature in element.defining_features
        if isinstance(getattr(element, feature), torch.Tensor)
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
            element.__class__(**{name: value.double()}, **required_arguments)

        # Conflict can be overriden by manual dtype selection
        element.__class__(
            **{name: value.double()}, **required_arguments, dtype=torch.float16
        )


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_forced_beam_dtype(BeamClass):
    """
    Test that the dtype is properly overriden on beam creation.
    """
    beam = BeamClass.from_parameters(
        mu_x=torch.tensor(1e-5, dtype=torch.float32), dtype=torch.float64
    )
    beam_attributes = beam.UNVECTORIZED_NUM_ATTR_DIMS.keys()

    for attribute in beam_attributes:
        assert getattr(beam, attribute).dtype == torch.float64

    beam = BeamClass.from_twiss(
        beta_x=torch.tensor(1.0, dtype=torch.float16),
        beta_y=torch.tensor(2.0, dtype=torch.float64),
        dtype=torch.float32,
    )
    for attribute in beam_attributes:
        assert getattr(beam, attribute).dtype == torch.float32


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_infer_beam_dtype(BeamClass):
    """
    Test that the dtype is properly inferred on beam creation.
    """
    beam = BeamClass.from_parameters(mu_x=torch.tensor(1e-5, dtype=torch.float64))
    beam_attributes = beam.UNVECTORIZED_NUM_ATTR_DIMS.keys()

    for attribute in beam_attributes:
        assert getattr(beam, attribute).dtype == torch.float64

    beam = BeamClass.from_twiss(
        beta_x=torch.tensor(1.0, dtype=torch.float64),
        beta_y=torch.tensor(2.0, dtype=torch.float64),
    )
    for attribute in beam_attributes:
        assert getattr(beam, attribute).dtype == torch.float64


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_conflicting_beam_dtype(BeamClass):
    """Test if creating a beam with conflicting argument dtypes fails."""
    with pytest.raises(AssertionError):
        BeamClass.from_twiss(
            beta_x=torch.tensor(1.0, dtype=torch.float32),
            beta_y=torch.tensor(2.0, dtype=torch.float64),
        )


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_transformed_beam_dtype(BeamClass):
    """Test that `Beam.transformed_to` keeps the dtype by default."""
    beam = BeamClass.from_parameters(mu_x=torch.tensor(1e-5), dtype=torch.float64)
    beam_attributes = beam.UNVECTORIZED_NUM_ATTR_DIMS.keys()

    # Verify the dtype is kept by default
    transformed_beam = beam.transformed_to(mu_x=torch.tensor(-2e-5))
    for attribute in beam_attributes:
        assert getattr(transformed_beam, attribute).dtype == torch.float64

    # Check that the manual dtype selection works
    transformed_beam = beam.transformed_to(
        mu_x=torch.tensor(-2e-5), dtype=torch.float32
    )
    for attribute in beam_attributes:
        assert getattr(transformed_beam, attribute).dtype == torch.float32
