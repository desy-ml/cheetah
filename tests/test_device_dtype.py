import inspect

import pytest
import torch

import cheetah


@pytest.mark.for_every_element(
    "single_element",
    except_if=lambda e: isinstance(e, (cheetah.BPM, cheetah.Marker, cheetah.Segment)),
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
        beta_x=torch.tensor(1.0),
        beta_y=torch.tensor(2.0, dtype=torch.float64),
        dtype=torch.float32,
    )
    for attribute in beam_attributes:
        assert getattr(beam, attribute).dtype == torch.float32


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_transformed_beam_dtype(BeamClass):
    """Test that `Beam.transformed_to` retains the dtype."""
    beam = BeamClass.from_parameters(
        mu_x=torch.tensor(1e-5, dtype=torch.float64), dtype=torch.float64
    )
    beam_attributes = beam.UNVECTORIZED_NUM_ATTR_DIMS.keys()

    # Verify the dtype is kept
    transformed_beam = beam.transformed_to(
        mu_x=torch.tensor(-2e-5, dtype=torch.float64)
    )
    for attribute in beam_attributes:
        assert getattr(transformed_beam, attribute).dtype == torch.float64
