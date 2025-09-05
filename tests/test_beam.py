import pytest
import torch

import cheetah


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_change_beam_attribute_dtype(BeamClass):
    """
    Test that all beam attributes are properly converted between different dtypes. This
    transitively tests that all attributes are registered as PyTorch buffers.
    """
    beam = BeamClass.from_parameters()
    beam_attributes = beam.UNVECTORIZED_NUM_ATTR_DIMS.keys()

    # Ensure all attributes have the same dtype initially
    for attribute in beam_attributes:
        assert getattr(beam, attribute).dtype == torch.float32

    beam.to(torch.float64)

    # Ensure all attributes have been converted to float64
    for attribute in beam_attributes:
        assert getattr(beam, attribute).dtype == torch.float64


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
