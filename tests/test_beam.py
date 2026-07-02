import pytest
import torch

import cheetah


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_change_beam_attribute_dtype(BeamClass):
    """
    Test that all beam features are properly converted between different dtypes. This
    transitively tests that all features are registered as PyTorch buffers.
    """
    beam = BeamClass.from_parameters()
    non_module_features = [
        feature for feature in beam.defining_features if feature != "species"
    ]

    # Ensure all features have the same dtype initially
    for feature in non_module_features:
        assert getattr(beam, feature).dtype == torch.float32

    beam.to(torch.float64)

    # Ensure all features have been converted to float64
    for feature in non_module_features:
        assert getattr(beam, feature).dtype == torch.float64


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_transformed_beam_dtype(BeamClass):
    """Test that `Beam.transformed_to` retains the dtype."""
    beam = BeamClass.from_parameters(
        mu_x=torch.tensor(1e-5, dtype=torch.float64), dtype=torch.float64
    )
    non_module_features = [
        feature for feature in beam.defining_features if feature != "species"
    ]

    # Verify the dtype is kept
    transformed_beam = beam.transformed_to(
        mu_x=torch.tensor(-2e-5, dtype=torch.float64)
    )

    for feature in non_module_features:
        assert getattr(transformed_beam, feature).dtype == torch.float64
