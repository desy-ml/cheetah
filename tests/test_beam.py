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


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_beam_summary(BeamClass):
    """Test that `Beam.summary` returns expected dict and string representations."""
    beam = BeamClass.from_parameters()
    data = beam.summary(as_string=False)
    assert isinstance(data, dict)
    assert data["type"] == BeamClass.__name__
    assert "species" in data
    assert "energy" in data
    assert "sigma_x" in data

    string_summary = beam.summary(as_string=True)
    assert isinstance(string_summary, str)
    assert f"Beam Summary ({BeamClass.__name__}):" in string_summary
    assert "Species:" in string_summary


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_beam_summary_preserves_singleton_vector_dimensions(BeamClass):
    """Keep singleton dimensions in the structured optics summary."""
    beam = BeamClass.from_twiss(
        energy=torch.tensor([1e8]),
        beta_x=torch.tensor([1.0]),
        beta_y=torch.tensor([1.0]),
        alpha_x=torch.tensor([0.5]),
        emittance_x=torch.tensor([1e-6]),
        emittance_y=torch.tensor([1e-6]),
    )
    data = beam.summary()
    assert data["energy"] == [1e8]
    assert data["alpha_x"] == [beam.alpha_x.item()]
    assert len(data["sigma_tau"]) == 1
