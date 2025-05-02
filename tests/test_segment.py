import pytest
import torch

import cheetah


def test_subcell_start_end():
    """
    Test that `start` and `end` have proper defaults and fallbacks if the named element
    does not exist in the original segment.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name=f"drift_{i}")
            for i in range(10)
        ]
    )

    # Test standard behaviour
    assert len(segment.subcell().elements) == 10
    assert len(segment.subcell(start="drift_3").elements) == 7
    assert len(segment.subcell(end="drift_4").elements) == 5
    assert len(segment.subcell(start="drift_3", end="drift_4").elements) == 2

    # Test with invalid start or end element
    with pytest.raises(ValueError):
        segment.subcell(start="drift_42")
    with pytest.raises(ValueError):
        segment.subcell(end="drift_42")
    with pytest.raises(ValueError):
        segment.subcell(start="drift_3", end="drift_42")
    assert len(segment.subcell(start="drift_3", end="drift_2").elements) == 0


def test_subcell_endpoint():
    """Test that the subcell method properly determines the new end element."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name=f"drift_{i}")
            for i in range(10)
        ]
    )

    # Test with fixed end
    assert len(segment.subcell("drift_2", "drift_7").elements) == 6
    assert len(segment.subcell("drift_2", "drift_7", include_end=True).elements) == 6
    assert len(segment.subcell("drift_2", "drift_7", include_end=False).elements) == 5

    # Test with open end
    assert len(segment.subcell("drift_2").elements) == 8
    assert len(segment.subcell("drift_2", include_end=True).elements) == 8
    assert len(segment.subcell("drift_2", include_end=False).elements) == 8


@pytest.mark.parametrize("is_recursive", [True, False])
def test_attr_setting_by_element_type_convenience_method(is_recursive):
    """
    Test that the convenience method for setting attributes by element type works as
    expected.
    """
    segment = cheetah.Segment(
        elements=[cheetah.Drift(length=torch.tensor(0.5)) for i in range(10)]
        + [
            cheetah.Segment(
                name="subsegment",
                elements=[cheetah.Drift(length=torch.tensor(0.4)) for i in range(5)],
            )
        ]
        + [cheetah.Quadrupole(length=torch.tensor(0.6))]
    )

    segment.set_attrs_on_every_element_of_type(
        cheetah.Drift, is_recursive=is_recursive, length=torch.tensor(4.2)
    )

    for element in segment.elements:
        if isinstance(element, cheetah.Drift):
            assert element.length == 4.2
        elif isinstance(element, cheetah.Segment):
            for subelement in element.elements:
                assert subelement.length == 4.2 if is_recursive else 0.5
        elif isinstance(element, cheetah.Quadrupole):
            assert element.length == 0.6
        else:
            raise ValueError(f"Unexpected element type: {type(element)}")


def test_beam_trajectory_beam_objects():
    """
    Test the `beam_property_trajectory` method in the case where it should return `Beam`
    objects.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.Quadrupole(length=torch.tensor(0.3)),
            cheetah.Drift(length=torch.tensor(0.2)),
        ]
    )
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    s_positions, longitudinal_beams = segment.beam_property_trajectory(
        incoming=incoming_beam
    )

    assert len(s_positions) == 4
    assert len(longitudinal_beams) == 4
    for longitudinal_beam in longitudinal_beams:
        assert isinstance(longitudinal_beam, incoming_beam.__class__)


def test_beam_trajectory_single_property():
    """
    Test the `beam_property_trajectory` method in the case where it should return a
    single property.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.Quadrupole(length=torch.tensor(0.3)),
            cheetah.Drift(length=torch.tensor(0.2)),
        ]
    )
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    s_positions, longitudinal_beta_xs = segment.beam_property_trajectory(
        incoming=incoming_beam, property_name="beta_x"
    )

    assert len(s_positions) == 4
    assert len(longitudinal_beta_xs) == 4
    assert isinstance(longitudinal_beta_xs, torch.Tensor)


def test_beam_trajectory_multiple_properties():
    """
    Test the `beam_property_trajectory` method in the case where it should return
    multiple properties.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.Quadrupole(length=torch.tensor(0.3)),
            cheetah.Drift(length=torch.tensor(0.2)),
        ]
    )
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    s_positions, longitudinal_properties = segment.beam_property_trajectory(
        incoming=incoming_beam, property_name=("beta_x", "beta_y")
    )

    assert len(s_positions) == 4
    assert isinstance(longitudinal_properties, tuple)
    assert len(longitudinal_properties) == 2
    for longitudinal_property in longitudinal_properties:
        assert len(longitudinal_property) == 4
        assert isinstance(longitudinal_property, torch.Tensor)


def test_beam_trajectory_resolution():
    """
    Test the `beam_property_trajectory` method in the case where it should return a
    single property with a specified resolution.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.Quadrupole(length=torch.tensor(0.3)),
            cheetah.Drift(length=torch.tensor(0.2)),
        ]
    )
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    s_positions, longitudinal_beta_xs = segment.beam_property_trajectory(
        incoming=incoming_beam, property_name="beta_x", resolution=0.1
    )

    assert len(s_positions) == 11
    assert len(longitudinal_beta_xs) == 11
