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


def test_elementwise_longitudinal_beam_generator():
    """
    Test that the longitudinal beam generator works as expected and generates the
    correct number of beams.
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

    longitudinal_beam_generator = segment.beam_along_segment_generator(
        incoming=incoming_beam
    )
    longitudinal_beam_list = list(longitudinal_beam_generator)

    assert len(longitudinal_beam_list) == 4
    for longitudinal_beam in longitudinal_beam_list:
        assert isinstance(longitudinal_beam, incoming_beam.__class__)


def test_resolution_longitudinal_beam_generator():
    """
    Test that the longitudinal beam generator works as expected and generates the
    correct number of beams with a specified resolution.
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

    longitudinal_beam_generator = segment.beam_along_segment_generator(
        incoming=incoming_beam, resolution=0.1
    )
    longitudinal_beam_list = list(longitudinal_beam_generator)

    assert len(longitudinal_beam_list) == 11
    for longitudinal_beam in longitudinal_beam_list:
        assert isinstance(longitudinal_beam, incoming_beam.__class__)


@pytest.mark.parametrize(
    "attr_names", ["beta_x", ("beta_x",), ("s", "beta_x"), ("x", "mu_x")]
)
def test_longitudinal_beam_metric(attr_names):
    """
    Test that the convenience method for computing a attributes along the lattice works
    as expected. Focus is put on the return being a single tensor when an attribute
    string is passed, and a tuple of tensors when a tuple of metric strings is passed
    (regardless of the length of the tuple).
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.Quadrupole(length=torch.tensor(0.3)),
            cheetah.Drift(length=torch.tensor(0.2)),
        ]
    )
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    result = segment.get_beam_attrs_along_segment(attr_names, incoming_beam)

    if isinstance(attr_names, str):
        assert isinstance(result, torch.Tensor)
        assert len(result) == 4
    else:
        assert isinstance(result, tuple)
        assert len(result) == len(attr_names)
        for attr_result, attr_name in zip(result, attr_names):
            assert isinstance(attr_result, torch.Tensor)
            assert attr_result.shape == (
                (4, incoming_beam.num_particles) if attr_name == "x" else (4,)
            )
