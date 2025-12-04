import pytest
import torch

import cheetah
from cheetah.utils.warnings import PhysicsWarning


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

    segment.set_attrs_on_every_element(
        filter_type=cheetah.Drift, is_recursive=is_recursive, length=torch.tensor(4.2)
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


@pytest.mark.parametrize("is_recursive", [True, False])
def test_attr_setting_by_element_name_convenience_method(is_recursive):
    """
    Test that the convenience method for setting attributes by element name works as
    expected.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name="drift_0"),
            cheetah.Drift(length=torch.tensor(0.5), name="drift_1"),
            cheetah.Drift(length=torch.tensor(0.5), name="drift_2"),
            cheetah.Segment(
                name="subsegment",
                elements=[
                    cheetah.Drift(length=torch.tensor(0.4), name="drift_0"),
                    cheetah.Drift(length=torch.tensor(0.4), name="drift_3"),
                ],
            ),
            cheetah.Quadrupole(length=torch.tensor(0.6), name="quad_0"),
        ]
    )

    # Filter by a single name using a string
    segment.set_attrs_on_every_element(
        filter_name="drift_0", is_recursive=is_recursive, length=torch.tensor(4.2)
    )

    # Check that only elements with name "drift_0" were modified
    assert segment.drift_0.length == 4.2
    assert segment.drift_1.length == 0.5
    assert segment.drift_2.length == 0.5
    assert segment.quad_0.length == 0.6
    # Check subsegment elements
    subsegment = segment.subsegment
    if is_recursive:
        assert subsegment.drift_0.length == 4.2
    else:
        assert subsegment.drift_0.length == 0.4
    assert subsegment.drift_3.length == 0.4


@pytest.mark.parametrize("is_recursive", [True, False])
def test_attr_setting_by_element_name_tuple_convenience_method(is_recursive):
    """
    Test that the convenience method for setting attributes by element name works with
    a tuple of names.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name="drift_0"),
            cheetah.Drift(length=torch.tensor(0.5), name="drift_1"),
            cheetah.Drift(length=torch.tensor(0.5), name="drift_2"),
            cheetah.Segment(
                name="subsegment",
                elements=[
                    cheetah.Drift(length=torch.tensor(0.4), name="drift_0"),
                    cheetah.Drift(length=torch.tensor(0.4), name="drift_3"),
                ],
            ),
            cheetah.Quadrupole(length=torch.tensor(0.6), name="quad_0"),
        ]
    )

    # Filter by multiple names using a tuple
    segment.set_attrs_on_every_element(
        filter_name=("drift_0", "drift_1"),
        is_recursive=is_recursive,
        length=torch.tensor(4.2),
    )

    # Check that only elements with name "drift_0" or "drift_1" were modified
    assert segment.drift_0.length == 4.2
    assert segment.drift_1.length == 4.2
    assert segment.drift_2.length == 0.5
    assert segment.quad_0.length == 0.6
    # Check subsegment elements
    subsegment = segment.subsegment
    if is_recursive:
        assert subsegment.drift_0.length == 4.2
    else:
        assert subsegment.drift_0.length == 0.4
    assert subsegment.drift_3.length == 0.4


@pytest.mark.parametrize("is_recursive", [True, False])
def test_attr_getting_by_element_name_convenience_method(is_recursive):
    """
    Test that the convenience method for getting attributes by element name works as
    expected.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name="drift_0"),
            cheetah.Drift(length=torch.tensor(0.6), name="drift_1"),
            cheetah.Drift(length=torch.tensor(0.7), name="drift_2"),
            cheetah.Segment(
                name="subsegment",
                elements=[
                    cheetah.Drift(length=torch.tensor(0.8), name="drift_0"),
                    cheetah.Drift(length=torch.tensor(0.9), name="drift_3"),
                ],
            ),
            cheetah.Quadrupole(length=torch.tensor(1.0), name="quad_0"),
        ]
    )

    # Filter by a single name
    lengths = segment.get_attr_from_every_element(
        "length", filter_name="drift_0", is_recursive=is_recursive
    )

    if is_recursive:
        assert len(lengths) == 2
        assert lengths[0] == 0.5
        assert lengths[1] == 0.8
    else:
        assert len(lengths) == 1
        assert lengths[0] == 0.5


@pytest.mark.parametrize("is_recursive", [True, False])
def test_attr_getting_by_element_name_tuple_convenience_method(is_recursive):
    """
    Test that the convenience method for getting attributes by element name works with
    a tuple of names.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name="drift_0"),
            cheetah.Drift(length=torch.tensor(0.6), name="drift_1"),
            cheetah.Drift(length=torch.tensor(0.7), name="drift_2"),
            cheetah.Segment(
                name="subsegment",
                elements=[
                    cheetah.Drift(length=torch.tensor(0.8), name="drift_0"),
                    cheetah.Drift(length=torch.tensor(0.9), name="drift_3"),
                ],
            ),
            cheetah.Quadrupole(length=torch.tensor(1.0), name="quad_0"),
        ]
    )

    # Filter by multiple names using a tuple
    lengths = segment.get_attr_from_every_element(
        "length", filter_name=("drift_0", "drift_1"), is_recursive=is_recursive
    )

    if is_recursive:
        assert len(lengths) == 3
        assert lengths[0] == 0.5
        assert lengths[1] == 0.6
        assert lengths[2] == 0.8
    else:
        assert len(lengths) == 2
        assert lengths[0] == 0.5
        assert lengths[1] == 0.6


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
    "attr_names",
    ["beta_x", ("beta_x",), ("s", "beta_x"), ("x", "mu_x")],
    ids=["string", "1-tuple", "2-tuple", "tensor"],
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


@pytest.mark.parametrize(
    "target_tracking_method",
    ["linear", "second_order", "drift_kick_drift", "unsupported"],
)
def test_setting_tracking_method(target_tracking_method):
    """
    Test that setting the tracking method over an entire `Segment` works as expected,
    especially that it only changes the tracking method of elements that support the
    requested tracking method.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name="d1"),
            cheetah.Quadrupole(length=torch.tensor(0.3), name="q1"),
            cheetah.Drift(length=torch.tensor(0.2), name="d2"),
            cheetah.Dipole(length=torch.tensor(0.5), name="b1"),
            cheetah.Sextupole(
                length=torch.tensor(0.4), k2=torch.tensor(0.1), name="s1"
            ),
            cheetah.Marker(name="m1"),
        ]
    )

    original_tracking_methods = {
        element.name: element.tracking_method
        for element in segment.elements
        if hasattr(element, "tracking_method")
    }

    # Set tracking method
    with pytest.warns(
        PhysicsWarning,
        match=(
            "Invalid tracking method '.+' for element .+ of type .+, supported methods "
            r"are \[.+\]. Keeping the previous tracking method .+."
        ),
    ):
        segment.set_attrs_on_every_element(tracking_method=target_tracking_method)

    # Check that elements have the target tracking iff they support it
    for element in segment.elements:
        correct_tracking_method = (
            target_tracking_method
            if target_tracking_method in element.supported_tracking_methods
            else original_tracking_methods[element.name]
        )
        assert element.tracking_method == correct_tracking_method


def test_element_names_correct():
    """
    Test that the `element_names` convenience method returns the expected element names.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name=f"drift_{i}")
            for i in range(10)
        ]
    )

    assert segment.element_names == [f"drift_{i}" for i in range(10)]


def test_element_index_correct():
    """Test that the `element_index` method returns the correct element index."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name=f"drift_{i}")
            for i in range(10)
        ]
    )

    assert segment.element_index("drift_3") == 3


def test_element_index_raises_for_none():
    """Test that the `element_index` method raises a ValueError for None."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name=f"drift_{i}")
            for i in range(10)
        ]
    )

    with pytest.raises(ValueError):
        segment.element_index("some_nonexistent_element")
