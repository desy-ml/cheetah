import pytest
import torch

import cheetah


def test_merged_transfer_maps_tracking():
    """
    Test that tracking through merged transfer maps results in the same beam as the
    original segment did.
    """
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6)),
            cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(4.2)),
            cheetah.Drift(length=torch.tensor(0.4)),
            cheetah.HorizontalCorrector(
                length=torch.tensor(0.1), angle=torch.tensor(1e-4)
            ),
        ]
    )
    merged_segment = original_segment.transfer_maps_merged(incoming_beam=incoming_beam)

    original_beam = original_segment.track(incoming_beam)
    merged_beam = merged_segment.track(incoming_beam)

    assert torch.isclose(original_beam.mu_x, merged_beam.mu_x)
    assert torch.isclose(original_beam.mu_px, merged_beam.mu_px)
    assert torch.isclose(original_beam.mu_y, merged_beam.mu_y)
    assert torch.isclose(original_beam.mu_py, merged_beam.mu_py)
    assert torch.isclose(original_beam.sigma_x, merged_beam.sigma_x)
    assert torch.isclose(original_beam.sigma_px, merged_beam.sigma_px)
    assert torch.isclose(original_beam.sigma_y, merged_beam.sigma_y)
    assert torch.isclose(original_beam.sigma_py, merged_beam.sigma_py)
    assert torch.isclose(original_beam.sigma_tau, merged_beam.sigma_tau)
    assert torch.isclose(original_beam.sigma_p, merged_beam.sigma_p)
    assert torch.isclose(original_beam.energy, merged_beam.energy)
    assert torch.isclose(original_beam.total_charge, merged_beam.total_charge)


def test_merged_transfer_maps_tracking_vectorized():
    """
    Test that tracking through merged transfer maps results in the same beam as the
    original segment did in a vectorised setup.
    """
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6)),
            cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(4.2)),
            cheetah.Drift(length=torch.linspace(0.3, 0.5, 10)),
            cheetah.HorizontalCorrector(
                length=torch.tensor(0.1), angle=torch.tensor(1e-4)
            ),
        ]
    )
    merged_segment = original_segment.transfer_maps_merged(incoming_beam=incoming_beam)

    original_beam = original_segment.track(incoming_beam)
    merged_beam = merged_segment.track(incoming_beam)

    assert torch.allclose(original_beam.mu_x, merged_beam.mu_x)
    assert torch.allclose(original_beam.mu_px, merged_beam.mu_px)
    assert torch.allclose(original_beam.mu_y, merged_beam.mu_y)
    assert torch.allclose(original_beam.mu_py, merged_beam.mu_py)
    assert torch.allclose(original_beam.sigma_x, merged_beam.sigma_x)
    assert torch.allclose(original_beam.sigma_px, merged_beam.sigma_px)
    assert torch.allclose(original_beam.sigma_y, merged_beam.sigma_y)
    assert torch.allclose(original_beam.sigma_py, merged_beam.sigma_py)
    assert torch.allclose(original_beam.sigma_tau, merged_beam.sigma_tau)
    assert torch.allclose(original_beam.sigma_p, merged_beam.sigma_p)
    assert torch.allclose(original_beam.energy, merged_beam.energy)
    assert torch.allclose(original_beam.total_charge, merged_beam.total_charge)


def test_merged_transfer_maps_num_elements():
    """
    Test that merging transfer maps results in the correct number of elements.
    """
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6)),
            cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(4.2)),
            cheetah.Drift(length=torch.tensor(0.4)),
            cheetah.HorizontalCorrector(
                length=torch.tensor(0.1), angle=torch.tensor(1e-4)
            ),
        ]
    )
    merged_segment = original_segment.transfer_maps_merged(incoming_beam=incoming_beam)

    assert len(merged_segment.elements) < len(original_segment.elements)
    assert len(merged_segment.elements) == 1


def test_no_markers_left_after_removal():
    """
    Test that when removing markers, no markers are left in the segment.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6)),
            cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(4.2)),
            cheetah.Marker(),
            cheetah.Drift(length=torch.tensor(0.4)),
            cheetah.HorizontalCorrector(
                length=torch.tensor(0.1), angle=torch.tensor(1e-4)
            ),
            cheetah.Marker(),
        ]
    )
    optimized_segment = segment.without_inactive_markers()

    assert not any(
        isinstance(element, cheetah.Marker) for element in optimized_segment.elements
    )


def test_inactive_magnet_is_replaced_by_drift():
    """
    Test that an inactive magnet is replaced by a drift as expected.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6)),
            cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(0.0)),
            cheetah.Drift(length=torch.tensor(0.4)),
        ]
    )

    optimized_segment = segment.inactive_elements_as_drifts()

    assert all(
        isinstance(element, cheetah.Drift) for element in optimized_segment.elements
    )
    assert torch.allclose(segment.length, optimized_segment.length)


def test_active_elements_not_replaced_by_drift():
    """
    Test that an active magnet is not replaced by a drift.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6)),
            cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor([4.2, 0.0])),
            cheetah.Drift(length=torch.tensor(0.4)),
        ]
    )

    optimized_segment = segment.inactive_elements_as_drifts()

    assert isinstance(optimized_segment.elements[1], cheetah.Quadrupole)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inactive_magnet_drift_replacement_dtype(dtype: torch.dtype):
    """
    Test that when an inactive magnet is replaced by a drift, the drift has the same
    dtype as the original element.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6), dtype=dtype),
            cheetah.Quadrupole(
                length=torch.tensor(0.2), k1=torch.tensor(0.0), dtype=dtype
            ),
            cheetah.Drift(length=torch.tensor(0.4), dtype=dtype),
        ]
    )

    optimized_segment = segment.inactive_elements_as_drifts()

    assert all(element.length.dtype == dtype for element in optimized_segment.elements)


def test_inactive_magnet_drift_except_for():
    """
    Test that an inactive magnet is not replaced by a drift when it is included in the
    list of exceptions.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6)),
            cheetah.Quadrupole(
                length=torch.tensor(0.2), k1=torch.tensor(0.0), name="my_quad"
            ),
            cheetah.Drift(length=torch.tensor(0.4)),
        ]
    )

    optimized_segment = segment.inactive_elements_as_drifts(except_for=["my_quad"])

    assert isinstance(optimized_segment.elements[1], cheetah.Quadrupole)


def test_skippable_elements_reset():
    """
    @cr-xu pointed out that the skippable elements are not always reset appropriately
    when merging transfer maps (see #88). This test catches the bug he pointed out.
    """
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6)),
            cheetah.Quadrupole(
                length=torch.tensor(0.2), k1=torch.tensor(4.2), name="Q1"
            ),
            cheetah.Drift(length=torch.tensor(0.4)),
            cheetah.HorizontalCorrector(
                length=torch.tensor(0.1), angle=torch.tensor(1e-4), name="HCOR_1"
            ),
            cheetah.Drift(length=torch.tensor(0.4)),
        ]
    )

    merged_segment = original_segment.transfer_maps_merged(
        incoming_beam=incoming_beam, except_for=["Q1", "HCOR_1"]
    )

    original_tm = original_segment.elements[2].transfer_map(
        energy=incoming_beam.energy, species=incoming_beam.species
    )
    merged_tm = merged_segment.elements[2].transfer_map(
        energy=incoming_beam.energy, species=incoming_beam.species
    )

    assert torch.allclose(original_tm, merged_tm)


def test_without_zero_length_elements():
    """Test that zero-length elements are properly recognized and removed."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Dipole(length=torch.tensor(0.0), angle=torch.tensor(0.0)),
            cheetah.Dipole(
                length=torch.tensor(0.0), angle=torch.tensor(0.0), name="my_dipole"
            ),
            cheetah.Dipole(length=torch.tensor([0.0, 0.1]), angle=torch.tensor(0.0)),
            cheetah.Drift(length=torch.tensor(0.0)),
            cheetah.Dipole(length=torch.tensor(0.0), angle=torch.tensor([0.5, 0.0])),
        ]
    )

    pruned = segment.without_inactive_zero_length_elements()
    pruned_except = segment.without_inactive_zero_length_elements(
        except_for=["my_dipole"]
    )

    assert len(segment.elements) == 6
    assert len(pruned.elements) == 3
    assert len(pruned_except.elements) == 4
    assert torch.allclose(segment.length, pruned.length)
    assert torch.allclose(segment.length, pruned_except.length)
