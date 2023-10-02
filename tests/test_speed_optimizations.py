import numpy as np
import torch

import cheetah


def test_merged_transfer_maps_tracking():
    """
    Test that tracking through merged transfer maps results in the same beam as the
    original segment did.
    """
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
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

    assert np.isclose(original_beam.mu_x, merged_beam.mu_x)
    assert np.isclose(original_beam.mu_xp, merged_beam.mu_xp)
    assert np.isclose(original_beam.mu_y, merged_beam.mu_y)
    assert np.isclose(original_beam.mu_yp, merged_beam.mu_yp)
    assert np.isclose(original_beam.sigma_x, merged_beam.sigma_x)
    assert np.isclose(original_beam.sigma_xp, merged_beam.sigma_xp)
    assert np.isclose(original_beam.sigma_y, merged_beam.sigma_y)
    assert np.isclose(original_beam.sigma_yp, merged_beam.sigma_yp)
    assert np.isclose(original_beam.sigma_s, merged_beam.sigma_s)
    assert np.isclose(original_beam.sigma_p, merged_beam.sigma_p)
    assert np.isclose(original_beam.energy, merged_beam.energy)
    assert np.isclose(original_beam.total_charge, merged_beam.total_charge)


def test_merged_transfer_maps_num_elements():
    """
    Test that merging transfer maps results in the correct number of elements.
    """
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
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


def test_active_elements_not_replaced_by_drift():
    """
    Test that an active magnet is not replaced by a drift.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.6)),
            cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(4.2)),
            cheetah.Drift(length=torch.tensor(0.4)),
        ]
    )

    optimized_segment = segment.inactive_elements_as_drifts()

    assert isinstance(optimized_segment.elements[1], cheetah.Quadrupole)


def test_skippable_elements_reset():
    """
    @cr-xu pointed out that the skippable elements are not always reset appropriately
    when merging transfer maps (see #88). This test catches the bug he pointed out.
    """
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
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

    original_tm = original_segment.elements[2].transfer_map(energy=incoming_beam.energy)
    merged_tm = merged_segment.elements[2].transfer_map(energy=incoming_beam.energy)

    assert np.allclose(original_tm, merged_tm)
