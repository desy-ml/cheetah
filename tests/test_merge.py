import torch

import cheetah


def test_merge_element_names():
    # Common prefix with short enumeration suffixes
    assert cheetah.merge_element_names("drift_1", "drift_2") == "drift"
    assert cheetah.merge_element_names("quad_1", "quad_2", "quad_3") == "quad"
    assert cheetah.merge_element_names("q1", "q2") == "q"

    # Common prefix with in/out suffixes
    assert cheetah.merge_element_names("quad_in", "quad_out") == "quad"

    # Single element name
    assert cheetah.merge_element_names("drift_1") == "drift_1"

    # No common prefix -> concatenation
    assert cheetah.merge_element_names("drift_1", "quad_1") == "drift_1_quad_1"

    # Suffix after clean prefix > 5 chars -> concatenation
    assert (
        cheetah.merge_element_names("section_part_a_module", "section_part_b_module")
        == "section_part_a_module_section_part_b_module"
    )


def test_drift_merge():
    d1 = cheetah.Drift(length=torch.tensor(0.5), name="drift_1")
    d2 = cheetah.Drift(length=torch.tensor(0.3), name="drift_2")

    merged = d1.merge(d2)
    assert merged is not None
    assert isinstance(merged, cheetah.Drift)
    assert torch.isclose(merged.length, torch.tensor(0.8))
    assert merged.name == "drift"

    # Mismatched tracking method
    d3 = cheetah.Drift(length=torch.tensor(0.3), tracking_method="drift_kick_drift")
    assert d1.merge(d3) is None

    # Mismatched element type
    q1 = cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(1.0))
    assert d1.merge(q1) is None


def test_quadrupole_merge():
    q1 = cheetah.Quadrupole(
        length=torch.tensor(0.2), k1=torch.tensor(4.2), name="quad_1"
    )
    q2 = cheetah.Quadrupole(
        length=torch.tensor(0.3), k1=torch.tensor(4.2), name="quad_2"
    )

    merged = q1.merge(q2)
    assert merged is not None
    assert isinstance(merged, cheetah.Quadrupole)
    assert torch.isclose(merged.length, torch.tensor(0.5))
    assert torch.equal(merged.k1, q1.k1)
    assert merged.name == "quad"

    # Mismatched k1
    q3 = cheetah.Quadrupole(length=torch.tensor(0.3), k1=torch.tensor(2.0))
    assert q1.merge(q3) is None


def test_solenoid_merge():
    s1 = cheetah.Solenoid(length=torch.tensor(0.4), k=torch.tensor(2.5), name="sol_in")
    s2 = cheetah.Solenoid(length=torch.tensor(0.4), k=torch.tensor(2.5), name="sol_out")

    merged = s1.merge(s2)
    assert merged is not None
    assert isinstance(merged, cheetah.Solenoid)
    assert torch.isclose(merged.length, torch.tensor(0.8))
    assert merged.name == "sol"

    # Mismatched k
    s3 = cheetah.Solenoid(length=torch.tensor(0.4), k=torch.tensor(1.0))
    assert s1.merge(s3) is None


def test_sextupole_merge():
    s1 = cheetah.Sextupole(
        length=torch.tensor(0.1), k2=torch.tensor(10.0), name="sex_1"
    )
    s2 = cheetah.Sextupole(
        length=torch.tensor(0.2), k2=torch.tensor(10.0), name="sex_2"
    )

    merged = s1.merge(s2)
    assert merged is not None
    assert isinstance(merged, cheetah.Sextupole)
    assert torch.isclose(merged.length, torch.tensor(0.3))
    assert merged.name == "sex"


def test_correctors_merge():
    # Thin correctors (length=0) merge angles
    hc1 = cheetah.HorizontalCorrector(
        length=torch.tensor(0.0), angle=torch.tensor(1e-4), name="hcor_1"
    )
    hc2 = cheetah.HorizontalCorrector(
        length=torch.tensor(0.0), angle=torch.tensor(2e-4), name="hcor_2"
    )
    merged_hc = hc1.merge(hc2)
    assert merged_hc is not None
    assert torch.isclose(merged_hc.length, torch.tensor(0.0))
    assert torch.isclose(merged_hc.angle, torch.tensor(3e-4))
    assert merged_hc.name == "hcor"

    # Zero angle correctors merge lengths
    vc1 = cheetah.VerticalCorrector(
        length=torch.tensor(0.1), angle=torch.tensor(0.0), name="vcor_1"
    )
    vc2 = cheetah.VerticalCorrector(
        length=torch.tensor(0.2), angle=torch.tensor(0.0), name="vcor_2"
    )
    merged_vc = vc1.merge(vc2)
    assert merged_vc is not None
    assert torch.isclose(merged_vc.length, torch.tensor(0.3))
    assert torch.isclose(merged_vc.angle, torch.tensor(0.0))
    assert merged_vc.name == "vcor"

    # Thick active correctors should not merge to preserve kick location
    hc_thick1 = cheetah.HorizontalCorrector(
        length=torch.tensor(0.1), angle=torch.tensor(1e-4)
    )
    hc_thick2 = cheetah.HorizontalCorrector(
        length=torch.tensor(0.1), angle=torch.tensor(1e-4)
    )
    assert hc_thick1.merge(hc_thick2) is None


def test_marker_merge():
    m1 = cheetah.Marker(name="m_1")
    m2 = cheetah.Marker(name="m_2")
    merged = m1.merge(m2)
    assert merged is not None
    assert isinstance(merged, cheetah.Marker)
    assert merged.name == "m"


def test_segment_merge():
    seg1 = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(2.0)),
        ],
        name="seg_1",
    )
    seg2 = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.3)),
        ],
        name="seg_2",
    )

    merged = seg1.merge(seg2)
    assert merged is not None
    assert isinstance(merged, cheetah.Segment)
    assert len(merged.elements) == 3
    assert merged.name == "seg"


def test_unmergeable_elements_default():
    dipole1 = cheetah.Dipole(length=torch.tensor(0.5), angle=torch.tensor(0.1))
    dipole2 = cheetah.Dipole(length=torch.tensor(0.5), angle=torch.tensor(0.1))
    # Dipole currently inherits default Element.merge which returns None
    assert dipole1.merge(dipole2) is None


def test_segment_with_consecutive_elements_merged():
    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name="d_1"),
            cheetah.Drift(length=torch.tensor(0.3), name="d_2"),
            cheetah.Quadrupole(
                length=torch.tensor(0.2), k1=torch.tensor(4.2), name="q_1"
            ),
            cheetah.Quadrupole(
                length=torch.tensor(0.1), k1=torch.tensor(4.2), name="q_2"
            ),
            cheetah.Drift(length=torch.tensor(0.4), name="d_3"),
        ]
    )

    merged_segment = original_segment.with_consecutive_elements_merged()
    assert len(merged_segment.elements) == 3
    assert isinstance(merged_segment.elements[0], cheetah.Drift)
    assert torch.isclose(merged_segment.elements[0].length, torch.tensor(0.8))
    assert merged_segment.elements[0].name == "d"

    assert isinstance(merged_segment.elements[1], cheetah.Quadrupole)
    assert torch.isclose(merged_segment.elements[1].length, torch.tensor(0.3))
    assert merged_segment.elements[1].name == "q"

    assert isinstance(merged_segment.elements[2], cheetah.Drift)
    assert torch.isclose(merged_segment.elements[2].length, torch.tensor(0.4))
    assert merged_segment.elements[2].name == "d_3"


def test_segment_with_consecutive_elements_merged_except_for():
    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name="d_1"),
            cheetah.Drift(length=torch.tensor(0.3), name="d_2"),
            cheetah.Quadrupole(
                length=torch.tensor(0.2), k1=torch.tensor(4.2), name="q_1"
            ),
            cheetah.Quadrupole(
                length=torch.tensor(0.1), k1=torch.tensor(4.2), name="q_2"
            ),
        ]
    )

    merged_segment = original_segment.with_consecutive_elements_merged(
        except_for=["d_2"]
    )
    assert len(merged_segment.elements) == 3
    assert merged_segment.elements[0].name == "d_1"
    assert merged_segment.elements[1].name == "d_2"
    assert merged_segment.elements[2].name == "q"


def test_merged_elements_tracking_equivalence():
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.3)),
            cheetah.Drift(length=torch.tensor(0.3)),
            cheetah.Quadrupole(length=torch.tensor(0.1), k1=torch.tensor(4.2)),
            cheetah.Quadrupole(length=torch.tensor(0.1), k1=torch.tensor(4.2)),
            cheetah.Drift(length=torch.tensor(0.2)),
            cheetah.Drift(length=torch.tensor(0.2)),
            cheetah.Solenoid(length=torch.tensor(0.2), k=torch.tensor(1.5)),
            cheetah.Solenoid(length=torch.tensor(0.2), k=torch.tensor(1.5)),
        ]
    )

    merged_segment = original_segment.with_consecutive_elements_merged()

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
