import pytest
import torch
from torch import nn

import cheetah


def test_branch_init_dict():
    """Test initialising `Branch` with a dictionary of segments."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_a")
    seg_b = cheetah.Segment([cheetah.Drift(length=torch.tensor(2.0))], name="line_b")
    branch = cheetah.Branch({"a": seg_a, "b": seg_b})

    assert len(branch) == 2
    assert branch.branch_names == ["a", "b"]
    assert branch["a"] is seg_a
    assert branch["b"] is seg_b
    assert branch.a is seg_a
    assert branch.b is seg_b


def test_branch_init_list():
    """Test initialising `Branch` with a list of segments."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_a")
    seg_b = cheetah.Segment([cheetah.Drift(length=torch.tensor(2.0))], name="line_b")
    branch = cheetah.Branch([seg_a, seg_b])

    assert len(branch) == 2
    assert branch.branch_names == ["line_a", "line_b"]
    assert branch[0] is seg_a
    assert branch[1] is seg_b
    assert branch.line_a is seg_a
    assert branch.line_b is seg_b


def test_branch_init_kwargs():
    """Test initialising `Branch` with keyword arguments."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))])
    seg_b = cheetah.Segment([cheetah.Drift(length=torch.tensor(2.0))])
    branch = cheetah.Branch(arm_1=seg_a, arm_2=seg_b)

    assert len(branch) == 2
    assert "arm_1" in branch
    assert "arm_2" in branch
    assert branch.arm_1 is seg_a
    assert branch.arm_2 is seg_b


def test_branch_init_raw_elements():
    """Test initialising `Branch` with raw element lists."""
    d1 = cheetah.Drift(length=torch.tensor(1.0))
    d2 = cheetah.Drift(length=torch.tensor(2.0))
    branch = cheetah.Branch(line_1=[d1], line_2=[d2])

    assert isinstance(branch["line_1"], cheetah.Segment)
    assert isinstance(branch["line_2"], cheetah.Segment)
    assert branch["line_1"].elements[0] is d1
    assert branch["line_2"].elements[0] is d2


def test_branch_empty_raises():
    """Test that creating a `Branch` with no branches raises `ValueError`."""
    with pytest.raises(ValueError, match="At least one branch"):
        cheetah.Branch()


def test_branch_invalid_type_raises():
    """Test that passing an invalid type for branches raises `TypeError`."""
    with pytest.raises(TypeError, match="Expected branches to be"):
        cheetah.Branch(123)


def test_branch_container_protocol():
    """Test container methods of `Branch`."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="seg_a")
    seg_b = cheetah.Segment([cheetah.Drift(length=torch.tensor(2.0))], name="seg_b")
    branch = cheetah.Branch({"a": seg_a, "b": seg_b})

    assert len(branch) == 2
    assert "a" in branch
    assert "c" not in branch
    assert list(branch.keys()) == ["a", "b"]
    assert list(branch.values()) == [seg_a, seg_b]
    assert list(branch.items()) == [("a", seg_a), ("b", seg_b)]
    assert [s for s in branch] == [seg_a, seg_b]
    assert branch[0] is seg_a
    assert branch[1] is seg_b


def test_branch_track_all_parameter_beam():
    """Test tracking a `ParameterBeam` through all branches."""
    d1 = cheetah.Drift(length=torch.tensor(1.0))
    d2 = cheetah.Drift(length=torch.tensor(3.0))
    seg_a = cheetah.Segment([d1], name="line_a")
    seg_b = cheetah.Segment([d2], name="line_b")
    branch = cheetah.Branch({"line_a": seg_a, "line_b": seg_b})

    beam = cheetah.ParameterBeam.from_parameters(
        mu_x=torch.tensor(1e-4),
        sigma_px=torch.tensor(2e-4),
    )

    out = branch.track(beam)

    assert isinstance(out, cheetah.BranchBeamMapping)
    assert len(out) == 2
    assert "line_a" in out
    assert "line_b" in out

    # Check attribute and index access on BranchBeamMapping
    assert out.line_a is out["line_a"]
    assert out[0] is out["line_a"]
    assert out[1] is out["line_b"]

    expected_a = seg_a.track(beam)
    expected_b = seg_b.track(beam)
    torch.testing.assert_close(out.line_a.s, expected_a.s)
    torch.testing.assert_close(out.line_b.s, expected_b.s)
    torch.testing.assert_close(out.line_a.mu, expected_a.mu)
    torch.testing.assert_close(out.line_b.mu, expected_b.mu)


def test_branch_track_all_particle_beam():
    """Test tracking a `ParticleBeam` through all branches."""
    d1 = cheetah.Drift(length=torch.tensor(1.0))
    d2 = cheetah.Drift(length=torch.tensor(3.0))
    seg_a = cheetah.Segment([d1], name="line_a")
    seg_b = cheetah.Segment([d2], name="line_b")
    branch = cheetah.Branch({"line_a": seg_a, "line_b": seg_b})

    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=500,
        mu_x=torch.tensor(1e-4),
        sigma_px=torch.tensor(2e-4),
    )

    out = branch.track(beam)

    assert isinstance(out, cheetah.BranchBeamMapping)
    assert len(out) == 2

    expected_a = seg_a.track(beam)
    expected_b = seg_b.track(beam)
    torch.testing.assert_close(out.line_a.particles, expected_a.particles)
    torch.testing.assert_close(out.line_b.particles, expected_b.particles)


def test_branch_active_branch_by_name():
    """Test tracking with a named active branch."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_a")
    seg_b = cheetah.Segment([cheetah.Drift(length=torch.tensor(2.0))], name="line_b")
    branch = cheetah.Branch({"line_a": seg_a, "line_b": seg_b}, active_branch="line_a")

    beam = cheetah.ParameterBeam.from_parameters()
    out = branch.track(beam)

    assert isinstance(out, cheetah.ParameterBeam)
    expected = seg_a.track(beam)
    torch.testing.assert_close(out.s, expected.s)


def test_branch_active_branch_by_index():
    """Test tracking with an index for active branch."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_a")
    seg_b = cheetah.Segment([cheetah.Drift(length=torch.tensor(2.0))], name="line_b")
    branch = cheetah.Branch([seg_a, seg_b], active_branch=1)

    beam = cheetah.ParameterBeam.from_parameters()
    out = branch.track(beam)

    assert isinstance(out, cheetah.ParameterBeam)
    expected = seg_b.track(beam)
    torch.testing.assert_close(out.s, expected.s)


def test_branch_invalid_active_branch():
    """Test that an invalid active branch raises `KeyError`."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))])
    branch = cheetah.Branch({"line_a": seg_a}, active_branch="invalid_line")

    beam = cheetah.ParameterBeam.from_parameters()
    with pytest.raises(KeyError, match="Active branch"):
        branch.track(beam)


def test_branch_split_fractions_parameter_beam():
    """Test tracking with split fractions on a `ParameterBeam`."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_a")
    seg_b = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_b")
    branch = cheetah.Branch(
        {"line_a": seg_a, "line_b": seg_b},
        split_fractions={"line_a": 0.3, "line_b": 0.7},
    )

    beam = cheetah.ParameterBeam.from_parameters(total_charge=torch.tensor(1e-9))
    out = branch.track(beam)

    torch.testing.assert_close(out.line_a.total_charge, torch.tensor(0.3e-9))
    torch.testing.assert_close(out.line_b.total_charge, torch.tensor(0.7e-9))


def test_branch_split_fractions_particle_beam():
    """Test tracking with split fractions on a `ParticleBeam`."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_a")
    seg_b = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_b")
    branch = cheetah.Branch(
        [seg_a, seg_b],
        split_fractions=[0.25, 0.75],
    )

    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=200, total_charge=torch.tensor(2e-9)
    )
    out = branch.track(beam)

    torch.testing.assert_close(out[0].total_charge, torch.tensor(0.5e-9))
    torch.testing.assert_close(out[1].total_charge, torch.tensor(1.5e-9))


def test_branch_differentiable_split_fractions():
    """Test that split fractions are differentiable with autograd."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_a")
    seg_b = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_b")

    frac_a = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
    frac_b = nn.Parameter(torch.tensor(0.6, dtype=torch.float64))

    branch = cheetah.Branch(
        {"line_a": seg_a, "line_b": seg_b},
        split_fractions={"line_a": frac_a, "line_b": frac_b},
    ).to(torch.float64)

    beam = cheetah.ParameterBeam.from_parameters(
        total_charge=torch.tensor(1e-9, dtype=torch.float64)
    ).to(torch.float64)

    out = branch.track(beam)
    loss = (out.line_a.total_charge - 0.5e-9) ** 2
    loss.backward()

    assert frac_a.grad is not None
    assert frac_a.grad != 0.0


def test_branch_differentiable_magnet_settings():
    """Test that magnet parameters in branch segments track gradients."""
    quad = cheetah.Quadrupole(
        length=torch.tensor(0.2), k1=nn.Parameter(torch.tensor(2.0))
    )
    seg = cheetah.Segment([quad])
    branch = cheetah.Branch({"arm": seg})

    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=100, mu_x=torch.tensor(1e-4)
    )
    out = branch.track(beam)

    loss = out.arm.particles[..., 0].sum()
    loss.backward()

    assert quad.k1.grad is not None


def test_branch_device_and_dtype():
    """Test converting `Branch` to different data types."""
    seg_a = cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))], name="line_a")
    branch = cheetah.Branch({"line_a": seg_a}, split_fractions={"line_a": 0.5}).to(
        torch.float64
    )

    assert branch.length.dtype == torch.float64
    assert branch.split_fractions["line_a"].dtype == torch.float64
    assert branch["line_a"].elements[0].length.dtype == torch.float64


def test_branch_in_segment_all_branches():
    """Test embedding a `Branch` element inside a `Segment` tracking all branches."""
    up_drift = cheetah.Drift(length=torch.tensor(1.0))
    down_drift = cheetah.Drift(length=torch.tensor(2.0))
    branch = cheetah.Branch(
        arm_a=[cheetah.Drift(length=torch.tensor(0.5))],
        arm_b=[cheetah.Drift(length=torch.tensor(1.5))],
    )

    # Segment with common downstream element
    full_line = cheetah.Segment([up_drift, branch, down_drift])
    beam = cheetah.ParameterBeam.from_parameters()

    out = full_line.track(beam)

    assert isinstance(out, cheetah.BranchBeamMapping)
    assert len(out) == 2
    # Branch a: 1.0 (up) + 0.5 (arm_a) + 2.0 (down) = 3.5
    torch.testing.assert_close(out.arm_a.s, torch.tensor(3.5))
    # Branch b: 1.0 (up) + 1.5 (arm_b) + 2.0 (down) = 4.5
    torch.testing.assert_close(out.arm_b.s, torch.tensor(4.5))


def test_branch_in_segment_active_branch():
    """Test embedding a `Branch` with active branch inside a `Segment`."""
    up_drift = cheetah.Drift(length=torch.tensor(1.0))
    down_drift = cheetah.Drift(length=torch.tensor(2.0))
    branch = cheetah.Branch(
        arm_a=[cheetah.Drift(length=torch.tensor(0.5))],
        arm_b=[cheetah.Drift(length=torch.tensor(1.5))],
        active_branch="arm_a",
    )

    full_line = cheetah.Segment([up_drift, branch, down_drift])
    beam = cheetah.ParameterBeam.from_parameters()

    out = full_line.track(beam)

    assert isinstance(out, cheetah.ParameterBeam)
    torch.testing.assert_close(out.s, torch.tensor(3.5))


def test_branch_flattened():
    """Test that `Branch.flattened()` flattens internal branch segments."""
    nested_seg = cheetah.Segment(
        [
            cheetah.Segment([cheetah.Drift(length=torch.tensor(1.0))]),
            cheetah.Segment([cheetah.Drift(length=torch.tensor(2.0))]),
        ]
    )
    branch = cheetah.Branch({"nested": nested_seg})
    flat = branch.flattened()

    assert len(flat["nested"].elements) == 2


def test_branch_clone():
    """Test that `Branch.clone()` creates an independent copy."""
    d = cheetah.Drift(length=torch.tensor(1.0))
    branch = cheetah.Branch({"line": [d]}, split_fractions={"line": 0.8})
    cloned = branch.clone()

    assert cloned is not branch
    assert cloned["line"] is not branch["line"]
    assert cloned.split_fractions["line"] is not branch.split_fractions["line"]
    torch.testing.assert_close(
        cloned.split_fractions["line"], branch.split_fractions["line"]
    )


def test_branch_beam_mapping_attributes():
    """Test attribute and index access on `BranchBeamMapping`."""
    mapping = cheetah.BranchBeamMapping()
    beam1 = cheetah.ParameterBeam.from_parameters(s=torch.tensor(1.0))
    beam2 = cheetah.ParameterBeam.from_parameters(s=torch.tensor(2.0))

    mapping["first"] = beam1
    mapping.second = beam2

    assert mapping.first is beam1
    assert mapping["second"] is beam2
    assert mapping[0] is beam1
    assert mapping[1] is beam2

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = mapping.nonexistent
