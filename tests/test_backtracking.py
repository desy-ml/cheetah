import warnings

import pytest
import torch

import cheetah
from cheetah.utils import invert_affine_map, is_mps_available_and_functional
from cheetah.utils.warnings import PhysicsWarning

ELEMENTS = [
    "Drift",
    "Quadrupole",
    "Sextupole",
    "Solenoid",
    "Undulator",
    "HorizontalCorrector",
    "VerticalCorrector",
    "CombinedCorrector",
    "Dipole",
    "RBend",
    "CustomTransferMap",
]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not is_mps_available_and_functional(), reason="MPS unavailable"
        ),
    ),
]


def make_element(name, dtype=torch.float64, batched=False, device="cpu"):
    """A simple helper to instantiate an element with a few non-default parameters."""
    kwargs = dict(
        length=torch.tensor(
            [[0.3], [0.6]] if batched else 0.3, dtype=dtype, device=device
        )
    )
    if name == "CustomTransferMap":
        tm = torch.eye(7, dtype=dtype, device=device)
        tm[0, 0] = 1.1
        tm[0, 1] = 0.2
        tm[2, 3] = 0.3
        tm[:6, 6] = torch.tensor(
            [2e-4, -3e-4, 1e-4, 2e-4, -1e-4, 1e-5], dtype=dtype, device=device
        )
        kwargs["predefined_transfer_map"] = tm
    elif name == "Quadrupole":
        kwargs.update(
            k1=torch.tensor(0.7, dtype=dtype, device=device),
            tilt=torch.tensor(0.2, dtype=dtype, device=device),
            misalignment=torch.tensor([2e-4, -3e-4], dtype=dtype, device=device),
        )
    elif name == "Sextupole":
        kwargs.update(
            k2=torch.tensor(2.0, dtype=dtype, device=device), tracking_method="linear"
        )
    elif name == "Solenoid":
        kwargs.update(
            k=torch.tensor(0.4, dtype=dtype, device=device),
            misalignment=torch.tensor([2e-4, -3e-4], dtype=dtype, device=device),
        )
    elif name == "Undulator":
        kwargs.update(
            period=torch.tensor(0.05, dtype=dtype, device=device),
            kx=torch.tensor(0.8, dtype=dtype, device=device),
            ky=torch.tensor(0.5, dtype=dtype, device=device),
        )
    elif name in ("HorizontalCorrector", "VerticalCorrector"):
        kwargs.update(angle=torch.tensor(2e-4, dtype=dtype, device=device))
    elif name == "CombinedCorrector":
        kwargs.update(
            horizontal_angle=torch.tensor(2e-4, dtype=dtype, device=device),
            vertical_angle=torch.tensor(-3e-4, dtype=dtype, device=device),
        )
    elif name in ("Dipole", "RBend"):
        prefix = "dipole" if name == "Dipole" else "rbend"
        kwargs.update(
            angle=torch.tensor(0.03, dtype=dtype, device=device),
            k1=torch.tensor(0.1, dtype=dtype, device=device),
            tilt=torch.tensor(0.2, dtype=dtype, device=device),
            gap=torch.tensor(0.02, dtype=dtype, device=device),
            fringe_integral=torch.tensor(0.4, dtype=dtype, device=device),
            fringe_integral_exit=torch.tensor(0.6, dtype=dtype, device=device),
        )
        kwargs.update(
            {
                f"{prefix}_e1": torch.tensor(0.04, dtype=dtype, device=device),
                f"{prefix}_e2": torch.tensor(-0.02, dtype=dtype, device=device),
            }
        )
    return getattr(cheetah, name)(**kwargs, dtype=dtype, device=device)


def make_beam(beam_type, dtype=torch.float64, batched=False, device="cpu"):
    """A simple helper to instantiate a beam with a few non-default parameters."""
    kwargs = dict(
        energy=torch.tensor(
            [8e7, 1e8, 1.2e8] if batched else 1e8, dtype=dtype, device=device
        ),
        mu_x=torch.tensor(3e-4, dtype=dtype, device=device),
        mu_y=torch.tensor(-2e-4, dtype=dtype, device=device),
        mu_px=torch.tensor(1e-4, dtype=dtype, device=device),
        mu_py=torch.tensor(-1e-4, dtype=dtype, device=device),
        s=torch.tensor(1.25, dtype=dtype, device=device),
        dtype=dtype,
        device=device,
    )
    if beam_type is cheetah.ParticleBeam:
        kwargs["num_particles"] = 32
    return beam_type.from_parameters(**kwargs)


@pytest.mark.parametrize("name", ELEMENTS)
@pytest.mark.parametrize("beam_type", [cheetah.ParticleBeam, cheetah.ParameterBeam])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_element_round_trip(name, beam_type, dtype, batched, device):
    """Test that tracking followed by backtracking recovers the original beam."""
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64")
    element = make_element(name, dtype, batched, device)
    beam = make_beam(beam_type, dtype, batched, device)
    original = beam.clone()
    length = element.length.clone()
    assert element.supports_backtracking
    non_module_features = [
        feature for feature in beam.defining_features if feature != "species"
    ]
    eps = torch.finfo(dtype).eps
    for recovered in (
        element.backtrack(element.track(beam)),
        element.track(element.backtrack(beam)),
    ):
        assert recovered.species.name == beam.species.name
        assert torch.equal(recovered.species.mass_eV, beam.species.mass_eV)
        assert torch.equal(
            recovered.species.num_elementary_charges,
            beam.species.num_elementary_charges,
        )
        for feature in non_module_features:
            actual, expected = getattr(recovered, feature), getattr(beam, feature)
            if feature in ("particles", "mu", "cov", "s"):
                # Exclude the homogeneous coordinate from the tolerance scale.
                scale = (
                    expected[..., :6] if feature in ("particles", "mu") else expected
                )
                assert torch.allclose(
                    actual,
                    expected,
                    rtol=128 * eps,
                    atol=128 * eps * scale.abs().max().item(),
                )
            else:
                assert torch.allclose(actual, expected, rtol=0.0, atol=0.0)
    for feature in non_module_features:
        assert torch.equal(getattr(beam, feature), getattr(original, feature))
    assert beam.species.name == original.species.name
    assert torch.equal(beam.species.mass_eV, original.species.mass_eV)
    assert torch.equal(
        beam.species.num_elementary_charges, original.species.num_elementary_charges
    )
    assert torch.allclose(element.length, length, rtol=0.0, atol=0.0)
    assert element.tracking_method == "linear"


@pytest.mark.parametrize("name", ELEMENTS)
def test_inverse_maps(name):
    """
    Test that the inverse first-order transfer map is indeed the inverse of the forward
    map.
    """

    element = make_element(name, batched=True)
    beam = make_beam(cheetah.ParameterBeam, batched=True)
    tm = element.first_order_transfer_map(beam.energy, beam.species)
    inverse = element.inverse_first_order_transfer_map(beam.energy, beam.species)
    identity = torch.eye(7, dtype=tm.dtype).expand_as(tm)
    assert torch.allclose(inverse @ tm, identity, rtol=1e-13, atol=1e-14)
    assert torch.allclose(tm @ inverse, identity, rtol=1e-13, atol=1e-14)
    assert torch.equal(inverse[..., 6, :], identity[..., 6, :])


def test_affine_inversion():
    """Test that the inversion preserves the affine structure of the transfer map."""
    tm = torch.eye(7, dtype=torch.float64).repeat(2, 3, 1, 1)
    tm[..., :6, :6] += 0.1 * torch.randn(2, 3, 6, 6, dtype=tm.dtype)
    tm[..., :6, 6] = torch.randn(2, 3, 6, dtype=tm.dtype)
    inverse = invert_affine_map(tm)
    identity = torch.eye(7, dtype=tm.dtype).expand_as(tm)
    assert torch.allclose(inverse @ tm, identity, rtol=1e-13, atol=1e-14)
    assert torch.allclose(tm @ inverse, identity, rtol=1e-13, atol=1e-14)
    assert torch.equal(inverse[..., 6, :], identity[..., 6, :])
    # Only the upper rows are free variables in an affine map.
    upper = tm[..., :6, :].clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda rows: invert_affine_map(torch.cat((rows, tm[..., 6:7, :]), dim=-2)),
        (upper,),
    )
    with pytest.raises(torch.linalg.LinAlgError):
        invert_affine_map(torch.zeros(7, 7, dtype=tm.dtype))


@pytest.mark.parametrize(
    "name",
    [
        "HorizontalCorrector",
        "VerticalCorrector",
        "CombinedCorrector",
        "Dipole",
        "RBend",
    ],
)
def test_analytic_inverse_without_matrix_inversion(name, monkeypatch):
    element = make_element(name, batched=True)
    beam = make_beam(cheetah.ParameterBeam, batched=True)
    forward = element.first_order_transfer_map(beam.energy, beam.species).clone()

    def unexpected(*args, **kwargs):
        raise AssertionError("Analytic backtracking must not invert or solve a matrix")

    monkeypatch.setattr(torch.linalg, "inv", unexpected)
    monkeypatch.setattr(torch.linalg, "solve", unexpected)
    inverse = element.inverse_first_order_transfer_map(beam.energy, beam.species)
    identity = torch.eye(7, dtype=forward.dtype).expand_as(forward)
    assert torch.allclose(inverse @ forward, identity, rtol=1e-13, atol=1e-14)
    assert torch.allclose(forward @ inverse, identity, rtol=1e-13, atol=1e-14)
    assert torch.equal(
        element.first_order_transfer_map(beam.energy, beam.species), forward
    )


@pytest.mark.parametrize("beam_type", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_nested_segment_and_superimposed(beam_type):
    bpm = cheetah.BPM(is_active=True, dtype=torch.float64)
    wrapped = cheetah.Superimposed(make_element("Drift"), bpm)
    segment = cheetah.Segment(
        [
            make_element("Quadrupole"),
            cheetah.Segment([make_element("CombinedCorrector"), wrapped]),
            make_element("RBend"),
        ]
    )
    beam = make_beam(beam_type)
    assert segment.supports_backtracking
    forward = segment.track(beam)
    reading = bpm.reading.clone()
    actual = segment.backtrack(forward)
    expected = beam
    assert actual.species.name == expected.species.name
    non_module_features = [
        feature for feature in expected.defining_features if feature != "species"
    ]
    for feature in non_module_features:
        assert torch.allclose(
            getattr(actual, feature),
            getattr(expected, feature),
            rtol=1e-13,
            atol=1e-14,
        )
    assert torch.allclose(bpm.reading, reading, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("beam_type", [cheetah.ParticleBeam, cheetah.ParameterBeam])
@pytest.mark.parametrize("name", ["Marker", "BPM", "Screen"])
def test_diagnostic_passthrough(name, beam_type):
    """Test that diagnostic elements pass the beam through and update their readings"""
    kwargs = {} if name == "Marker" else {"is_active": True}
    diagnostic = getattr(cheetah, name)(**kwargs, dtype=torch.float64)
    beam = make_beam(beam_type)
    assert diagnostic.supports_backtracking
    actual = diagnostic.backtrack(beam)
    expected = beam
    assert actual.species.name == expected.species.name
    non_module_features = [
        feature for feature in expected.defining_features if feature != "species"
    ]
    for feature in non_module_features:
        assert torch.allclose(
            getattr(actual, feature),
            getattr(expected, feature),
            rtol=0.0,
            atol=0.0,
        )
    if name == "BPM":
        assert torch.allclose(diagnostic.reading, torch.stack((beam.mu_x, beam.mu_y)))
    elif name == "Screen":
        actual = diagnostic.get_read_beam()
        expected = beam
        assert actual.species.name == expected.species.name
        non_module_features = [
            feature for feature in expected.defining_features if feature != "species"
        ]
        for feature in non_module_features:
            assert torch.allclose(
                getattr(actual, feature),
                getattr(expected, feature),
                rtol=0.0,
                atol=0.0,
            )


@pytest.mark.parametrize("beam_type", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_blocking_screen_warns_without_restoring_losses(beam_type):
    screen = cheetah.Screen(is_active=True, is_blocking=True, dtype=torch.float64)
    beam = make_beam(beam_type)
    blocked = screen.track(beam)
    assert screen.supports_backtracking
    with pytest.warns(PhysicsWarning, match="without restoring lost charge"):
        recovered = screen.backtrack(blocked)
    actual = recovered
    expected = blocked
    assert actual.species.name == expected.species.name
    non_module_features = [
        feature for feature in expected.defining_features if feature != "species"
    ]
    for feature in non_module_features:
        assert torch.allclose(
            getattr(actual, feature),
            getattr(expected, feature),
            rtol=0.0,
            atol=0.0,
        )
    screen.is_active = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = screen.backtrack(beam)
        expected = beam
        assert actual.species.name == expected.species.name
        non_module_features = [
            feature for feature in expected.defining_features if feature != "species"
        ]
        for feature in non_module_features:
            assert torch.allclose(
                getattr(actual, feature),
                getattr(expected, feature),
                rtol=0.0,
                atol=0.0,
            )


@pytest.mark.parametrize(
    "name", ["Drift", "Quadrupole", "Sextupole", "Dipole", "RBend"]
)
@pytest.mark.parametrize("method", ["second_order", "drift_kick_drift"])
def test_unsupported_tracking_method(name, method):
    element = make_element(name)
    if method not in element.supported_tracking_methods:
        pytest.skip("Method not available on this element")
    element.tracking_method = method
    assert not element.supports_backtracking
    with pytest.raises(NotImplementedError, match=method):
        element.backtrack(make_beam(cheetah.ParameterBeam))
    assert element.tracking_method == method


@pytest.mark.parametrize(
    "element",
    [
        cheetah.Cavity(length=torch.tensor(0.3)),
        cheetah.TransverseDeflectingCavity(length=torch.tensor(0.3)),
        cheetah.Aperture(is_active=True),
        cheetah.Aperture(is_active=False),
        cheetah.SpaceChargeKick(effect_length=torch.tensor(0.3)),
    ],
)
def test_unsupported_elements(element):
    assert not element.supports_backtracking
    with pytest.raises(NotImplementedError, match=element.name):
        element.backtrack(make_beam(cheetah.ParameterBeam))
    beam = make_beam(cheetah.ParameterBeam)
    with pytest.raises(NotImplementedError):
        element.inverse_first_order_transfer_map(beam.energy, beam.species)


def test_segment_checks_entire_subtree_before_propagating(monkeypatch):
    last = make_element("Drift")

    def unexpected(_):
        pytest.fail("Propagation happened before whole-segment capability check")

    monkeypatch.setattr(last, "backtrack", unexpected)
    unsupported = cheetah.Segment([cheetah.Aperture()], name="unsupported_nested")
    segment = cheetah.Segment([unsupported, last])
    assert not segment.supports_backtracking
    with pytest.raises(NotImplementedError, match="unsupported_nested"):
        segment.backtrack(make_beam(cheetah.ParameterBeam))


def test_empty_segment():
    segment = cheetah.Segment([])
    beam = make_beam(cheetah.ParameterBeam)
    assert segment.supports_backtracking
    assert segment.backtrack(beam) is beam


@pytest.mark.parametrize(
    "name, parameter",
    [
        ("Quadrupole", "k1"),
        ("HorizontalCorrector", "angle"),
        ("VerticalCorrector", "angle"),
        ("CombinedCorrector", "vertical_angle"),
        ("Dipole", "angle"),
        ("Dipole", "length"),
        ("Dipole", "fringe_integral_exit"),
        ("RBend", "angle"),
    ],
)
def test_backtracking_gradients_and_repeated_backward(name, parameter):
    element = make_element(name)
    value = torch.nn.Parameter(getattr(element, parameter).clone())
    setattr(element, parameter, value)
    beam = make_beam(cheetah.ParameterBeam)
    gradients = []
    for _ in range(2):
        element.zero_grad()
        recovered = element.backtrack(beam)
        loss = recovered.mu[..., :6].square().sum() + recovered.cov.square().sum()
        loss.backward()
        gradients.append(value.grad.clone())
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().max() > 0.0
    assert torch.allclose(gradients[0], gradients[1])
    step = 1e-5
    with torch.no_grad():
        value.add_(step)
        upper = element.backtrack(beam)
        upper_loss = upper.mu[..., :6].square().sum() + upper.cov.square().sum()
        value.sub_(2.0 * step)
        lower = element.backtrack(beam)
        lower_loss = lower.mu[..., :6].square().sum() + lower.cov.square().sum()
        value.add_(step)
    assert torch.allclose(
        gradients[0], (upper_loss - lower_loss) / (2.0 * step), rtol=1e-5, atol=1e-12
    )


@pytest.mark.parametrize("name", ["Quadrupole", "HorizontalCorrector"])
def test_inverse_cache_invalidation(name):
    element = make_element(name)
    beam = make_beam(cheetah.ParameterBeam)
    forward = element.first_order_transfer_map(beam.energy, beam.species).clone()
    inverse = element.inverse_first_order_transfer_map(beam.energy, beam.species)
    assert torch.allclose(
        forward @ inverse, torch.eye(7, dtype=forward.dtype), rtol=1e-13, atol=1e-14
    )
    element.length.add_(0.2)
    changed = element.inverse_first_order_transfer_map(beam.energy, beam.species)
    assert not torch.equal(changed, inverse)
    low_energy = beam.energy / 100.0
    changed_energy = element.inverse_first_order_transfer_map(low_energy, beam.species)
    assert not torch.equal(changed_energy, changed)
    element.float()
    assert (
        element.inverse_first_order_transfer_map(
            low_energy.float(), beam.species.float()
        ).dtype
        == torch.float32
    )


@pytest.mark.parametrize("direction", ["track", "backtrack"])
def test_trainable_map_after_no_grad(direction):
    element = make_element("Quadrupole")
    element.k1 = torch.nn.Parameter(element.k1.clone())
    beam = make_beam(cheetah.ParameterBeam)
    with torch.no_grad():
        getattr(element, direction)(beam)
    result = getattr(element, direction)(beam)
    result.mu.square().sum().backward()
    assert element.k1.grad is not None
    assert torch.isfinite(element.k1.grad).all()


@pytest.mark.parametrize("name", ["Segment", "Superimposed", "Marker", "BPM", "Screen"])
def test_backtracking_without_inverse_map(name):
    """
    Test that certain diagnostic elements without an explicite inverse map method
    can still backtrack.
    """
    if name == "Segment":
        element = cheetah.Segment([make_element("Drift")])
    elif name == "Superimposed":
        element = cheetah.Superimposed(make_element("Drift"), cheetah.Marker())
    else:
        element = getattr(cheetah, name)(dtype=torch.float64)
    beam = make_beam(cheetah.ParameterBeam)
    with pytest.raises(NotImplementedError):
        element.inverse_first_order_transfer_map(beam.energy, beam.species)
    actual = element.backtrack(element.track(beam))
    expected = beam
    assert actual.species.name == expected.species.name
    non_module_features = [
        feature for feature in expected.defining_features if feature != "species"
    ]
    for feature in non_module_features:
        assert torch.allclose(
            getattr(actual, feature),
            getattr(expected, feature),
            rtol=1e-13,
            atol=1e-14,
        )


def test_segment_backtracking_updates_with_child():
    child = make_element("Quadrupole")
    segment = cheetah.Segment([child])
    beam = make_beam(cheetah.ParameterBeam)
    before = segment.backtrack(beam)
    child.k1.add_(0.2)
    after = segment.backtrack(beam)
    assert not torch.equal(before.mu, after.mu)
    actual = after
    expected = child.backtrack(beam)
    assert actual.species.name == expected.species.name
    non_module_features = [
        feature for feature in expected.defining_features if feature != "species"
    ]
    for feature in non_module_features:
        assert torch.allclose(
            getattr(actual, feature),
            getattr(expected, feature),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("beam_type", [cheetah.ParticleBeam, cheetah.ParameterBeam])
@pytest.mark.parametrize("name", ["Quadrupole", "CombinedCorrector"])
def test_backtracking_beam_gradients(beam_type, name):
    element = make_element(name)
    beam = make_beam(beam_type)
    coordinates = beam.particles if beam_type is cheetah.ParticleBeam else beam.mu
    coordinates.requires_grad_()
    result = element.backtrack(beam)
    recovered = result.particles if beam_type is cheetah.ParticleBeam else result.mu
    recovered[..., :6].square().sum().backward()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()


@pytest.mark.parametrize("column", range(6))
def test_custom_map_rejects_non_affine_row(column):
    tm = torch.eye(7)
    tm[6, column] = 0.1
    with pytest.raises(AssertionError, match="seventh row"):
        cheetah.CustomTransferMap(tm)


def test_custom_map_singular_backtracking():
    tm = torch.eye(7)
    tm[0, 0] = 0.0
    element = cheetah.CustomTransferMap(tm)
    assert element.supports_backtracking
    with pytest.raises(torch.linalg.LinAlgError):
        element.backtrack(make_beam(cheetah.ParameterBeam))


def test_custom_map_inverse_cache_and_gradients():
    element = make_element("CustomTransferMap")
    beam = make_beam(cheetah.ParameterBeam)
    before = element.inverse_first_order_transfer_map(beam.energy, beam.species).clone()
    element.predefined_transfer_map[0, 0] += 0.2
    after = element.inverse_first_order_transfer_map(beam.energy, beam.species)
    assert not torch.equal(before, after)
    element.predefined_transfer_map = torch.nn.Parameter(
        element.predefined_transfer_map.clone()
    )
    gradients = []
    for _ in range(2):
        element.zero_grad()
        result = element.backtrack(beam)
        result.mu[:6].square().sum().backward()
        grad = element.predefined_transfer_map.grad.clone()
        assert torch.isfinite(grad).all()
        assert grad.abs().max() > 0.0
        gradients.append(grad)
    torch.testing.assert_close(gradients[0], gradients[1])
