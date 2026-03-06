import pickle
from pathlib import Path

import pytest
import torch

import cheetah
from cheetah.utils import is_mps_available_and_functional


@pytest.mark.for_every_element("element")
def test_element_subclasses_is_active_boolean(element):
    """
    Test that the `is_active` property of all `Element` subclasses returns a boolean if
    the element class has an `is_active` property.
    """
    assert not hasattr(element, "is_active") or isinstance(element.is_active, bool)


@pytest.mark.for_every_element("element")
def test_all_element_subclasses_is_skippable_boolean(element):
    """
    Test that the `is_skippable` property of all `Element` subclasses returns a boolean.
    """
    assert isinstance(element.is_skippable, bool)


@pytest.mark.for_every_element("element")
def test_defining_features_dtype(element):
    """
    Test that all defining features of `Element` subclasses that are `torch.Tensor`are
    properly converted between different dtypes. This transitively tests if all defining
    features are registered as pytorch buffers.
    """

    # Ensure all features have the same dtype initially
    for feature in element.defining_tensors:
        assert getattr(element, feature).dtype == torch.float32

    element.to(torch.float64)

    # Ensure all features have been converted to float64
    for feature in element.defining_tensors:
        assert getattr(element, feature).dtype == torch.float64


@pytest.mark.for_every_element("element")
@pytest.mark.parametrize(
    "device, dtype",
    [
        (torch.device("cpu"), torch.float32),
        (torch.device("cpu"), torch.float64),
        pytest.param(
            torch.device("cuda"),
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            torch.float32,
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
    ids=["cpu-float32", "cpu-float64", "cuda-float32", "mps-float32"],
)
def test_particle_beam_tracking_with_device_and_dtype(element, device, dtype):
    """
    Test that `Element` subclasses work correctly on various devices and with various
    dtypes if tracked with a `ParticleBeam`.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.25)),
            element,
            cheetah.Drift(length=torch.tensor(0.25)),
        ]
    ).to(device=device, dtype=dtype)
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        total_charge=torch.tensor(1e-9),
        mu_x=torch.tensor(5e-5),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
    ).to(device=device, dtype=dtype)

    # Run in part to see if errors are raised
    outgoing_beam = segment.track(incoming_beam)

    # Check device and dtype of the output
    for attribute in outgoing_beam.UNVECTORIZED_NUM_ATTR_DIMS.keys():
        assert getattr(outgoing_beam, attribute).device.type == device.type
        assert getattr(outgoing_beam, attribute).dtype == dtype


@pytest.mark.for_every_element(
    "element",
    xfail_if=lambda element: isinstance(
        element,
        (
            cheetah.SpaceChargeKick3D,
            cheetah.SpaceChargeKick2D,
            cheetah.TransverseDeflectingCavity,
        ),
    )
    or (
        isinstance(
            element,
            (
                cheetah.Dipole,
                cheetah.Drift,
                cheetah.Quadrupole,
                cheetah.RBend,
                cheetah.Sextupole,
            ),
        )
        and element.tracking_method != "linear"
    ),
)
@pytest.mark.parametrize(
    "device, dtype",
    [
        (torch.device("cpu"), torch.float32),
        (torch.device("cpu"), torch.float64),
        pytest.param(
            torch.device("cuda"),
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            torch.float32,
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
    ids=["cpu-float32", "cpu-float64", "cuda-float32", "mps-float32"],
)
def test_parameter_beam_tracking_with_device_and_dtype(element, device, dtype):
    """
    Test that `Element` subclasses work correctly on various devices and with various
    dtypes if tracked with a `ParticleBeam`.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.25)),
            element,
            cheetah.Drift(length=torch.tensor(0.25)),
        ]
    ).to(device=device, dtype=dtype)
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        total_charge=torch.tensor(1e-9),
        mu_x=torch.tensor(5e-5),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
    ).to(device=device, dtype=dtype)

    # Run in part to see if errors are raised
    outgoing_beam = segment.track(incoming_beam)

    # Check device and dtype of the output
    for attribute in outgoing_beam.UNVECTORIZED_NUM_ATTR_DIMS.keys():
        assert getattr(outgoing_beam, attribute).device.type == device.type
        assert getattr(outgoing_beam, attribute).dtype == dtype


@pytest.mark.for_every_element("element")
@pytest.mark.parametrize("beam_cls", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_species_preservation(element, beam_cls):
    """
    Test that tracking through an element preserves the particle species.

    Regression test for GitHub issue #570.
    """
    # Skip tests with ParameterBeam and drift-kick-drift tracking because that
    # combination is not supported.
    if beam_cls == cheetah.ParameterBeam and element.tracking_method != "linear":
        pytest.xfail("ParameterBeam does not support drift-kick-drift elements")

    # Test with proton beam because it is different from the default electron species
    incoming = beam_cls.from_twiss(
        beta_x=torch.tensor(3.14),
        beta_y=torch.tensor(42.0),
        species=cheetah.Species("proton"),
        energy=torch.tensor(155e6),
    )

    outgoing = element.track(incoming)

    assert outgoing.species.name == incoming.species.name
    assert (
        outgoing.species.num_elementary_charges
        == incoming.species.num_elementary_charges
    )
    assert outgoing.species.mass_eV == incoming.species.mass_eV


def test_transfer_map_cache():
    """Test that the transfer map is cached after the first computation."""
    quadrupole = cheetah.Quadrupole(length=torch.tensor(0.5), k1=torch.tensor(1.0))
    energy = torch.tensor(155e6)
    species = cheetah.Species("electron")

    first_cached_transfer_map = quadrupole.first_order_transfer_map(energy, species)

    second_cached_transfer_map = quadrupole.first_order_transfer_map(energy, species)

    assert id(first_cached_transfer_map) == id(second_cached_transfer_map)
    assert torch.equal(first_cached_transfer_map, second_cached_transfer_map)


def test_transfer_map_cache_caches_different_between_elements_of_same_type():
    """
    Test that the transfer map is cached after the first computation, while two elements
    of the same type do not share the same cache.
    """
    quadrupole_1 = cheetah.Quadrupole(length=torch.tensor(0.5), k1=torch.tensor(1.0))
    energy = torch.tensor(155e6)
    species = cheetah.Species("electron")

    first_cached_transfer_map = quadrupole_1.first_order_transfer_map(energy, species)

    quadrupole_2 = cheetah.Quadrupole(length=torch.tensor(0.5), k1=torch.tensor(1.0))
    quadrupole_2.first_order_transfer_map(energy, species)

    second_cached_transfer_map = quadrupole_1.first_order_transfer_map(energy, species)

    assert id(first_cached_transfer_map) == id(second_cached_transfer_map)
    assert torch.equal(first_cached_transfer_map, second_cached_transfer_map)


def test_transfer_map_cache_dtype_conversion():
    """
    Test that converting an element to a different dtype invalidates the cache and that
    the transfer map is recomputed in the new dtype.
    """
    quadrupole = cheetah.Quadrupole(length=torch.tensor(0.5), k1=torch.tensor(1.0))
    energy = torch.tensor(155e6)
    species = cheetah.Species("electron")

    original_transfer_map = quadrupole.first_order_transfer_map(energy, species)
    assert original_transfer_map.dtype == torch.float32

    quadrupole.to(torch.float64)
    converted_transfer_map = quadrupole.first_order_transfer_map(energy, species)

    assert id(original_transfer_map) != id(converted_transfer_map)
    assert converted_transfer_map.dtype == torch.float64
    assert torch.allclose(
        original_transfer_map.to(torch.float64), converted_transfer_map
    )


def test_transfer_map_cache_invalidation_element_property_assignment():
    """Test that assigning to an element property invalidates the transfer map cache."""
    quadrupole = cheetah.Quadrupole(length=torch.tensor(0.5), k1=torch.tensor(1.0))
    energy = torch.tensor(155e6)
    species = cheetah.Species("electron")

    original_transfer_map = quadrupole.first_order_transfer_map(energy, species)

    # Assign to a property
    quadrupole.k1 = torch.tensor(2.0)
    updated_transfer_map = quadrupole.first_order_transfer_map(energy, species)

    assert not torch.equal(original_transfer_map, updated_transfer_map)


def test_transfer_map_cache_invalidation_element_property_inplace():
    """
    Test that changing an element property inplace invalidates the transfer map cache.
    """
    strength = torch.tensor([1.0, 2.0])

    quadrupole = cheetah.Quadrupole(length=torch.tensor(0.5), k1=strength)
    energy = torch.tensor(155e6)
    species = cheetah.Species("electron")

    original_transfer_map = quadrupole.first_order_transfer_map(energy, species)

    # Change property inplace
    strength[0] = torch.tensor(42.0)
    updated_transfer_map = quadrupole.first_order_transfer_map(energy, species)

    assert not torch.equal(original_transfer_map, updated_transfer_map)


def test_transfer_map_cache_invalidation_energy():
    """Test that changing the beam energy invalidates the transfer map cache."""
    quadrupole = cheetah.Quadrupole(length=torch.tensor(0.5), k1=torch.tensor(1.0))
    original_energy = torch.tensor(155e6)
    species = cheetah.Species("electron")

    original_transfer_map = quadrupole.first_order_transfer_map(
        original_energy, species
    )

    updated_energy = torch.tensor(200e6)
    updated_transfer_map = quadrupole.first_order_transfer_map(updated_energy, species)

    assert not torch.equal(original_transfer_map, updated_transfer_map)


def test_transfer_map_cache_invalidation_species():
    """Test that changing the beam species invalidates the transfer map cache."""
    quadrupole = cheetah.Quadrupole(length=torch.tensor(0.5), k1=torch.tensor(1.0))
    energy = torch.tensor(155e6)
    original_species = cheetah.Species("electron")

    original_transfer_map = quadrupole.first_order_transfer_map(
        energy, original_species
    )

    updated_species = cheetah.Species("proton")
    updated_transfer_map = quadrupole.first_order_transfer_map(energy, updated_species)

    assert not torch.equal(original_transfer_map, updated_transfer_map)


@pytest.mark.for_every_element("element")
@pytest.mark.parametrize("beam_cls", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_consistency(element, beam_cls):
    """
    Test that tracking results are consistent with a previous version of Cheetah.

    NOTE: To generate new ground truth expected outputs, uncomment the relevant blocks
        below, and run `pytest -k test_consistency`. Please undo the commenting change
        before committing. Please do NOT commit changes to the ground truth expected
        outputs for elements that did not have intentional changes to their tracking
        behaviour.
    """
    if beam_cls == cheetah.ParameterBeam and element.tracking_method != "linear":
        pytest.xfail("ParameterBeam does not support drift-kick-drift elements")

    element = element.to(torch.float64)

    incoming_beam_path = (
        Path("tests") / "resources" / "ACHIP_EA1_2021.1351.001_subsampled_3000.pkl"
    )
    with incoming_beam_path.open("rb") as f:
        incoming_beam = pickle.load(f).to(torch.float64)
        if not isinstance(incoming_beam, beam_cls):
            incoming_beam = incoming_beam.as_parameter_beam()

    expected_outgoing_beam_path = (
        Path("tests")
        / "resources"
        / "consistency_expected_outgoing"
        / f"{element.__class__.__name__}_{beam_cls.__name__}_{element.name}.pkl"
    )
    # NOTE: Comment out the following block to generate ground truth expected outputs
    with expected_outgoing_beam_path.open("rb") as f:
        expected_outgoing_beam = pickle.load(f)

    actual_outgoing_beam = element.track(incoming_beam)

    # NOTE: Uncomment the following block to generate ground truth expected outputs
    # expected_outgoing_beam_path.parent.mkdir(parents=True, exist_ok=True)
    # with expected_outgoing_beam_path.open("wb") as f:
    #     pickle.dump(actual_outgoing_beam, f)

    assert actual_outgoing_beam.species.name == expected_outgoing_beam.species.name
    assert torch.allclose(actual_outgoing_beam.energy, expected_outgoing_beam.energy)
    assert torch.allclose(actual_outgoing_beam.s, expected_outgoing_beam.s)
    if isinstance(actual_outgoing_beam, cheetah.ParameterBeam):
        assert torch.allclose(actual_outgoing_beam.mu, expected_outgoing_beam.mu)
        assert torch.allclose(actual_outgoing_beam.cov, expected_outgoing_beam.cov)
        assert torch.allclose(
            actual_outgoing_beam.total_charge, expected_outgoing_beam.total_charge
        )
    else:  # ParticleBeam
        assert torch.allclose(
            actual_outgoing_beam.particles, expected_outgoing_beam.particles
        )
        assert torch.allclose(
            actual_outgoing_beam.particle_charges,
            expected_outgoing_beam.particle_charges,
        )
        assert torch.allclose(
            actual_outgoing_beam.survival_probabilities,
            expected_outgoing_beam.survival_probabilities,
        )
