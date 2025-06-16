import numpy as np
import pytest
import torch

import cheetah

from .resources import ARESlatticeStage3v1_9 as ocelot_lattice


@pytest.mark.parametrize("screen_method", ["histogram", "kde"])
def test_reading_shows_beam_particle(screen_method):
    """
    Test that a screen has a reading that shows some sign of the beam having hit it.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Screen(
                resolution=(100, 100),
                pixel_size=torch.tensor((1e-5, 1e-5)),
                is_active=True,
                method=screen_method,
                name="my_screen",
            ),
        ],
    )
    beam = cheetah.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert np.allclose(segment.my_screen.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert torch.all(segment.my_screen.reading >= 0.0)
    assert torch.any(segment.my_screen.reading > 0.0)


@pytest.mark.parametrize("kde_bandwidth", [5e-6, 1e-5, 5e-5])
def test_screen_kde_bandwidth(kde_bandwidth):
    """Test screen reading with KDE method and different explicit bandwidths."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Screen(
                resolution=(100, 100),
                pixel_size=torch.tensor((1e-5, 1e-5)),
                is_active=True,
                method="kde",
                name="my_screen",
                kde_bandwidth=torch.tensor(kde_bandwidth),
            ),
        ],
    )
    beam = cheetah.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert np.allclose(segment.my_screen.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert torch.all(segment.my_screen.reading >= 0.0)
    assert torch.any(segment.my_screen.reading > 0.0)


@pytest.mark.parametrize("screen_method", ["histogram", "kde"])
def test_reading_shows_beam_parameter(screen_method):
    """
    Test that a screen has a reading that shows some sign of the beam having hit it.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Screen(
                resolution=(100, 100),
                pixel_size=torch.tensor((1e-5, 1e-5)),
                is_active=True,
                method=screen_method,
                name="my_screen",
            ),
        ],
        name="my_segment",
    )
    beam = cheetah.ParameterBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert np.allclose(segment.my_screen.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert torch.all(segment.my_screen.reading >= 0.0)
    assert torch.any(segment.my_screen.reading > 0.0)


@pytest.mark.filterwarnings("ignore::cheetah.utils.DefaultParameterWarning")
@pytest.mark.parametrize("screen_method", ["histogram", "kde"])
def test_reading_shows_beam_ares(screen_method):
    """
    Test that a screen has a reading that shows some sign of the beam having hit it.
    """
    segment = cheetah.Segment.from_ocelot(ocelot_lattice.cell).subcell(
        "AREASOLA1", "AREABSCR1"
    )
    beam = cheetah.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    segment.AREABSCR1.method = screen_method

    segment.AREABSCR1.resolution = (2448, 2040)
    segment.AREABSCR1.pixel_size = torch.tensor(
        (3.3198e-6, 2.4469e-6),
        device=segment.AREABSCR1.pixel_size.device,
        dtype=segment.AREABSCR1.pixel_size.dtype,
    )
    segment.AREABSCR1.binning = 1
    segment.AREABSCR1.is_active = True

    assert isinstance(segment.AREABSCR1.reading, torch.Tensor)
    assert segment.AREABSCR1.reading.shape == (2040, 2448)
    assert np.allclose(segment.AREABSCR1.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.AREABSCR1.reading, torch.Tensor)
    assert segment.AREABSCR1.reading.shape == (2040, 2448)
    assert torch.all(segment.AREABSCR1.reading >= 0.0)
    assert torch.any(segment.AREABSCR1.reading > 0.0)


def test_reading_dtype_conversion():
    """Test that a dtype conversion is correctly reflected in the screen reading."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0), dtype=torch.float32),
            cheetah.Screen(name="screen", is_active=True, dtype=torch.float32),
        ],
    )
    beam = cheetah.ParameterBeam.from_parameters(dtype=torch.float32)
    assert segment.screen.reading.dtype == torch.float32

    # Test generating new image
    cloned = segment.clone()
    cloned.track(beam)
    cloned = cloned.double()
    assert cloned.screen.reading.dtype == torch.float64

    # Test reading from cache
    segment.track(beam)
    assert segment.screen.reading.dtype == torch.float32
    segment = segment.double()
    assert segment.screen.reading.dtype == torch.float64


def test_screen_reading_not_unintentionally_modified_parameter_beam():
    """
    Test that the screen reading is not unintentionally modified if the user modifies
    the incoming out outgoing beam after tracking. Considers the case where the incoming
    beam is a `ParameterBeam`.
    """
    incoming = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    screen = cheetah.Screen(is_active=True)

    outgoing = screen.track(incoming)
    original_read_beam = screen.get_read_beam().clone()

    incoming.mu *= 2.0
    incoming.cov *= 3.0
    incoming.energy *= 4.0
    incoming.total_charge *= 5.0
    incoming.species.charge_coulomb *= 6.0
    outgoing.mu *= 0.7
    outgoing.cov *= 0.5
    outgoing.energy *= 0.3
    outgoing.total_charge *= 0.2
    outgoing.species.charge_coulomb *= 0.1

    read_beam_after_modification = screen.get_read_beam()

    assert torch.all(original_read_beam.mu == read_beam_after_modification.mu)
    assert torch.all(original_read_beam.cov == read_beam_after_modification.cov)
    assert original_read_beam.energy == read_beam_after_modification.energy
    assert original_read_beam.total_charge == read_beam_after_modification.total_charge
    assert (
        original_read_beam.species.charge_coulomb
        == read_beam_after_modification.species.charge_coulomb
    )


def test_screen_reading_not_unintentionally_modified_particle_beam():
    """
    Test that the screen reading is not unintentionally modified if the user modifies
    the incoming out outgoing beam after tracking. Considers the case where the incoming
    beam is a `ParticleBeam`.
    """
    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    screen = cheetah.Screen(is_active=True)

    outgoing = screen.track(incoming)
    original_read_beam = screen.get_read_beam().clone()

    incoming.particles *= 2.0
    incoming.energy *= 3.0
    incoming.particle_charges *= 4.0
    incoming.survival_probabilities *= 0.9
    incoming.species.charge_coulomb *= 5.0
    outgoing.particles *= 0.7
    outgoing.energy *= 0.5
    outgoing.particle_charges *= 0.3
    outgoing.survival_probabilities *= 0.1
    outgoing.species.charge_coulomb *= 0.2

    read_beam_after_modification = screen.get_read_beam()

    assert torch.all(
        original_read_beam.particles == read_beam_after_modification.particles
    )
    assert original_read_beam.energy == read_beam_after_modification.energy
    assert torch.all(
        original_read_beam.particle_charges
        == read_beam_after_modification.particle_charges
    )
    assert torch.all(
        original_read_beam.survival_probabilities
        == read_beam_after_modification.survival_probabilities
    )
    assert (
        original_read_beam.species.charge_coulomb
        == read_beam_after_modification.species.charge_coulomb
    )
