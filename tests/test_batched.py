import torch

import cheetah

from .resources import ARESlatticeStage3v1_9 as ares


def test_segment_length_shape():
    """Test that the shape of a segment's length matches the input."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([0.6, 0.5])),
            cheetah.Quadrupole(
                length=torch.tensor([0.2, 0.25]), k1=torch.tensor([4.2, 4.2])
            ),
            cheetah.Drift(length=torch.tensor([0.4, 0.3])),
        ]
    )

    assert segment.length.shape == (2,)


def test_track_particle_single_element_shape():
    """
    Test that the shape of a beam tracked through a single element matches the input.
    """
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor([0.2, 0.25]), k1=torch.tensor([4.2, 4.2])
    )
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, sigma_x=torch.tensor([1e-5, 2e-5])
    )

    outgoing = quadrupole.track(incoming)

    assert outgoing.particles.shape == incoming.particles.shape
    assert outgoing.particles.shape == (2, 100_000, 7)
    assert outgoing.mu_x.shape == (2,)
    assert outgoing.mu_xp.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_yp.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_xp.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_yp.shape == (2,)
    assert outgoing.sigma_s.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)
    assert outgoing.particle_charges.shape == (2, 100_000)
    assert isinstance(outgoing.num_particles, int)


def test_track_particle_segment_shape():
    """
    Test that the shape of a beam tracked through a segment matches the input.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([0.6, 0.5])),
            cheetah.Quadrupole(
                length=torch.tensor([0.2, 0.25]), k1=torch.tensor([4.2, 4.2])
            ),
            cheetah.Drift(length=torch.tensor([0.4, 0.3])),
        ]
    )
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, sigma_x=torch.tensor([1e-5, 2e-5])
    )

    outgoing = segment.track(incoming)

    assert outgoing.particles.shape == incoming.particles.shape
    assert outgoing.particles.shape == (2, 100_000, 7)
    assert outgoing.mu_x.shape == (2,)
    assert outgoing.mu_xp.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_yp.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_xp.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_yp.shape == (2,)
    assert outgoing.sigma_s.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)
    assert outgoing.particle_charges.shape == (2, 100_000)
    assert isinstance(outgoing.num_particles, int)


def test_track_parameter_single_element_shape():
    """
    Test that the shape of a beam tracked through a single element matches the input.
    """
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor([0.2, 0.25]), k1=torch.tensor([4.2, 4.2])
    )
    incoming = cheetah.ParameterBeam.from_parameters(sigma_x=torch.tensor([1e-5, 2e-5]))

    outgoing = quadrupole.track(incoming)

    assert outgoing.mu_x.shape == (2,)
    assert outgoing.mu_xp.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_yp.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_xp.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_yp.shape == (2,)
    assert outgoing.sigma_s.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)


def test_track_parameter_segment_shape():
    """
    Test that the shape of a beam tracked through a segment matches the input.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([0.6, 0.5])),
            cheetah.Quadrupole(
                length=torch.tensor([0.2, 0.25]), k1=torch.tensor([4.2, 4.2])
            ),
            cheetah.Drift(length=torch.tensor([0.4, 0.3])),
        ]
    )
    incoming = cheetah.ParameterBeam.from_parameters(sigma_x=torch.tensor([1e-5, 2e-5]))

    outgoing = segment.track(incoming)

    assert outgoing.mu_x.shape == (2,)
    assert outgoing.mu_xp.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_yp.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_xp.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_yp.shape == (2,)
    assert outgoing.sigma_s.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)


def test_enormous_through_ares():
    """Test ARES EA with a huge number of settings."""
    segment = cheetah.Segment.from_ocelot(ares.cell).subcell("AREASOLA1", "AREABSCR1")
    incoming = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    segment_broadcast = segment.broadcast((100_000,))
    incoming_broadcast = incoming.broadcast((100_000,))

    segment_broadcast.AREAMQZM1.k1 = torch.linspace(-30.0, 30.0, 100_000)

    outgoing = segment_broadcast.track(incoming_broadcast)

    assert outgoing.mu_x.shape == (100_000,)
    assert outgoing.mu_xp.shape == (100_000,)
    assert outgoing.mu_y.shape == (100_000,)
    assert outgoing.mu_yp.shape == (100_000,)
    assert outgoing.sigma_x.shape == (100_000,)
    assert outgoing.sigma_xp.shape == (100_000,)
    assert outgoing.sigma_y.shape == (100_000,)
    assert outgoing.sigma_yp.shape == (100_000,)
    assert outgoing.sigma_s.shape == (100_000,)
    assert outgoing.sigma_p.shape == (100_000,)
    assert outgoing.energy.shape == (100_000,)
    assert outgoing.total_charge.shape == (100_000,)


def test_before_after_broadcast_tracking_equal():
    """
    Test that when tracking through a segment after broadcasting, the resulting beam is
    the same as in the segment before broadcasting.
    """
    segment = cheetah.Segment.from_ocelot(ares.cell).subcell("AREASOLA1", "AREABSCR1")
    incoming = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    segment.AREAMQZM1.k1 = torch.tensor([4.2])
    outgoing = segment.track(incoming)

    broadcast_segment = segment.broadcast((10,))
    broadcast_incoming = incoming.broadcast((10,))
    broadcast_outgoing = broadcast_segment.track(broadcast_incoming)

    for i in range(10):
        assert torch.all(broadcast_outgoing._mu[i] == outgoing._mu[0])
        assert torch.all(broadcast_outgoing._cov[i] == outgoing._cov[0])


def test_broadcast_customtransfermap():
    """Test that broadcasting a `CustomTransferMap` element gives the correct result."""
    tm = torch.tensor(
        [
            [
                [1.0, 4.0e-02, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0e-05],
                [0.0, 0.0, 1.0, 4.0e-02, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -4.6422e-07, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )

    element = cheetah.CustomTransferMap(length=torch.tensor([0.4]), transfer_map=tm)
    broadcast_element = element.broadcast((10,))

    assert broadcast_element.length.shape == (10,)
    assert broadcast_element._transfer_map.shape == (10, 7, 7)
    for i in range(10):
        assert torch.all(broadcast_element._transfer_map[i] == tm[0])


def test_broadcast_drift():
    """Test that broadcasting a `Drift` element gives the correct result."""
    element = cheetah.Drift(length=torch.tensor([0.4]))
    broadcast_element = element.broadcast((10,))

    assert broadcast_element.length.shape == (10,)
    for i in range(10):
        assert broadcast_element.length[i] == 0.4


def test_broadcast_quadrupole():
    """Test that broadcasting a `Quadrupole` element gives the correct result."""

    # TODO Add misalignment to the test
    # TODO Add tilt to the test

    element = cheetah.Quadrupole(length=torch.tensor([0.4]), k1=torch.tensor([4.2]))
    broadcast_element = element.broadcast((10,))

    assert broadcast_element.length.shape == (10,)
    assert broadcast_element.k1.shape == (10,)
    for i in range(10):
        assert broadcast_element.length[i] == 0.4
        assert broadcast_element.k1[i] == 4.2
