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


def test_segment_length_shape_2d():
    """
    Test that the shape of a segment's length matches the input for a batch with
    multiple dimensions.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([[0.6, 0.5], [0.4, 0.3], [0.4, 0.3]])),
            cheetah.Quadrupole(
                length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.3, 0.35]]),
                k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.3, 4.3]]),
            ),
            cheetah.Drift(length=torch.tensor([[0.4, 0.3], [0.2, 0.1], [0.2, 0.1]])),
        ]
    )

    assert segment.length.shape == (3, 2)


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
    assert outgoing.mu_px.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_py.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_px.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_py.shape == (2,)
    assert outgoing.sigma_tau.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)
    assert outgoing.particle_charges.shape == (2, 100_000)
    assert isinstance(outgoing.num_particles, int)


def test_track_particle_single_element_shape_2d():
    """
    Test that the shape of a beam tracked through a single element matches the input for
    an n-dimensional batch.
    """
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.4, 0.45]]),
        k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.4, 4.4]]),
    )
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000,
        sigma_x=torch.tensor([[1e-5, 2e-5], [2e-5, 3e-5], [3e-5, 4e-5]]),
    )

    outgoing = quadrupole.track(incoming)

    assert outgoing.particles.shape == incoming.particles.shape
    assert outgoing.particles.shape == (3, 2, 100_000, 7)
    assert outgoing.mu_x.shape == (3, 2)
    assert outgoing.mu_px.shape == (3, 2)
    assert outgoing.mu_y.shape == (3, 2)
    assert outgoing.mu_py.shape == (3, 2)
    assert outgoing.sigma_x.shape == (3, 2)
    assert outgoing.sigma_px.shape == (3, 2)
    assert outgoing.sigma_y.shape == (3, 2)
    assert outgoing.sigma_py.shape == (3, 2)
    assert outgoing.sigma_tau.shape == (3, 2)
    assert outgoing.sigma_p.shape == (3, 2)
    assert outgoing.energy.shape == (3, 2)
    assert outgoing.total_charge.shape == (3, 2)
    assert outgoing.particle_charges.shape == (3, 2, 100_000)
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
    assert outgoing.mu_px.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_py.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_px.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_py.shape == (2,)
    assert outgoing.sigma_tau.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)
    assert outgoing.particle_charges.shape == (2, 100_000)
    assert isinstance(outgoing.num_particles, int)


def test_track_particle_segment_shape_2d():
    """
    Test that the shape of a beam tracked through a segment matches the input for the
    case of a multi-dimensional batch.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]])),
            cheetah.Quadrupole(
                length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.4, 0.45]]),
                k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.4, 4.4]]),
            ),
            cheetah.Drift(length=torch.tensor([[0.4, 0.3], [0.6, 0.5], [0.8, 0.7]])),
        ]
    )
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000,
        sigma_x=torch.tensor([[1e-5, 2e-5], [2e-5, 3e-5], [3e-5, 4e-5]]),
    )

    outgoing = segment.track(incoming)

    assert outgoing.particles.shape == incoming.particles.shape
    assert outgoing.particles.shape == (3, 2, 100_000, 7)
    assert outgoing.mu_x.shape == (3, 2)
    assert outgoing.mu_px.shape == (3, 2)
    assert outgoing.mu_y.shape == (3, 2)
    assert outgoing.mu_py.shape == (3, 2)
    assert outgoing.sigma_x.shape == (3, 2)
    assert outgoing.sigma_px.shape == (3, 2)
    assert outgoing.sigma_y.shape == (3, 2)
    assert outgoing.sigma_py.shape == (3, 2)
    assert outgoing.sigma_tau.shape == (3, 2)
    assert outgoing.sigma_p.shape == (3, 2)
    assert outgoing.energy.shape == (3, 2)
    assert outgoing.total_charge.shape == (3, 2)
    assert outgoing.particle_charges.shape == (3, 2, 100_000)
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
    assert outgoing.mu_px.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_py.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_px.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_py.shape == (2,)
    assert outgoing.sigma_tau.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)


def test_track_parameter_single_element_shape_2d():
    """
    Test that the shape of a beam tracked through a single element matches the input for
    an n-dimensional batch.
    """
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.4, 0.45]]),
        k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.4, 4.4]]),
    )
    incoming = cheetah.ParameterBeam.from_parameters(
        sigma_x=torch.tensor([[1e-5, 2e-5], [2e-5, 3e-5], [3e-5, 4e-5]])
    )

    outgoing = quadrupole.track(incoming)

    assert outgoing.mu_x.shape == (3, 2)
    assert outgoing.mu_px.shape == (3, 2)
    assert outgoing.mu_y.shape == (3, 2)
    assert outgoing.mu_py.shape == (3, 2)
    assert outgoing.sigma_x.shape == (3, 2)
    assert outgoing.sigma_px.shape == (3, 2)
    assert outgoing.sigma_y.shape == (3, 2)
    assert outgoing.sigma_py.shape == (3, 2)
    assert outgoing.sigma_tau.shape == (3, 2)
    assert outgoing.sigma_p.shape == (3, 2)
    assert outgoing.energy.shape == (3, 2)
    assert outgoing.total_charge.shape == (3, 2)


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
    assert outgoing.mu_px.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_py.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_px.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_py.shape == (2,)
    assert outgoing.sigma_tau.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)


def test_track_parameter_segment_shape_2d():
    """
    Test that the shape of a beam tracked through a segment matches the input for the
    case of a multi-dimensional batch.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]])),
            cheetah.Quadrupole(
                length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.4, 0.45]]),
                k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.4, 4.4]]),
            ),
            cheetah.Drift(length=torch.tensor([[0.4, 0.3], [0.6, 0.5], [0.8, 0.7]])),
        ]
    )
    incoming = cheetah.ParameterBeam.from_parameters(
        sigma_x=torch.tensor([[1e-5, 2e-5], [2e-5, 3e-5], [3e-5, 4e-5]])
    )

    outgoing = segment.track(incoming)

    assert outgoing.mu_x.shape == (3, 2)
    assert outgoing.mu_px.shape == (3, 2)
    assert outgoing.mu_y.shape == (3, 2)
    assert outgoing.mu_py.shape == (3, 2)
    assert outgoing.sigma_x.shape == (3, 2)
    assert outgoing.sigma_px.shape == (3, 2)
    assert outgoing.sigma_y.shape == (3, 2)
    assert outgoing.sigma_py.shape == (3, 2)
    assert outgoing.sigma_tau.shape == (3, 2)
    assert outgoing.sigma_p.shape == (3, 2)
    assert outgoing.energy.shape == (3, 2)
    assert outgoing.total_charge.shape == (3, 2)


def test_enormous_through_ares():
    """Test ARES EA with a huge number of settings."""
    segment = cheetah.Segment.from_ocelot(ares.cell).subcell("AREASOLA1", "AREABSCR1")
    incoming = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    segment.AREAMQZM1.k1 = torch.linspace(-30.0, 30.0, 100_000).repeat(3, 1)

    outgoing = segment.track(incoming)

    assert outgoing.mu_x.shape == (3, 100_000)
    assert outgoing.mu_px.shape == (3, 100_000)
    assert outgoing.mu_y.shape == (3, 100_000)
    assert outgoing.mu_py.shape == (3, 100_000)
    assert outgoing.sigma_x.shape == (3, 100_000)
    assert outgoing.sigma_px.shape == (3, 100_000)
    assert outgoing.sigma_y.shape == (3, 100_000)
    assert outgoing.sigma_py.shape == (3, 100_000)
    assert outgoing.sigma_tau.shape == (3, 100_000)
    assert outgoing.sigma_p.shape == (3, 100_000)
    assert outgoing.energy.shape == (3, 100_000)
    assert outgoing.total_charge.shape == (3, 100_000)


def test_cavity_with_zero_and_non_zero_voltage():
    """
    Tests that if zero and non-zero voltages are passed to a cavity in a single batch,
    there are no errors. This test does NOT check physical correctness.
    """
    cavity = cheetah.Cavity(
        length=torch.tensor([3.0441, 3.0441, 3.0441]),
        voltage=torch.tensor([0.0, 48198468.0, 0.0]),
        phase=torch.tensor([48198468.0, 48198468.0, 48198468.0]),
        frequency=torch.tensor([2.8560e09, 2.8560e09, 2.8560e09]),
        name="my_test_cavity",
    )
    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, sigma_x=torch.tensor(1e-5)
    )

    _ = cavity.track(beam)
