import torch

import cheetah


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
