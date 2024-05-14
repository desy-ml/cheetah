import torch

from cheetah import Drift, ParticleBeam, Quadrupole, Segment


def test_tilted_quadrupole_batch():
    batch_shape = torch.Size([3])
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1000000),
        energy=torch.tensor([1e9]),
        mu_x=torch.tensor([1e-5]),
    ).broadcast(batch_shape)
    segment = Segment(
        [
            Quadrupole(
                length=torch.tensor([0.5, 0.5, 0.5]),
                k1=torch.tensor([1.0, 1.0, 1.0]),
                tilt=torch.tensor([torch.pi / 4, torch.pi / 2, torch.pi * 5 / 4]),
            ),
            Drift(length=torch.tensor([0.5])).broadcast(batch_shape),
        ]
    )
    outgoing = segment(incoming)

    # Check pi/4 and 5/4*pi rotations is the same for quadrupole
    assert torch.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check pi/2 rotation is different
    assert not torch.allclose(outgoing.particles[0], outgoing.particles[1])


def test_tilted_quadrupole_multiple_batch_dimension():
    batch_shape = torch.Size([3, 2])
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(10000),
        energy=torch.tensor([1e9]),
        mu_x=torch.tensor([1e-5]),
    ).broadcast(batch_shape)
    segment = Segment(
        [
            Quadrupole(
                length=torch.tensor([0.5]),
                k1=torch.tensor([1.0]),
                tilt=torch.tensor([torch.pi / 4]),
            ),
            Drift(length=torch.tensor([0.5])),
        ]
    ).broadcast(batch_shape)
    outgoing = segment(incoming)

    assert torch.allclose(outgoing.particles[0, 0], outgoing.particles[0, 1])
