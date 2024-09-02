import pytest
import torch

from cheetah import (
    Dipole,
    Drift,
    ParameterBeam,
    ParticleBeam,
    Quadrupole,
    RBend,
    Segment,
)


def test_dipole_off():
    """
    Test that a dipole with angle=0 behaves still like a drift.
    """
    dipole = Dipole(length=torch.tensor(1.0), angle=torch.tensor(0.0))
    drift = Drift(length=torch.tensor(1.0))
    incoming_beam = ParameterBeam.from_parameters(
        sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
    )
    outbeam_dipole_off = dipole(incoming_beam)
    outbeam_drift = drift(incoming_beam)

    dipole.angle = torch.tensor(1.0, device=dipole.angle.device)
    outbeam_dipole_on = dipole(incoming_beam)

    assert dipole.name is not None
    assert torch.allclose(outbeam_dipole_off.sigma_x, outbeam_drift.sigma_x)
    assert not torch.allclose(outbeam_dipole_on.sigma_x, outbeam_drift.sigma_x)


def test_dipole_focussing():
    """
    Test that a dipole with focussing moment behaves like a quadrupole.
    """
    dipole = Dipole(length=torch.tensor([1.0]), k1=torch.tensor([10.0]))
    quadrupole = Quadrupole(length=torch.tensor([1.0]), k1=torch.tensor([10.0]))
    incoming_beam = ParameterBeam.from_parameters(
        sigma_px=torch.tensor([2e-7]), sigma_py=torch.tensor([2e-7])
    )
    outbeam_dipole_on = dipole.track(incoming_beam)
    outbeam_quadrupole = quadrupole.track(incoming_beam)

    dipole.k1 = torch.tensor([0.0], device=dipole.k1.device)
    outbeam_dipole_off = dipole.track(incoming_beam)

    assert dipole.name is not None
    assert torch.allclose(outbeam_dipole_on.sigma_x, outbeam_quadrupole.sigma_x)
    assert not torch.allclose(outbeam_dipole_off.sigma_x, outbeam_quadrupole.sigma_x)


@pytest.mark.parametrize("DipoleType", [Dipole, RBend])
def test_dipole_batched_execution(DipoleType):
    """
    Test that a dipole with batch dimensions behaves as expected.
    """
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(100),
        energy=torch.tensor(1e9),
        mu_x=torch.tensor(1e-5),
    )

    # Test batching to generate 3 beam lines
    segment = Segment(
        [
            DipoleType(
                length=torch.tensor([0.5, 0.5, 0.5]),
                angle=torch.tensor([0.1, 0.2, 0.1]),
            ),
            Drift(length=torch.tensor(0.5)),
        ]
    )
    outgoing = segment(incoming)

    assert outgoing.particles.shape == torch.Size([3, 100, 7])
    assert outgoing.mu_x.shape == torch.Size([3])

    # Check that dipole with same bend angle produce same output
    assert torch.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check different angles do make a difference
    assert not torch.allclose(outgoing.particles[0], outgoing.particles[1])

    # Test batching to generate 18 beamlines
    segment = Segment(
        [
            Dipole(
                length=torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1),
                angle=torch.tensor([0.1, 0.2, 0.1]).reshape(1, 3),
            ),
            Drift(length=torch.tensor([0.5, 1.0]).reshape(2, 1, 1)),
        ]
    )
    outgoing = segment(incoming)
    assert outgoing.particles.shape == torch.Size([2, 3, 3, 100, 7])

    # Test improper batching -- this does not obey torch broadcasting rules
    segment = Segment(
        [
            Dipole(
                length=torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1),
                angle=torch.tensor([0.1, 0.2, 0.1]).reshape(1, 3),
            ),
            Drift(length=torch.tensor([0.5, 1.0]).reshape(2, 1)),
        ]
    )
    with pytest.raises(RuntimeError):
        segment(incoming)
