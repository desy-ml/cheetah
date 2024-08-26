import torch
from scipy.constants import physical_constants

from cheetah import Dipole, Drift, ParameterBeam, ParticleBeam, Quadrupole, Segment
from cheetah.utils.bmadx import cheetah_to_bmad_coords


def test_dipole_off():
    """
    Test that a dipole with angle=0 behaves still like a drift.
    """
    dipole = Dipole(length=torch.tensor([1.0]), angle=torch.tensor([0.0]))
    drift = Drift(length=torch.tensor([1.0]))
    incoming_beam = ParameterBeam.from_parameters(
        sigma_px=torch.tensor([2e-7]), sigma_py=torch.tensor([2e-7])
    )
    outbeam_dipole_off = dipole(incoming_beam)
    outbeam_drift = drift(incoming_beam)

    dipole.angle = torch.tensor([1.0], device=dipole.angle.device)
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


def test_dipole_batched_execution():
    """
    Test that a dipole with batch dimensions behaves as expected.
    """
    batch_shape = torch.Size([3])
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1000000),
        energy=torch.tensor([1e9]),
        mu_x=torch.tensor([1e-5]),
    ).broadcast(batch_shape)
    segment = Segment(
        [
            Dipole(
                length=torch.tensor([0.5, 0.5, 0.5]),
                angle=torch.tensor([0.1, 0.2, 0.1]),
            ),
            Drift(length=torch.tensor([0.5])).broadcast(batch_shape),
        ]
    )
    outgoing = segment(incoming)

    # Check that dipole with same bend angle produce same output
    assert torch.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check different angles do make a difference
    assert not torch.allclose(outgoing.particles[0], outgoing.particles[1])


def test_dipole_bmadx_tracking():
    """
    Test that the results of tracking through a dipole with the `"bmadx"` tracking
    method match the results from Bmad-X.
    """
    incoming = torch.load("tests/resources/bmadx/incoming_beam.pt")
    mc2 = torch.tensor(
        physical_constants["electron mass energy equivalent in MeV"][0] * 1e6,
        dtype=torch.float64,
    )
    _, p0c_particle = cheetah_to_bmad_coords(incoming.particles, incoming.energy, mc2)
    p0c = 1 * p0c_particle
    angle = torch.tensor([20 * torch.pi / 180], dtype=torch.float64)
    e1 = angle / 2
    e2 = angle - e1
    dipole_cheetah_bmadx = Dipole(
        length=torch.tensor([0.5]),
        p0c=p0c,
        angle=angle,
        e1=e1,
        e2=e2,
        tilt=torch.tensor([0.1], dtype=torch.float64),
        fringe_integral=torch.tensor([0.5]),
        fringe_integral_exit=torch.tensor([0.5]),
        gap=torch.tensor([0.05], dtype=torch.float64),
        gap_exit=torch.tensor([0.05], dtype=torch.float64),
        fringe_at="both",
        fringe_type="linear_edge",
        tracking_method="bmadx",
        dtype=torch.float64,
    )
    segment_cheetah_bmadx = Segment(elements=[dipole_cheetah_bmadx])

    outgoing_cheetah_bmadx = segment_cheetah_bmadx.track(incoming)

    # Load reference result computed with Bmad-X
    outgoing_bmadx = torch.load("tests/resources/bmadx/outgoing_bmadx_dipole.pt")

    assert torch.allclose(
        outgoing_cheetah_bmadx.particles, outgoing_bmadx, atol=1e-14, rtol=1e-14
    )
