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


@pytest.mark.parametrize("DipoleType", [Dipole, RBend])
def test_dipole_batched_execution(DipoleType):
    """
    Test that a dipole with batch dimensions behaves as expected.
    """
    batch_shape = torch.Size([6])
    incoming = ParticleBeam.from_parameters(
        num_particles=torch.tensor(1_000_000),
        energy=torch.tensor([1e9]),
        mu_x=torch.tensor([1e-5]),
    ).broadcast(batch_shape)
    segment = Segment(
        [
            DipoleType(
                length=torch.tensor([0.5, 0.5, 0.5]),
                angle=torch.tensor([0.1, 0.2, 0.1]),
            ).broadcast((2,)),
            Drift(length=torch.tensor([0.5])).broadcast(batch_shape),
        ]
    )
    outgoing = segment(incoming)

    # Check that dipole with same bend angle produce same output
    assert torch.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check different angles do make a difference
    assert not torch.allclose(outgoing.particles[0], outgoing.particles[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dipole_bmadx_tracking(dtype):
    """
    Test that the results of tracking through a dipole with the `"bmadx"` tracking
    method match the results from Bmad-X.
    """
    incoming = torch.load("tests/resources/bmadx/incoming.pt", weights_only=False).to(
        dtype
    )

    angle = torch.tensor([20 * torch.pi / 180], dtype=dtype)
    e1 = angle / 2
    e2 = angle - e1
    dipole_cheetah_bmadx = Dipole(
        length=torch.tensor([0.5]),
        angle=angle,
        e1=e1,
        e2=e2,
        tilt=torch.tensor([0.1], dtype=dtype),
        fringe_integral=torch.tensor([0.5]),
        fringe_integral_exit=torch.tensor([0.5]),
        gap=torch.tensor([0.05], dtype=dtype),
        gap_exit=torch.tensor([0.05], dtype=dtype),
        fringe_at="both",
        fringe_type="linear_edge",
        tracking_method="bmadx",
        dtype=dtype,
    )
    segment_cheetah_bmadx = Segment(elements=[dipole_cheetah_bmadx])

    outgoing_cheetah_bmadx = segment_cheetah_bmadx.track(incoming)

    # Load reference result computed with Bmad-X
    outgoing_bmadx = torch.load(
        "tests/resources/bmadx/outgoing_dipole.pt", weights_only=False
    )

    assert torch.allclose(
        outgoing_cheetah_bmadx.particles,
        outgoing_bmadx.to(dtype),
        rtol=1e-14 if dtype == torch.float64 else 0.00001,
        atol=1e-14 if dtype == torch.float64 else 1e-6,
    )
