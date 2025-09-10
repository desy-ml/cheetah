import pytest
import torch

import cheetah
from cheetah.track_methods import rotation_matrix


def test_dipole_off():
    """
    Test that a dipole with angle=0 behaves still like a drift.
    """
    dipole = cheetah.Dipole(length=torch.tensor(1.0), angle=torch.tensor(0.0))
    drift = cheetah.Drift(length=torch.tensor(1.0))
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
    )
    outbeam_dipole_off = dipole.track(incoming_beam)
    outbeam_drift = drift.track(incoming_beam)

    dipole.angle = torch.tensor(1.0, device=dipole.angle.device)
    outbeam_dipole_on = dipole.track(incoming_beam)

    assert dipole.name is not None
    assert torch.allclose(outbeam_dipole_off.sigma_x, outbeam_drift.sigma_x)
    assert not torch.allclose(outbeam_dipole_on.sigma_x, outbeam_drift.sigma_x)


def test_dipole_focussing():
    """
    Test that a dipole with focussing moment behaves like a quadrupole.
    """
    dipole = cheetah.Dipole(length=torch.tensor([1.0]), k1=torch.tensor([10.0]))
    quadrupole = cheetah.Quadrupole(length=torch.tensor([1.0]), k1=torch.tensor([10.0]))
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_px=torch.tensor([2e-7]), sigma_py=torch.tensor([2e-7])
    )
    outbeam_dipole_on = dipole.track(incoming_beam)
    outbeam_quadrupole = quadrupole.track(incoming_beam)

    dipole.k1 = torch.tensor([0.0], device=dipole.k1.device)
    outbeam_dipole_off = dipole.track(incoming_beam)

    assert dipole.name is not None
    assert torch.allclose(outbeam_dipole_on.sigma_x, outbeam_quadrupole.sigma_x)
    assert not torch.allclose(outbeam_dipole_off.sigma_x, outbeam_quadrupole.sigma_x)


@pytest.mark.parametrize("DipoleType", [cheetah.Dipole, cheetah.RBend])
def test_dipole_vectorized_execution(DipoleType):
    """
    Test that a dipole with vector dimensions behaves as expected.
    """
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100, energy=torch.tensor(1e9), mu_x=torch.tensor(1e-5)
    )

    # Test vectorisation to generate 3 beam lines
    segment = cheetah.Segment(
        [
            DipoleType(
                length=torch.tensor([0.5, 0.5, 0.5]),
                angle=torch.tensor([0.1, 0.2, 0.1]),
            ),
            cheetah.Drift(length=torch.tensor(0.5)),
        ]
    )
    outgoing = segment.track(incoming)

    assert outgoing.particles.shape == torch.Size([3, 100, 7])
    assert outgoing.mu_x.shape == torch.Size([3])

    # Check that dipole with same bend angle produce same output
    assert torch.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check different angles do make a difference
    assert not torch.allclose(outgoing.particles[0], outgoing.particles[1])

    # Test vectorisation to generate 18 beamlines
    segment = cheetah.Segment(
        [
            cheetah.Dipole(
                length=torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1),
                angle=torch.tensor([0.1, 0.2, 0.1]).reshape(1, 3),
            ),
            cheetah.Drift(length=torch.tensor([0.5, 1.0]).reshape(2, 1, 1)),
        ]
    )
    outgoing = segment.track(incoming)
    assert outgoing.particles.shape == torch.Size([2, 3, 3, 100, 7])

    # Test improper vectorisation -- this does not obey torch broadcasting rules
    segment = cheetah.Segment(
        [
            cheetah.Dipole(
                length=torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1),
                angle=torch.tensor([0.1, 0.2, 0.1]).reshape(1, 3),
            ),
            cheetah.Drift(length=torch.tensor([0.5, 1.0]).reshape(2, 1)),
        ]
    )
    with pytest.raises(RuntimeError):
        segment.track(incoming)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
def test_dipole_drift_kick_drift_tracking(dtype):
    """
    Test that the results of tracking through a dipole with the `"drift_kick_drift"`
    tracking method match the results from Bmad-X.
    """
    incoming = torch.load("tests/resources/bmadx/incoming.pt", weights_only=False).to(
        dtype
    )

    # TODO: See if Bmad-X test dtypes can be cleaned up now that dtype PR was merged
    angle = torch.tensor(20 * torch.pi / 180, dtype=dtype)
    e1 = angle / 2
    e2 = angle - e1
    dipole_cheetah_bmadx = cheetah.Dipole(
        length=torch.tensor(0.5),
        angle=angle,
        dipole_e1=e1,
        dipole_e2=e2,
        tilt=torch.tensor(0.1, dtype=dtype),
        fringe_integral=torch.tensor(0.5),
        fringe_integral_exit=torch.tensor(0.5),
        gap=torch.tensor(0.05, dtype=dtype),
        gap_exit=torch.tensor(0.05, dtype=dtype),
        fringe_at="both",
        fringe_type="linear_edge",
        tracking_method="drift_kick_drift",
        dtype=dtype,
    )
    segment_cheetah_bmadx = cheetah.Segment(elements=[dipole_cheetah_bmadx])

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


def test_drift_kick_drift_zero_angle():
    """
    Test that a dipole with zero angle using the drift_kick_drift tracking method works
    at all.

    There was a bug in the past where a division by zero due to the angle being zero
    resulted in NaN values in the output.
    """
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001", dtype=torch.float64
    )
    dipole = cheetah.Dipole(
        length=torch.tensor(1.0601),
        angle=torch.tensor(0.0),
        tracking_method="drift_kick_drift",
        dtype=torch.float64,
    )

    outgoing_beam = dipole.track(incoming_beam)

    assert not outgoing_beam.particles.isnan().any()


@pytest.mark.parametrize(
    "tracking_method", ["linear", "second_order", "drift_kick_drift"]
)
def test_dipole_tilt_sanity(tracking_method):
    """
    Test that tracking through a tilted dipole and untilting a tilted beam tracked
    through non-tilted dipole gives the same result.
    """
    TILT = 0.52

    # Track non-tilted beam through tilted dipole
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001", dtype=torch.float64
    )

    tilted_dipole = cheetah.Dipole(
        length=torch.tensor(1.0601),
        angle=torch.tensor(1e-3),
        tilt=torch.tensor(TILT),
        fringe_integral=torch.tensor(1e3),
        tracking_method=tracking_method,
    ).to(torch.float64)

    tilted_dipole_outgoing_beam = tilted_dipole.track(incoming_beam)

    # Track tilted beam through non-tilted dipole and then untilt the beam
    tilted_incoming_beam = incoming_beam.clone()
    tilted_incoming_beam.particles = (
        tilted_incoming_beam.particles
        @ rotation_matrix(torch.tensor(TILT, dtype=torch.float64)).T
    )

    non_tilted_dipole = tilted_dipole.clone()
    non_tilted_dipole.tilt = torch.tensor(0.0, dtype=torch.float64)

    non_tilted_dipole_outgoing_beam = non_tilted_dipole.track(tilted_incoming_beam)
    non_tilted_dipole_outgoing_beam.particles = (
        non_tilted_dipole_outgoing_beam.particles
        @ rotation_matrix(torch.tensor(-TILT, dtype=torch.float64)).T
    )

    # Check that the two outgoing beams are equal
    assert torch.allclose(
        tilted_dipole_outgoing_beam.particles, non_tilted_dipole_outgoing_beam.particles
    )
