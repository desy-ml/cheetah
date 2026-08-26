import pytest
import torch

import cheetah


@pytest.mark.parametrize("tracking_method", ["linear", "drift_kick_drift"])
def test_transverse_deflecting_cavity_off(tracking_method):
    """
    Test that tracking through a TDC with zero voltage is equivalent to tracking
    through a drift of the same length.
    """

    incoming = cheetah.ParticleBeam.from_parameters(energy=torch.tensor(10e6))
    length = torch.tensor(0.5)
    tdc = cheetah.TransverseDeflectingCavity(
        length=length,
        voltage=torch.tensor(0.0),
        frequency=torch.tensor(1.3e9),
        phase=torch.tensor(0.0),
        tracking_method=tracking_method,
        num_steps=1,  # Made explicit to fix the number of drifts needed below.
    )

    # Drift-kick-drift tracking applies two half drifts which are not identical to one
    # full length drift. This behaviour is reproduced here. Irrelevant for linear
    # tracking.
    half_drift = cheetah.Drift(length=length / 2.0, tracking_method=tracking_method)
    full_drift = cheetah.Segment(elements=[half_drift, half_drift])

    outgoing_tdc = tdc.track(incoming)
    outgoing_drift = full_drift.track(incoming)

    assert torch.allclose(outgoing_tdc.particles, outgoing_drift.particles)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
def test_transverse_deflecting_cavity_drift_kick_drift_tracking(dtype):
    """
    Test that the results of tracking through a TDC with the `"drift_kick_drift"`
        tracking method match the results from Bmad-X.
    """
    incoming_beam = torch.load(
        "tests/resources/bmadx/incoming.pt", weights_only=False
    ).to(dtype)
    tdc = cheetah.TransverseDeflectingCavity(
        length=torch.tensor(1.0, dtype=dtype),
        voltage=torch.tensor(1e7, dtype=dtype),
        phase=torch.tensor(0.2, dtype=dtype),
        frequency=torch.tensor(1e9, dtype=dtype),
        tracking_method="drift_kick_drift",
        dtype=dtype,
    )

    # Run tracking
    outgoing_beam = tdc.track(incoming_beam)

    # Load reference result computed with Bmad-X
    outgoing_bmadx = torch.load(
        "tests/resources/bmadx/outgoing_transverse_deflecting_cavity.pt",
        weights_only=False,
    )

    assert torch.allclose(
        outgoing_beam.particles,
        outgoing_bmadx.to(dtype),
        atol=1e-14 if dtype == torch.float64 else 0.00001,
        rtol=1e-14 if dtype == torch.float64 else 1e-6,
    )


@pytest.mark.parametrize("tracking_method", ["linear", "drift_kick_drift"])
def test_transverse_deflecting_cavity_energy_length_vectorization(tracking_method):
    """
    Test that vectorised tracking through a TDC throws now exception and outputs the
    correct shape, when the input beam's energy and the TDC's length are vectorised.
    """
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        sigma_px=torch.tensor(2e-7),
        sigma_py=torch.tensor(2e-7),
        energy=torch.tensor([50e6, 60e6]),
    )
    tdc = cheetah.TransverseDeflectingCavity(
        length=torch.tensor(1.0),
        voltage=torch.tensor([[1e7], [2e7], [3e7]]),
        phase=torch.tensor(0.4),
        frequency=torch.tensor(1e9),
        tracking_method=tracking_method,
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == torch.Size([3, 2])


@pytest.mark.parametrize("tracking_method", ["linear", "drift_kick_drift"])
def test_transverse_deflecting_cavity_energy_phase_vectorization(tracking_method):
    """
    Test that vectorised tracking through a TDC throws now exception and outputs the
    correct shape, when the input beam's energy and the TDC's phase are vectorised.
    """
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        sigma_px=torch.tensor(2e-7),
        sigma_py=torch.tensor(2e-7),
        energy=torch.tensor([50e6, 60e6]),
    )
    tdc = cheetah.TransverseDeflectingCavity(
        length=torch.tensor(1.0),
        voltage=torch.tensor(1e7),
        phase=torch.tensor([[0.6], [0.5], [0.4]]),
        frequency=torch.tensor(1e9),
        tracking_method=tracking_method,
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == torch.Size([3, 2])


@pytest.mark.parametrize("tracking_method", ["linear", "drift_kick_drift"])
def test_transverse_deflecting_cavity_energy_frequency_vectorization(tracking_method):
    """
    Test that vectorised tracking through a TDC throws now exception and outputs the
    correct shape, when the input beam's energy and the TDC's frequency are vectorised.
    """
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        sigma_px=torch.tensor(2e-7),
        sigma_py=torch.tensor(2e-7),
        energy=torch.tensor([50e6, 60e6]),
    )
    tdc3 = cheetah.TransverseDeflectingCavity(
        length=torch.tensor(1.0),
        voltage=torch.tensor(1e7),
        phase=torch.tensor(0.4),
        frequency=torch.tensor([[1e9], [2e9], [3e9]]),
        tracking_method=tracking_method,
    )

    _ = tdc3.track(incoming_beam)

    assert _.particles.shape[:-2] == torch.Size([3, 2])


@pytest.mark.parametrize("tracking_method", ["linear", "drift_kick_drift"])
def test_transverse_deflecting_cavity_all_parameters_vectorization(tracking_method):
    """
    Test that vectorised tracking through a TDC throws now exception and outputs the
    correct shape, when all parameters are vectorised.
    """
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10_000,
        sigma_px=torch.tensor(2e-7),
        sigma_py=torch.tensor(2e-7),
        energy=torch.tensor([50e6, 60e6]),
    )
    tdc = cheetah.TransverseDeflectingCavity(
        length=torch.tensor(1.0),
        voltage=torch.ones([4, 1, 1, 1]) * 1e7,
        phase=torch.ones([1, 3, 1, 1]) * 0.4,
        frequency=torch.ones([1, 1, 2, 1]) * 1e9,
        tracking_method=tracking_method,
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == torch.Size([4, 3, 2, 2])
