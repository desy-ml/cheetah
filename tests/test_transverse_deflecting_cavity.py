import pytest
import torch

import cheetah


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transverse_deflecting_cavity_bmadx_tracking(dtype):
    """
    Test that the results of tracking through a TDC with the `"bmadx"` tracking method
    match the results from Bmad-X.
    """
    incoming_beam = torch.load(
        "tests/resources/bmadx/incoming.pt", weights_only=False
    ).to(dtype)
    tdc = cheetah.TransverseDeflectingCavity(
        length=torch.tensor(1.0),
        voltage=torch.tensor(1e7),
        phase=torch.tensor(0.2, dtype=dtype),
        frequency=torch.tensor(1e9),
        tracking_method="bmadx",
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


def test_transverse_deflecting_cavity_energy_length_vectorization():
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
        tracking_method="bmadx",
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == torch.Size([3, 2])


def test_transverse_deflecting_cavity_energy_phase_vectorization():
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
        tracking_method="bmadx",
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == torch.Size([3, 2])


def test_transverse_deflecting_cavity_energy_frequency_vectorization():
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
        tracking_method="bmadx",
    )

    _ = tdc3.track(incoming_beam)

    assert _.particles.shape[:-2] == torch.Size([3, 2])


def test_transverse_deflecting_cavity_all_parameters_vectorization():
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
        tracking_method="bmadx",
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == torch.Size([4, 3, 2, 2])


def test_tracking_inactive_in_segment():
    """
    Test that tracking through a `Segment` that contains an inactive
    `TransverseDeflectingCavity` does not throw an exception. This was an issue in #290.
    """
    segment = cheetah.Segment(
        elements=[cheetah.TransverseDeflectingCavity(length=torch.tensor(1.0))]
    )
    beam = cheetah.ParticleBeam.from_parameters()

    segment.track(beam)
