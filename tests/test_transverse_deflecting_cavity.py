import pytest
import torch

import cheetah


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
        tracking_method="drift_kick_drift",
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
        tracking_method="drift_kick_drift",
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
        tracking_method="drift_kick_drift",
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
        tracking_method="drift_kick_drift",
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == torch.Size([4, 3, 2, 2])


def test_tdc_benchmark():
    cavity = cheetah.TransverseDeflectingCavity(
        length=torch.tensor(0.2),
        voltage=torch.tensor(1.0e6),
        phase=torch.tensor(0.0),
        frequency=torch.tensor(1.0e9)
    )
    test_beam = cheetah.ParticleBeam(
        torch.tensor([2e-3,3e-3,-3e-3,-1e-3,-2e-3, 2e-3, 1.0]).unsqueeze(0),
        energy=torch.tensor(4.0e7)
    )
    assert torch.allclose(
        cavity.track(test_beam).particles.flatten()[:-1],
        torch.tensor([2.705670627614420e-03, 4.047421479988640e-03, -3.200281391645270e-03,
        -1.000000000000000e-03, 1.998582178711370e-03, -7.955332028185950e-04],
       ), atol=1e-2
    )


def test_transverse_deflecting_cavity_split():
    """
    Test that splitting a TDC into smaller segments works as expected.
    """
    tdc = cheetah.TransverseDeflectingCavity(
        length=torch.tensor(1.0),
        voltage=torch.tensor(1e7),
        phase=torch.tensor(0.4),
        frequency=torch.tensor(1e9),
        tracking_method="drift_kick_drift",
        num_steps=11,
    )

    segments = tdc.split(resolution=torch.tensor(0.5))

    assert len(segments) == 2
    for segment in segments:
        assert isinstance(segment, cheetah.TransverseDeflectingCavity)
        assert torch.isclose(
            segment.length, torch.tensor(0.5), rtol=1e-5
        )
        #assert torch.equal(segment.voltage, tdc.voltage)
        assert torch.equal(segment.phase, tdc.phase)
        assert torch.equal(segment.frequency, tdc.frequency)
        assert segment.num_steps == 5
        assert segment.tracking_method == tdc.tracking_method

    # test to make sure that tracking through the split segments gives the same result
    # as tracking through the original segment
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=10,
        sigma_px=torch.tensor(2e-7),
        sigma_py=torch.tensor(2e-7),
        energy=torch.tensor(50e6),
    )

    outgoing_beam_full = tdc.track(incoming_beam)
    outgoing_beam_split = incoming_beam
    for segment in segments:
        outgoing_beam_split = segment.track(outgoing_beam_split)

    assert torch.allclose(
        outgoing_beam_full.particles,
        outgoing_beam_split.particles,
        rtol=1e-2,
    )
    
