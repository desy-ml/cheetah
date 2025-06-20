import pytest
import torch

import cheetah


def test_diverging_parameter_beam():
    """
    Test that that a parameter beam with sigma_px > 0 and sigma_py > 0 increases in
    size in both dimensions when travelling through a drift section.
    """
    drift = cheetah.Drift(length=torch.tensor(1.0))
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
    )
    outgoing_beam = drift.track(incoming_beam)

    assert outgoing_beam.sigma_x > incoming_beam.sigma_x
    assert outgoing_beam.sigma_y > incoming_beam.sigma_y
    assert torch.isclose(outgoing_beam.total_charge, incoming_beam.total_charge)


def test_diverging_particle_beam():
    """
    Test that that a particle beam with sigma_px > 0 and sigma_py > 0 increases in
    size in both dimensions when travelling through a drift section.
    """
    drift = cheetah.Drift(length=torch.tensor(1.0))
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=1_000, sigma_px=torch.tensor(2e-4), sigma_py=torch.tensor(2e-4)
    )
    outgoing_beam = drift.track(incoming_beam)

    assert outgoing_beam.sigma_x > incoming_beam.sigma_x
    assert outgoing_beam.sigma_y > incoming_beam.sigma_y
    assert torch.allclose(
        outgoing_beam.particle_charges, incoming_beam.particle_charges
    )


@pytest.mark.skip(
    reason="Requires rewriting Element and Beam member variables to be buffers."
)
def test_device_like_torch_module():
    """
    Test that when changing the device, Drift reacts like a `torch.nn.Module`.
    """
    # There is no point in running this test, if there aren't two different devices to
    # move between
    if not torch.cuda.is_available():
        return

    element = cheetah.Drift(length=torch.tensor(0.2), device="cuda")

    assert element.length.device.type == "cuda"

    element = element.cpu()

    assert element.length.device.type == "cpu"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_drift_bmadx_tracking(dtype):
    """
    Test that the results of tracking through a drift with the `"bmadx"` tracking method
    match the results from Bmad-X.
    """
    incoming_beam = torch.load(
        "tests/resources/bmadx/incoming.pt", weights_only=False
    ).to(dtype)
    drift = cheetah.Drift(
        length=torch.tensor(1.0), tracking_method="bmadx", dtype=dtype
    )

    # Run tracking
    outgoing_beam = drift.track(incoming_beam)

    # Load reference result computed with Bmad-X
    outgoing_bmadx = torch.load(
        "tests/resources/bmadx/outgoing_drift.pt", weights_only=False
    )

    assert torch.allclose(
        outgoing_beam.particles,
        outgoing_bmadx.to(dtype),
        atol=1e-14 if dtype == torch.float64 else 0.00001,
        rtol=1e-14 if dtype == torch.float64 else 1e-6,
    )


def test_length_as_parameter():
    """Test that the drift length can be set as a `torch.nn.Parameter`."""
    length = torch.tensor(1.0)
    parameter = torch.nn.Parameter(length)

    # Create to equal drifts, one with Tensor, one with Parameter
    drift = cheetah.Drift(length=length)
    drift_parameter = cheetah.Drift(length=parameter)

    incoming = cheetah.ParameterBeam.from_parameters()
    outgoing = drift.track(incoming)
    outgoing_parameter = drift_parameter.track(incoming)

    # Check that all properties of the two outgoing beams are same
    for attribute in outgoing.UNVECTORIZED_NUM_ATTR_DIMS.keys():
        assert torch.allclose(
            getattr(outgoing, attribute), getattr(outgoing_parameter, attribute)
        )


def test_inversion_with_negative_length():
    """
    Tests that tracking through a drift with negative lengths reverts the effect of
    tracking through a positive length drift.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.7)),
            cheetah.Drift(length=torch.tensor(-0.7)),
        ]
    )

    incoming = cheetah.ParticleBeam.from_parameters()
    outgoing = segment.track(incoming)

    # Check that all properties of the two beams are same
    for attribute in outgoing.UNVECTORIZED_NUM_ATTR_DIMS.keys():
        assert torch.allclose(
            getattr(incoming, attribute), getattr(outgoing, attribute)
        )
