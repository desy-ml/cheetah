import pytest
import torch

import cheetah


def test_diverging_parameter_beam():
    """
    Test that that a parameter beam with sigma_xp > 0 and sigma_yp > 0 increases in
    size in both dimensions when travelling through a drift section.
    """
    drift = cheetah.Drift(length=torch.tensor(1.0))
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_xp=torch.tensor(2e-7), sigma_yp=torch.tensor(2e-7)
    )
    outgoing_beam = drift.track(incoming_beam)

    assert outgoing_beam.sigma_x > incoming_beam.sigma_x
    assert outgoing_beam.sigma_y > incoming_beam.sigma_y
    assert torch.isclose(outgoing_beam.total_charge, incoming_beam.total_charge)


def test_diverging_particle_beam():
    """
    Test that that a particle beam with sigma_xp > 0 and sigma_yp > 0 increases in
    size in both dimensions when travelling through a drift section.
    """
    drift = cheetah.Drift(length=torch.tensor(1.0))
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=torch.tensor(1000),
        sigma_xp=torch.tensor(2e-7),
        sigma_yp=torch.tensor(2e-7),
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
