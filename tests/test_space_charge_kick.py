import pytest
import torch

import cheetah

def test_charge_deposition():
    """
    Test that the charge deposition is correct for a particle beam. The first test checks that the total charge is preserved, and the second test checks that the charge is deposited in the correct grid cells.
    """
    space_charge_kick = cheetah.SpaceChargeKick(nx=32,ny=32,ns=32,dx=3e-9,dy=3e-9,ds=2e-6)
    incoming_beam = cheetah.ParticleBeam.from_parameters(
        num_particles=torch.tensor(1000),
        sigma_xp=torch.tensor(2e-7),
        sigma_yp=torch.tensor(2e-7),
    )
    total_charge = incoming_beam.total_charge
    space_charge_grid = space_charge_kick.space_charge_deposition(incoming_beam) 

    assert torch.isclose(space_charge_grid.sum() * space_charge_kick.grid_resolution ** 3, torch.tensor(total_charge), atol=1e-12)  # grid_resolution is a parameter of the space charge kick #Total charge is preserved

    # something similar to the read function in the CIC code should be implemented
    assert outgoing_beam.sigma_y > incoming_beam.sigma_y


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