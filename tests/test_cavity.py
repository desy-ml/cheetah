import pytest
import torch

import cheetah


def test_assert_ei_greater_zero():
    """
    Reproduces

    ```
       1127 Ef = (energy + delta_energy) / particle_mass_eV
       1128 Ep = (Ef - Ei) / self.length  # Derivative of the energy
    -> 1129 assert Ei > 0, "Initial energy must be larger than 0"
       1131 alpha = torch.sqrt(eta / 8) / torch.cos(phi) * torch.log(Ef / Ei)
       1133 r11 = torch.cos(alpha) - torch.sqrt(2 / eta) * torch.cos(phi) * torch.sin(alpha)   # noqa: E501

    RuntimeError: Boolean value of Tensor with more than one value is ambiguous
    ```
    """
    cavity = cheetah.Cavity(
        length=torch.tensor([3.0441, 3.0441, 3.0441]),
        voltage=torch.tensor([48198468.0, 48198468.0, 48198468.0]),
        phase=torch.tensor([48198468.0, 48198468.0, 48198468.0]),
        frequency=torch.tensor([2.8560e09, 2.8560e09, 2.8560e09]),
        name="k26_2a",
    )
    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, sigma_x=torch.tensor(1e-5)
    )

    _ = cavity.track(beam)


@pytest.mark.parametrize(
    "voltage",
    [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1e6]), torch.tensor([1e6, 1e6])],
)
def test_vectorized_cavity_zero_voltage(voltage):
    """
    Tests that a vectorised cavity with zero voltage does not produce NaNs and that
    zero voltage can be vectorised with non-zero voltage.

    This was a bug introduced during the vectorisation of Cheetah, when the special
    case of zero was removed and the `_cavity_rmatrix` method was also used in the case
    of zero voltage. The latter produced NaNs in the transfer matrix when the voltage
    is zero.
    """
    cavity = cheetah.Cavity(
        length=torch.tensor([3.0441, 3.0441]),
        voltage=voltage,
        phase=torch.tensor([-0.0, -0.0]),
        frequency=torch.tensor([2.8560e09, 2.8560e09]),
        name="k27_1a",
        dtype=torch.float64,
    )
    incoming = cheetah.ParameterBeam.from_parameters(
        mu_x=torch.tensor(0.0),
        mu_px=torch.tensor(0.0),
        mu_y=torch.tensor(0.0),
        mu_py=torch.tensor(0.0),
        sigma_x=torch.tensor(4.8492e-06),
        sigma_px=torch.tensor(1.5603e-07),
        sigma_y=torch.tensor(4.1209e-07),
        sigma_py=torch.tensor(1.1035e-08),
        sigma_tau=torch.tensor(1.0000e-10),
        sigma_p=torch.tensor(1.0000e-06),
        energy=torch.tensor(8.0000e09),
        total_charge=torch.tensor(0.0),
        dtype=torch.float64,
    )

    outgoing = cavity.track(incoming)

    assert not torch.isnan(cavity.transfer_map(incoming.energy, incoming.species)).any()

    assert not torch.isnan(outgoing.sigma_x).any()
    assert not torch.isnan(outgoing.sigma_y).any()
    assert not torch.isnan(outgoing.beta_x).any()
    assert not torch.isnan(outgoing.beta_y).any()
