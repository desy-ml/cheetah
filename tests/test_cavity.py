import torch

import cheetah


def test_assert_ei_greater_zero():
    """
    Reproduces

    ```
       1127 Ef = (energy + delta_energy) / electron_mass_eV
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
        num_particles=100_000, sigma_x=torch.tensor([1e-5])
    )

    _ = cavity.track(beam)
