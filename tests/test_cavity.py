import pytest
import torch

import cheetah


@pytest.mark.parametrize("beam_type", [cheetah.ParticleBeam, cheetah.ParameterBeam])
@pytest.mark.parametrize("cavity_type", ["standing_wave", "traveling_wave"])
def test_active_length_matches_explicit_end_drifts(beam_type, cavity_type):
    """An explicit active length must track like drift-cavity-drift geometry."""
    incoming = beam_type.from_parameters(
        energy=torch.tensor(6e6, dtype=torch.float64),
        dtype=torch.float64,
        **({"num_particles": 32} if beam_type is cheetah.ParticleBeam else {}),
    )
    length = torch.tensor(3.095244, dtype=torch.float64)
    active_length = torch.tensor([3.044111093137255, 3.0], dtype=torch.float64)
    settings = {
        "voltage": torch.tensor(58e6, dtype=torch.float64),
        "frequency": torch.tensor(2.856e9, dtype=torch.float64),
        "phase": torch.tensor(1.1, dtype=torch.float64),
        "cavity_type": cavity_type,
    }
    cavity = cheetah.Cavity(length=length, active_length=active_length, **settings)
    active = cheetah.Cavity(length=active_length, **settings)
    drift = cheetah.Drift(length=(length - active_length) / 2.0)
    expected = drift.track(active.track(drift.track(incoming)))
    actual = cavity.track(incoming)
    if beam_type is cheetah.ParticleBeam:
        torch.testing.assert_close(actual.particles, expected.particles)
    else:
        torch.testing.assert_close(actual.mu, expected.mu)
        torch.testing.assert_close(actual.cov, expected.cov)
    torch.testing.assert_close(actual.energy, expected.energy)
    torch.testing.assert_close(actual.s, expected.s)
    torch.testing.assert_close(actual.s, (incoming.s + length).expand_as(actual.s))
    expected_matrix = (
        drift.first_order_transfer_map(expected.energy, incoming.species)
        @ active.first_order_transfer_map(incoming.energy, incoming.species)
        @ drift.first_order_transfer_map(incoming.energy, incoming.species)
    )
    torch.testing.assert_close(
        cavity.first_order_transfer_map(incoming.energy, incoming.species),
        expected_matrix,
    )


@pytest.mark.parametrize("active_length", [0.0, -0.1, 3.2, float("nan"), float("inf")])
def test_invalid_active_length(active_length):
    """Reject active RF regions that cannot fit inside a finite cavity."""
    cavity = cheetah.Cavity(
        length=torch.tensor(3.095244),
        active_length=torch.tensor(active_length),
    )
    with pytest.raises(ValueError, match="active_length"):
        cavity.track(cheetah.ParameterBeam.from_parameters())


def test_active_length_gradients_and_serialization(tmp_path):
    """Active and total lengths must remain differentiable and survive persistence."""
    species = cheetah.Species("electron", dtype=torch.float64)

    def matrix(lengths):
        """Evaluate a transverse map with independent total and active lengths."""
        cavity = cheetah.Cavity(
            length=lengths[0],
            active_length=lengths[1],
            voltage=torch.tensor(58e6, dtype=torch.float64),
            frequency=torch.tensor(2.856e9, dtype=torch.float64),
            cavity_type="traveling_wave",
        )
        return cavity.first_order_transfer_map(
            torch.tensor(6e6, dtype=torch.float64), species
        )[:4, :4]

    lengths = torch.tensor(
        [3.095244, 3.044111093137255], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(matrix, (lengths,))
    cavity = cheetah.Cavity(
        length=lengths[0].detach(), active_length=lengths[1].detach(), name="cavity"
    )
    segment = cheetah.Segment([cavity])
    path = tmp_path / "cavity.json"
    segment.to_lattice_json(path)
    restored = cheetah.Segment.from_lattice_json(path, dtype=torch.float64).cavity
    for other in (cavity.clone(), restored):
        torch.testing.assert_close(other.active_length, cavity.active_length)
        torch.testing.assert_close(other.length, cavity.length)


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
    ("voltage", "phase"),
    [
        (torch.tensor(0.0), torch.tensor([-90.0, 90.0])),
        (torch.tensor([0.0, 1e6]), torch.tensor([[-90.0], [0.0], [90.0], [180.0]])),
        (torch.tensor(1e6), torch.tensor([0.0, 180.0])),
    ],
    ids=["off", "mixed", "on"],
)
@pytest.mark.parametrize("cavity_type", ["standing_wave", "traveling_wave"])
def test_vectorized_inactive_cavity(cavity_type, voltage, phase):
    """
    Tests that a vectorised cavity with zero voltage or off-crest phase does not produce
    NaNs and that switched-off cavities can be vectorized with switched-on.

    This was a bug introduced during the vectorisation of Cheetah, when the special
    case of zero was removed and the `_cavity_rmatrix` method was also used in the case
    of zero voltage or off-crest phase. The latter produced NaNs in the transfer matrix.
    """
    cavity = cheetah.Cavity(
        cavity_type=cavity_type,
        length=torch.tensor(3.0441),
        voltage=voltage,
        phase=phase,
        frequency=torch.tensor(2.8560e09),
    ).to(torch.float64)
    incoming = cheetah.ParameterBeam.from_parameters(
        sigma_x=torch.tensor(4.8492e-06),
        sigma_px=torch.tensor(1.5603e-07),
        sigma_y=torch.tensor(4.1209e-07),
        sigma_py=torch.tensor(1.1035e-08),
        sigma_tau=torch.tensor(1.0000e-10),
        sigma_p=torch.tensor(1.0000e-06),
        energy=torch.tensor(8.0000e09),
        total_charge=torch.tensor(0.0),
    ).to(torch.float64)

    outgoing = cavity.track(incoming)

    assert (
        not cavity.first_order_transfer_map(incoming.energy, incoming.species)
        .isnan()
        .any()
    )

    assert not outgoing.sigma_x.isnan().any()
    assert not outgoing.sigma_y.isnan().any()
    assert not outgoing.beta_x.isnan().any()
    assert not outgoing.beta_y.isnan().any()


def test_traveling_wave_conserves_normalized_transverse_emittance():
    """Exact-momentum damping must conserve normalized emittance at low energy."""
    incoming = cheetah.ParameterBeam.from_twiss(
        beta_x=torch.tensor(25.4731864, dtype=torch.float64),
        alpha_x=torch.tensor(-4.2158781, dtype=torch.float64),
        beta_y=torch.tensor(5.5114672, dtype=torch.float64),
        alpha_y=torch.tensor(-3.8090071, dtype=torch.float64),
        emittance_x=torch.tensor(1e-6, dtype=torch.float64),
        emittance_y=torch.tensor(2e-6, dtype=torch.float64),
        energy=torch.tensor(6e6, dtype=torch.float64),
        dtype=torch.float64,
    )
    cavity = cheetah.Cavity(
        length=torch.tensor(3.0441111, dtype=torch.float64),
        voltage=torch.tensor([-2e6, 0.0, 58e6], dtype=torch.float64),
        phase=torch.tensor([[0.0], [20.0]], dtype=torch.float64),
        frequency=torch.tensor(2.856e9, dtype=torch.float64),
        cavity_type="traveling_wave",
    )
    outgoing = cavity.track(incoming)
    for name in ("normalized_emittance_x", "normalized_emittance_y"):
        expected = getattr(incoming, name).expand(2, 3)
        torch.testing.assert_close(
            getattr(outgoing, name), expected, rtol=1e-10, atol=0.0
        )


@pytest.mark.parametrize("voltage_MV", [-2.0, 0.0, 58.0])
@pytest.mark.parametrize("active_length", [None, 3.0])
def test_traveling_wave_transverse_gradients(voltage_MV, active_length):
    """Transverse transport must have correct gradients through zero energy gain."""
    species = cheetah.Species("electron", dtype=torch.float64)

    def transverse_matrix(parameters):
        """Use MeV-scaled parameters for well-conditioned finite-difference checks."""
        length, voltage, energy, phase = parameters.unbind()
        cavity = cheetah.Cavity(
            length=length,
            active_length=(
                length.new_tensor(active_length) if active_length is not None else None
            ),
            voltage=voltage * 1e6,
            phase=phase,
            frequency=torch.tensor(2.856e9, dtype=torch.float64),
            cavity_type="traveling_wave",
        )
        return cavity.first_order_transfer_map(energy * 1e6, species)[:2, :2]

    parameters = torch.tensor(
        [3.0441111, voltage_MV, 6.0, 1.1], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(transverse_matrix, (parameters,))
    if voltage_MV == 0.0:
        torch.testing.assert_close(
            transverse_matrix(parameters),
            torch.tensor([[1.0, 3.0441111], [0.0, 1.0]], dtype=torch.float64),
        )


@pytest.mark.parametrize(("energy", "voltage"), [(0.5e6, 1e6), (6e6, -5.5e6)])
def test_traveling_wave_rejects_energies_below_rest_energy(energy, voltage):
    """Entrance and exit momenta must both correspond to propagating particles."""
    cavity = cheetah.Cavity(
        length=torch.tensor(1.0, dtype=torch.float64),
        voltage=torch.tensor(voltage, dtype=torch.float64),
        cavity_type="traveling_wave",
    )
    with pytest.raises(AssertionError, match="must exceed the rest energy"):
        cavity.first_order_transfer_map(
            torch.tensor(energy, dtype=torch.float64),
            cheetah.Species("electron", dtype=torch.float64),
        )
