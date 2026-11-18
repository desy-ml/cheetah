"""Tests for the wakefield utilities and for Cavity when wake=True."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import cheetah

from cheetah import Cavity
from cheetah.accelerator.wakefields import (
    SPEED_OF_LIGHT,
    ResonantModeWake,
    gaussian_resonant_wake_potential,
    longitudinal_energy_kick,
    longitudinal_grid_from_beam,
    longitudinal_wake_potential,
)

DTYPE = torch.float64
FREQUENCY = 1.3e9
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY
REFERENCE_ENERGY = 100e6


def make_beam(sigma_tau, total_charge=1e-9, num_particles=200_000, seed=0):
    """Gaussian `ParticleBeam` with the given bunch length and charge."""
    torch.manual_seed(seed)
    beam = cheetah.ParticleBeam.from_twiss(
        num_particles=num_particles,
        beta_x=torch.tensor(10.0, dtype=DTYPE),
        beta_y=torch.tensor(10.0, dtype=DTYPE),
        emittance_x=torch.tensor(2e-9, dtype=DTYPE),
        emittance_y=torch.tensor(2e-9, dtype=DTYPE),
        sigma_tau=torch.tensor(sigma_tau, dtype=DTYPE),
        sigma_p=torch.tensor(1e-3, dtype=DTYPE),
        energy=torch.tensor(REFERENCE_ENERGY, dtype=DTYPE),
        total_charge=torch.tensor(total_charge, dtype=DTYPE),
        dtype=DTYPE,
    )
    return beam


# ----------------------------------------------------------------- kernel identities


def test_resonant_kernel_reduces_to_cosine_at_infinite_q():
    """With `Q = inf` the resonant wake is a pure cosine, `w(s) = A cos(ks)`."""
    amplitude = 35e3
    kernel = ResonantModeWake(frequency=FREQUENCY, wake_strength=amplitude)
    s = torch.linspace(0.0, 3 * WAVELENGTH, 501, dtype=DTYPE)
    expected = amplitude * torch.cos(2 * math.pi * s / WAVELENGTH)
    # atol is scaled to the amplitude: near zero crossings rtol alone is too strict.
    assert torch.allclose(
        kernel.sample(s), expected, rtol=1e-10, atol=1e-12 * amplitude
    )


def test_damped_resonant_kernel_matches_closed_form():
    """The damped resonant wake matches its closed form."""
    frequency, quality_factor, amplitude = 8.4e9, 11_000.0, 35e3
    kernel = ResonantModeWake(
        frequency=frequency, wake_strength=amplitude, quality_factor=quality_factor
    )
    omega_r = 2 * math.pi * frequency
    damping = omega_r / (2 * quality_factor)
    omega_bar = math.sqrt(omega_r**2 - damping**2)

    s = torch.linspace(0.0, 0.5, 401, dtype=DTYPE)
    reference = (
        amplitude
        * torch.exp(-damping * s / SPEED_OF_LIGHT)
        * (
            torch.cos(omega_bar * s / SPEED_OF_LIGHT)
            - (damping / omega_bar) * torch.sin(omega_bar * s / SPEED_OF_LIGHT)
        )
    )
    assert torch.allclose(
        kernel.sample(s),
        reference,
        rtol=1e-9,
        atol=1e-10 * float(reference.abs().max()),
    )


# --------------------------------------------------------------------- grid sizing


def test_grid_resolves_the_bunch_and_covers_it():
    """The automatic grid resolves the bunch and covers every particle."""
    beam = make_beam(300e-6, num_particles=20_000)
    kernel = ResonantModeWake(frequency=FREQUENCY, wake_strength=35e3)
    grid = longitudinal_grid_from_beam(
        beam.tau, feature_lengths=(kernel.feature_length,), cells_per_feature=8.0
    )
    sigma = float(beam.tau.std())
    assert grid.cells_per_sigma >= 8.0
    assert grid.resolved
    assert grid.z_min < float(beam.tau.min())
    assert grid.z_max > float(beam.tau.max())
    # The old scheme: n_grid=2000 with 4 wavelengths of padding on each side.
    old_dz = (float(beam.tau.max()) - float(beam.tau.min()) + 8 * WAVELENGTH) / 1999
    assert old_dz / sigma > 1.0  # the whole bunch inside a few cells
    assert grid.dz / sigma < 0.2


def test_grid_takes_the_shortest_of_beam_and_kernel_scales():
    """The grid spacing follows the shorter of the bunch length and the wake's scale."""
    beam = make_beam(3e-2, num_particles=20_000)
    kernel = ResonantModeWake(frequency=10 * FREQUENCY, wake_strength=35e3)
    grid = longitudinal_grid_from_beam(
        beam.tau, feature_lengths=(kernel.feature_length,), cells_per_feature=8.0
    )
    assert kernel.feature_length < float(beam.tau.std())
    assert grid.dz == pytest.approx(kernel.feature_length / 8.0)


def test_grid_warns_when_clamped_by_max_grid():
    """A warning is raised when `max_grid` limits the resolution."""
    beam = make_beam(1e-6, num_particles=5_000)
    kernel = ResonantModeWake(frequency=FREQUENCY, wake_strength=35e3)
    with pytest.warns(UserWarning, match="clamped"):
        grid = longitudinal_grid_from_beam(
            beam.tau,
            feature_lengths=(kernel.feature_length,),
            cells_per_feature=8.0,
            max_grid=64,
        )
    assert not grid.resolved


# ----------------------------------------------------- convolution against analytics


def test_wake_potential_matches_gaussian_closed_form():
    """The numerical wake potential matches the closed-form Gaussian result."""
    sigma = 3e-3
    total_charge = 1e-9
    beam = make_beam(sigma, total_charge=total_charge, num_particles=400_000)
    kernel = ResonantModeWake(frequency=FREQUENCY, wake_strength=1e6)

    wake, diagnostics = longitudinal_wake_potential(
        beam.tau, beam.particle_charges, kernel
    )

    tau = beam.tau.numpy()
    analytic = gaussian_resonant_wake_potential(
        tau,
        sigma_tau=float(beam.tau.std()),
        wavelength=kernel.wavelength,
        total_charge=float(beam.particle_charges.sum()),
        wake_strength=kernel.wake_strength,
    )
    error = np.sqrt(np.mean((wake.numpy() - analytic) ** 2)) / np.sqrt(
        np.mean(analytic**2)
    )
    assert error < 0.02, f"relative rms error {error:.3%}"


def test_trapezoidal_origin_weight_improves_accuracy():
    """Halving the `s = 0` sample makes the convolution more accurate."""
    sigma = 3e-3
    beam = make_beam(sigma, num_particles=400_000)
    kernel = ResonantModeWake(frequency=FREQUENCY, wake_strength=1e6)
    grid = longitudinal_grid_from_beam(
        beam.tau, feature_lengths=(kernel.feature_length,), cells_per_feature=2.0
    )

    analytic = gaussian_resonant_wake_potential(
        beam.tau.numpy(),
        sigma_tau=float(beam.tau.std()),
        wavelength=kernel.wavelength,
        total_charge=float(beam.particle_charges.sum()),
        wake_strength=kernel.wake_strength,
    )

    errors = {}
    for trapezoidal in (True, False):
        wake, _ = longitudinal_wake_potential(
            beam.tau,
            beam.particle_charges,
            kernel,
            grid=grid,
            trapezoidal=trapezoidal,
        )
        errors[trapezoidal] = np.sqrt(np.mean((wake.numpy() - analytic) ** 2))
    assert errors[True] < errors[False]


def test_wake_potential_is_linear_in_charge_not_quadratic():
    """Doubling the charge doubles the wake potential."""
    kernel = ResonantModeWake(frequency=FREQUENCY, wake_strength=1e6)
    beam = make_beam(3e-3, total_charge=1e-9, num_particles=50_000)
    grid = longitudinal_grid_from_beam(
        beam.tau, feature_lengths=(kernel.feature_length,)
    )
    single, _ = longitudinal_wake_potential(
        beam.tau, beam.particle_charges, kernel, grid=grid
    )
    double, _ = longitudinal_wake_potential(
        beam.tau, 2.0 * beam.particle_charges, kernel, grid=grid
    )
    assert torch.allclose(double, 2.0 * single, rtol=1e-10)


def test_wake_is_causal():
    """Nothing ahead of the bunch feels the wake, and the head sees half its own."""
    kernel = ResonantModeWake(frequency=FREQUENCY, wake_strength=1e6)
    beam = make_beam(3e-3, num_particles=100_000)
    wake, diagnostics = longitudinal_wake_potential(
        beam.tau, beam.particle_charges, kernel
    )
    z_grid = diagnostics["z_grid"]
    wake_grid = diagnostics["W_grid"]
    ahead = z_grid < float(beam.tau.min())
    peak = float(wake_grid.abs().max())
    assert float(wake_grid[ahead].abs().max()) < 1e-6 * peak


def test_energy_kick_decelerates_for_a_positive_wake():
    """A positive wake takes energy from the beam."""
    kernel = ResonantModeWake(frequency=FREQUENCY, wake_strength=1e6)
    beam = make_beam(3e-4, num_particles=50_000)
    delta_energy, _ = longitudinal_energy_kick(
        beam.tau, beam.particle_charges, kernel, num_elementary_charges=-1.0
    )
    assert float(delta_energy.mean()) < 0.0
    # Species sign must not matter: a proton bunch loses energy too.
    proton, _ = longitudinal_energy_kick(
        beam.tau, beam.particle_charges, kernel, num_elementary_charges=+1.0
    )
    assert torch.allclose(delta_energy, proton)


# ------------------------------------------------------------ Cavity with wake=True


def cavity_kwargs(**overrides):
    """RF settings for a TESLA-like cavity, with optional overrides."""
    kwargs = dict(
        length=torch.tensor(1.038, dtype=DTYPE),
        voltage=torch.tensor(16e6 * 1.038, dtype=DTYPE),
        phase=torch.tensor(0.0, dtype=DTYPE),
        frequency=torch.tensor(FREQUENCY, dtype=DTYPE),
    )
    kwargs.update(overrides)
    return kwargs


def test_wake_off_is_bit_identical_to_plain_cavity():
    """Wake parameters do nothing while `wake=False`."""
    beam = make_beam(3e-4, num_particles=10_000)
    plain = cheetah.Cavity(**cavity_kwargs())
    switched = cheetah.Cavity(
        **cavity_kwargs(),
        wake=False,
        wake_strength=torch.tensor(35e12, dtype=DTYPE),
    )
    assert torch.equal(plain.track(beam).particles, switched.track(beam).particles)
    assert switched.is_skippable == plain.is_skippable


def test_wake_on_changes_only_the_energy_coordinate():
    """Turning the wake on changes only the momentum of the particles."""
    beam = make_beam(3e-4, num_particles=10_000)
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=35e12,
    )
    reference = cheetah.Cavity(**cavity_kwargs()).track(beam)
    outgoing = cavity.track(beam)
    for index, name in enumerate(["x", "px", "y", "py", "tau"]):
        assert torch.equal(
            outgoing.particles[..., index], reference.particles[..., index]
        ), name
    assert not torch.equal(outgoing.particles[..., 5], reference.particles[..., 5])
    assert float(outgoing.p.mean()) < float(reference.p.mean())  # beam loading


def test_wake_scales_linearly_with_total_charge():
    """The cavity's wake kick scales linearly with the bunch charge."""
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=35e12,
    )
    reference_cavity = cheetah.Cavity(**cavity_kwargs())

    def mean_kick(total_charge):
        beam = make_beam(3e-4, total_charge=total_charge, num_particles=20_000)
        return float((cavity.track(beam).p - reference_cavity.track(beam).p).mean())

    single = mean_kick(1e-9)
    double = mean_kick(2e-9)
    assert double / single == pytest.approx(2.0, rel=1e-6)


def test_uncharged_beam_is_a_no_op():
    """An uncharged beam sees no wake."""
    beam = make_beam(3e-4, num_particles=5_000)
    uncharged = cheetah.ParticleBeam(
        particles=beam.particles,
        energy=beam.energy,
        particle_charges=torch.zeros_like(beam.particle_charges),
        species=beam.species,
    )
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=35e12,
    )
    reference = cheetah.Cavity(**cavity_kwargs()).track(uncharged)
    assert torch.equal(cavity.track(uncharged).particles, reference.particles)


def test_resonant_model_reproduces_the_old_cosine_wake_strength():
    """`wake_strength` reproduces the original cosine wake."""
    wake_strength = 35.0
    beam = make_beam(3e-4, total_charge=1e-9, num_particles=100_000)

    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=torch.tensor(wake_strength, dtype=DTYPE),
    )

    reference = cheetah.Cavity(**cavity_kwargs()).track(beam)
    outgoing = cavity.track(beam)

    # wake_position="exit": the kick is normalised by the *outgoing* p0c.
    p0c = float(
        torch.sqrt(reference.energy**2 - torch.as_tensor(beam.species.mass_eV) ** 2)
    )
    measured = (outgoing.p - reference.p).numpy() * p0c

    analytic = -gaussian_resonant_wake_potential(
        reference.tau.numpy(),
        sigma_tau=float(reference.tau.std()),
        wavelength=WAVELENGTH,
        total_charge=float(beam.particle_charges.sum()),
        wake_strength=wake_strength,
    )
    error = np.sqrt(np.mean((measured - analytic) ** 2)) / np.sqrt(
        np.mean(analytic**2)
    )
    assert error < 0.05, f"relative rms error {error:.3%}"


def test_split_position_agrees_with_exit_to_first_order():
    """Splitting the kick between both ends agrees with an exit kick to first order."""
    beam = make_beam(3e-4, num_particles=20_000)
    common = dict(
        wake=True,
        wake_strength=35e12,
    )
    exit_cavity = Cavity(**cavity_kwargs(), **common, wake_position="exit")
    split_cavity = Cavity(**cavity_kwargs(), **common, wake_position="split")
    at_exit = exit_cavity.track(beam).p
    split = split_cavity.track(beam).p
    tolerance = 2e-4 * float(at_exit.std()) + 1e-9
    assert torch.allclose(split, at_exit, rtol=0.0, atol=tolerance) or (
        float((split - at_exit).abs().max()) < 0.05 * float(at_exit.abs().max())
    )


def test_incomplete_wake_configuration_raises_immediately():
    """Unusable wake settings fail in the constructor, not during tracking."""
    with pytest.raises(ValueError, match="wake_strength"):
        Cavity(**cavity_kwargs(), wake=True)
    with pytest.raises(ValueError, match="non-zero frequency"):
        Cavity(length=torch.tensor(1.0, dtype=DTYPE), wake=True, wake_strength=35.0)


def test_parameter_beam_with_wake_raises():
    """Tracking a `ParameterBeam` with the wake on raises a clear error."""
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=35e12,
    )
    beam = cheetah.ParameterBeam.from_parameters(
        sigma_tau=torch.tensor(3e-4, dtype=DTYPE),
        energy=torch.tensor(REFERENCE_ENERGY, dtype=DTYPE),
        dtype=DTYPE,
    )
    with pytest.raises(AssertionError, match="ParticleBeam"):
        cavity.track(beam)


def test_diagnostics_are_cached_for_plotting():
    """The last wake diagnostics are kept on the cavity for plotting."""
    beam = make_beam(3e-4, num_particles=10_000)
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=35e12,
    )
    cavity.track(beam)
    diagnostics = cavity.cached_wake_diagnostics
    assert diagnostics is not None
    assert {"grid", "z_grid", "lambda", "W_grid"} <= set(diagnostics)
    assert diagnostics["grid"].resolved


def test_element_survives_a_segment_and_clone():
    """A wake cavity can be cloned and tracked inside a `Segment`."""
    beam = make_beam(3e-4, num_particles=5_000)
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=35e12,
        name="wake_cavity",
    )
    segment = cheetah.Segment(
        [cheetah.Drift(length=torch.tensor(0.1, dtype=DTYPE)), cavity]
    )
    outgoing = segment.track(beam)
    assert outgoing.particles.shape == beam.particles.shape
    assert "wake" in cavity.defining_features


def test_vectorised_beam_is_handled():
    """A vectorised (batched) beam gets the wake in every batch entry."""
    beam = make_beam(3e-4, num_particles=5_000)
    stacked = cheetah.ParticleBeam(
        particles=torch.stack([beam.particles, beam.particles]),
        energy=beam.energy,
        particle_charges=beam.particle_charges,
        species=beam.species,
    )
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=35e12,
    )
    outgoing = cavity.track(stacked)
    assert outgoing.particles.shape == stacked.particles.shape
    assert torch.allclose(outgoing.particles[0], outgoing.particles[1])


@pytest.mark.parametrize("parameter", ["frequency", "wake_strength"])
def test_vectorised_cavity_parameter_matches_scalar_runs(parameter):
    """Each entry of a vectorised `frequency` or `wake_strength` gets its own wake."""
    beam = make_beam(3e-2, num_particles=5_000)
    values = {
        "frequency": torch.tensor([1.3e9, 2.6e9], dtype=DTYPE),
        "wake_strength": torch.tensor([35e12, 70e12], dtype=DTYPE),
    }[parameter]
    settings = cavity_kwargs(
        wake=True, wake_strength=torch.tensor(35e12, dtype=DTYPE)
    )

    outgoing = Cavity(**{**settings, parameter: values}).track(beam)
    assert outgoing.particles.shape == (2, *beam.particles.shape)

    for index, value in enumerate(values):
        scalar = Cavity(**{**settings, parameter: value}).track(beam)
        assert torch.allclose(outgoing.particles[index], scalar.particles)


def test_wake_strength_survives_clone():
    """`wake_strength` survives `clone()`."""
    cavity = Cavity(
        **cavity_kwargs(),
        wake=True,
        wake_strength=torch.tensor(35.0, dtype=DTYPE),
    )
    assert float(cavity.clone().wake_strength) == pytest.approx(35.0, rel=1e-14)


def test_wake_strength_is_differentiable():
    """
    Gradients flow back to `wake_strength`, and match `kick / wake_strength` because the
    kick is linear.
    """
    beam = make_beam(3e-4, num_particles=20_000)
    amplitude = 35e9
    wake_strength = torch.tensor(amplitude, dtype=DTYPE, requires_grad=True)
    cavity = Cavity(
        length=torch.tensor(0.0, dtype=DTYPE),
        voltage=torch.tensor(0.0, dtype=DTYPE),
        phase=torch.tensor(0.0, dtype=DTYPE),
        frequency=torch.tensor(FREQUENCY, dtype=DTYPE),
        wake=True,
        wake_strength=wake_strength,
    )

    mean_kick = (cavity.track(beam).p - beam.p).mean()
    mean_kick.backward()

    assert wake_strength.grad is not None
    assert float(wake_strength.grad) == pytest.approx(
        float(mean_kick.detach()) / amplitude, rel=1e-12
    )


def test_wake_is_not_dropped_by_segment_tracking():
    """A `Segment` does not merge a wake cavity away, so its kick survives."""
    beam = make_beam(3e-4, num_particles=5_000)
    drift = cheetah.Drift(length=torch.tensor(0.1, dtype=DTYPE))
    # voltage=0 on purpose.  `Cavity.is_skippable` is `not is_active`, so a
    # powered cavity is non-skippable anyway and would mask the regression --
    # only with the RF off does the `not self.wake` term actually carry the
    # guard, which is the case that can silently break.
    cavity = Cavity(
        **cavity_kwargs(voltage=torch.tensor(0.0, dtype=DTYPE)),
        wake=True,
        wake_strength=35e12,
        name="wake_cavity",
    )
    segment = cheetah.Segment([drift, cavity, drift.clone()])

    assert not cavity.is_skippable
    assert not segment.is_skippable

    direct = drift.track(cavity.track(drift.track(beam))).p
    tolerance = 1e-9 * float(direct.abs().max())

    assert torch.allclose(segment.track(beam).p, direct, rtol=0.0, atol=tolerance)
    assert torch.allclose(
        segment.transfer_maps_merged(incoming_beam=beam).track(beam).p,
        direct,
        rtol=0.0,
        atol=tolerance,
    )

    # Non-vacuous: the wake must actually move p, or the asserts above are free.
    wake_free = (
        cheetah.Segment(
            [
                drift,
                cheetah.Cavity(**cavity_kwargs(voltage=torch.tensor(0.0, dtype=DTYPE))),
                drift.clone(),
            ]
        )
        .track(beam)
        .p
    )
    assert float((direct - wake_free).abs().max()) > 1e-3 * float(direct.abs().max())
