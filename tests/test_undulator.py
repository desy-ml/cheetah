import math

import torch

import cheetah


def test_undulator_off():
    """
    Test that an undulator with kx=ky=0 behaves like a drift and that an undulator with
    either kx or ky non-zero behaves differently from a drift.
    """
    undulator = cheetah.Undulator(length=torch.tensor(1.0), period=torch.tensor(0.1))
    drift = cheetah.Drift(length=torch.tensor(1.0))
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_px=torch.tensor(2e-7), sigma_py=torch.tensor(2e-7)
    )

    outgoing_beam_drift = drift.track(incoming_beam)
    outgoing_beam_undulator_off = undulator.track(incoming_beam)

    undulator.kx = torch.tensor(1.0)
    undulator.ky = torch.tensor(0.0)
    outgoing_beam_undulator_on_y = undulator.track(incoming_beam)

    undulator.kx = torch.tensor(0.0)
    undulator.ky = torch.tensor(1.0)
    outgoing_beam_undulator_on_x = undulator.track(incoming_beam)

    assert torch.allclose(
        outgoing_beam_undulator_off.sigma_x, outgoing_beam_drift.sigma_x
    )
    assert torch.allclose(
        outgoing_beam_undulator_off.sigma_y, outgoing_beam_drift.sigma_y
    )
    assert not torch.allclose(
        outgoing_beam_undulator_on_y.sigma_y, outgoing_beam_drift.sigma_y
    )
    assert not torch.allclose(
        outgoing_beam_undulator_on_x.sigma_x, outgoing_beam_drift.sigma_x
    )


def test_undulator_fel_tracking():
    """Test 3D FEL simulation tracking mode in `Undulator`."""
    undulator = cheetah.Undulator(
        length=torch.tensor(5.0),
        period=torch.tensor(0.04),
        kx=torch.tensor(0.0),
        ky=torch.tensor(3.36),
        tracking_method="fel",
        seed_power=torch.tensor(1.090578e5),
        step_size=torch.tensor(0.24),
        num_macroparticles=512,
    )

    energy = torch.tensor(16.3e9)
    gamma0 = energy / cheetah.Species("electron").mass_eV
    emit_factor = math.sqrt(gamma0.item() ** 2 - 1.0)

    incoming_beam = cheetah.ParameterBeam.from_twiss(
        alpha_x=torch.tensor(-1.599),
        alpha_y=torch.tensor(0.2545),
        beta_x=torch.tensor(0.8756),
        beta_y=torch.tensor(3.1995),
        emittance_x=torch.tensor(0.4e-6 / emit_factor),
        emittance_y=torch.tensor(0.4e-6 / emit_factor),
        energy=energy,
    )

    outgoing_beam = undulator.track(incoming_beam)

    assert hasattr(outgoing_beam, "radiation")
    assert isinstance(outgoing_beam.radiation, cheetah.RadiationField)
    assert undulator.radiation is not None
    assert outgoing_beam.radiation.power > 0.0
    assert torch.allclose(outgoing_beam.radiation.power, undulator.radiation.power)


def test_undulator_fel_lattice_and_autograd():
    """
    Test FEL propagation across a multi-element lattice with quads and drifts,
    and verify autograd differentiability.
    """
    ky_param = torch.tensor(3.36, requires_grad=True)
    und1 = cheetah.Undulator(
        length=torch.tensor(5.0),
        period=torch.tensor(0.04),
        kx=torch.tensor(0.0),
        ky=ky_param,
        tracking_method="fel",
        seed_power=torch.tensor(1.090578e5),
        step_size=torch.tensor(0.24),
        num_macroparticles=512,
    )
    drift = cheetah.Drift(length=torch.tensor(0.5))
    quad = cheetah.Quadrupole(length=torch.tensor(0.1), k1=torch.tensor(0.5))
    und2 = cheetah.Undulator(
        length=torch.tensor(5.0),
        period=torch.tensor(0.04),
        kx=torch.tensor(0.0),
        ky=ky_param,
        tracking_method="fel",
        step_size=torch.tensor(0.24),
        num_macroparticles=512,
    )

    lattice = cheetah.Segment(elements=[und1, drift, quad, und2])

    energy = torch.tensor(16.3e9)
    gamma0 = energy / cheetah.Species("electron").mass_eV
    emit_factor = math.sqrt(gamma0.item() ** 2 - 1.0)

    incoming_beam = cheetah.ParameterBeam.from_twiss(
        alpha_x=torch.tensor(-1.599),
        alpha_y=torch.tensor(0.2545),
        beta_x=torch.tensor(0.8756),
        beta_y=torch.tensor(3.1995),
        emittance_x=torch.tensor(0.4e-6 / emit_factor),
        emittance_y=torch.tensor(0.4e-6 / emit_factor),
        energy=energy,
    )

    outgoing_beam = lattice.track(incoming_beam)
    final_power = outgoing_beam.radiation.power

    assert final_power > 0.0
    final_power.backward()
    assert ky_param.grad is not None
    assert ky_param.grad.abs() > 0.0


def test_undulator_fel_particle_beam():
    """Test FEL tracking with a `ParticleBeam` input."""
    undulator = cheetah.Undulator(
        length=torch.tensor(1.0),
        period=torch.tensor(0.04),
        kx=torch.tensor(0.0),
        ky=torch.tensor(2.0),
        tracking_method="fel",
        step_size=torch.tensor(0.2),
        num_macroparticles=256,
    )

    energy = torch.tensor(1e9)
    gamma0 = energy / cheetah.Species("electron").mass_eV
    emit_factor = math.sqrt(gamma0.item() ** 2 - 1.0)

    incoming_beam = cheetah.ParticleBeam.from_twiss(
        num_particles=256,
        alpha_x=torch.tensor(0.0),
        alpha_y=torch.tensor(0.0),
        beta_x=torch.tensor(10.0),
        beta_y=torch.tensor(10.0),
        emittance_x=torch.tensor(1e-6 / emit_factor),
        emittance_y=torch.tensor(1e-6 / emit_factor),
        energy=energy,
    )

    outgoing_beam = undulator.track(incoming_beam)
    assert isinstance(outgoing_beam, cheetah.ParticleBeam)
    assert hasattr(outgoing_beam, "radiation")
    assert outgoing_beam.radiation.power > 0.0


def test_undulator_fel_energy_conservation():
    """Test energy conservation between electron beam and radiation field."""
    from scipy.constants import electron_mass, elementary_charge, speed_of_light

    seed_power = torch.tensor(1e6, dtype=torch.float64)
    current = torch.tensor(3000.0, dtype=torch.float64)
    undulator = cheetah.Undulator(
        length=torch.tensor(0.5, dtype=torch.float64),
        period=torch.tensor(0.04, dtype=torch.float64),
        kx=torch.tensor(0.0, dtype=torch.float64),
        ky=torch.tensor(2.0, dtype=torch.float64),
        tracking_method="fel",
        seed_power=seed_power,
        bunching=torch.tensor(0.1, dtype=torch.float64),
        current=current,
        step_size=torch.tensor(0.01, dtype=torch.float64),
        grid_extent=torch.tensor(5e-4, dtype=torch.float64),
        grid_shape=(65, 65),
        num_macroparticles=1024,
    )

    energy = torch.tensor(1e9, dtype=torch.float64)
    incoming_beam = cheetah.ParameterBeam.from_twiss(
        beta_x=torch.tensor(5.0, dtype=torch.float64),
        beta_y=torch.tensor(5.0, dtype=torch.float64),
        emittance_x=torch.tensor(1e-10, dtype=torch.float64),
        emittance_y=torch.tensor(1e-10, dtype=torch.float64),
        energy=energy,
        dtype=torch.float64,
    )

    outgoing_beam = undulator.track(incoming_beam)
    delta_p_rad = outgoing_beam.radiation.power - seed_power
    mean_delta_p = outgoing_beam.mu[..., 5]
    p_beam_conversion = current * (
        electron_mass * speed_of_light**2 / elementary_charge
    )
    gamma0 = energy / cheetah.Species("electron", dtype=torch.float64).mass_eV
    delta_p_beam = p_beam_conversion * gamma0 * mean_delta_p

    assert delta_p_rad > 0.0
    assert torch.isclose(delta_p_rad, -delta_p_beam, rtol=0.05)


def test_undulator_fel_focusing_axes_and_zero_k():
    """
    Test that FEL natural focusing matches `first_order_transfer_map` axes (`ky` focuses
    `x`, `kx` focuses `y`) and that `kx=ky=0` produces finite results.
    """
    energy = torch.tensor(1e8)
    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=64,
        sigma_x=torch.tensor(1e-4),
        sigma_y=torch.tensor(1e-4),
        energy=energy,
    )

    und_off = cheetah.Undulator(
        length=torch.tensor(1.0),
        period=torch.tensor(0.04),
        kx=torch.tensor(0.0),
        ky=torch.tensor(0.0),
        tracking_method="fel",
        wavelength=torch.tensor(1e-6),
        step_size=torch.tensor(0.5),
    )
    out_off = und_off.track(beam)
    assert torch.isfinite(out_off.particles).all()
    assert torch.isfinite(out_off.radiation.power)

    und_ky = cheetah.Undulator(
        length=torch.tensor(2.0),
        period=torch.tensor(0.04),
        kx=torch.tensor(0.0),
        ky=torch.tensor(3.0),
        tracking_method="fel",
        step_size=torch.tensor(0.5),
    )
    out_ky_fel = und_ky.track(beam)
    und_ky.tracking_method = "linear"
    out_ky_lin = und_ky.track(beam)

    assert torch.allclose(out_ky_fel.sigma_x, out_ky_lin.sigma_x, rtol=1e-4)
    assert torch.allclose(out_ky_fel.sigma_y, out_ky_lin.sigma_y, rtol=1e-4)


def test_undulator_fel_upstream_autograd_and_moments():
    """
    Test that gradients flow from FEL power back through `ParameterBeam` covariance
    to an upstream `Quadrupole`, and that exit transverse moments are updated.
    """
    k1 = torch.tensor(1.5, requires_grad=True)
    quad = cheetah.Quadrupole(length=torch.tensor(0.2), k1=k1)
    drift = cheetah.Drift(length=torch.tensor(0.5))
    und = cheetah.Undulator(
        length=torch.tensor(2.0),
        period=torch.tensor(0.04),
        kx=torch.tensor(0.0),
        ky=torch.tensor(3.36),
        tracking_method="fel",
        step_size=torch.tensor(0.25),
        num_macroparticles=256,
    )
    segment = cheetah.Segment(elements=[quad, drift, und])

    energy = torch.tensor(1e9)
    incoming = cheetah.ParameterBeam.from_twiss(
        beta_x=torch.tensor(5.0),
        beta_y=torch.tensor(5.0),
        emittance_x=torch.tensor(1e-9),
        emittance_y=torch.tensor(1e-9),
        energy=energy,
    )

    outgoing = segment.track(incoming)
    outgoing.radiation.power.backward()

    assert k1.grad is not None
    assert k1.grad.abs() > 0.0
    assert not torch.allclose(outgoing.sigma_x, incoming.sigma_x)


def test_undulator_fel_nonsquare_grid_and_boundary_absorption():
    """
    Test non-square `grid_shape`, preservation of `incoming.radiation.extent`,
    and boundary absorption in `RadiationField.diffract`.
    """
    und1 = cheetah.Undulator(
        length=torch.tensor(0.5),
        period=torch.tensor(0.04),
        ky=torch.tensor(2.0),
        tracking_method="fel",
        grid_shape=(33, 65),
        grid_extent=torch.tensor(4e-4),
        step_size=torch.tensor(0.25),
        num_macroparticles=128,
    )
    und2 = cheetah.Undulator(
        length=torch.tensor(0.5),
        period=torch.tensor(0.04),
        ky=torch.tensor(2.0),
        tracking_method="fel",
        grid_shape=(17, 17),
        grid_extent=torch.tensor(1e-4),
        step_size=torch.tensor(0.25),
        num_macroparticles=128,
    )
    beam = cheetah.ParameterBeam.from_twiss(
        beta_x=torch.tensor(5.0),
        beta_y=torch.tensor(5.0),
        emittance_x=torch.tensor(1e-9),
        emittance_y=torch.tensor(1e-9),
        energy=torch.tensor(1e9),
    )

    mid = und1.track(beam)
    assert mid.radiation.field.shape[-2:] == (33, 65)
    out = und2.track(mid)
    assert out.radiation.field.shape[-2:] == (33, 65)
    assert torch.isclose(out.radiation.extent, torch.tensor(4e-4))

    # Diffracting far enough to hit the boundary should attenuate power, not wrap around
    far = mid.radiation.diffract(torch.tensor(10.0), absorbing_boundary=True)
    assert far.power < mid.radiation.power
