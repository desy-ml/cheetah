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

    torch.testing.assert_close(
        outgoing_beam_undulator_off.sigma_x, outgoing_beam_drift.sigma_x
    )
    torch.testing.assert_close(
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
