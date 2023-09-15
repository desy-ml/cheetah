"""
Tests that ensure that both beam types produce (roughly) the same results.
"""
import numpy as np
import torch

import cheetah


def test_from_twiss():
    """
    Test that a beams created from Twiss parameters have the same properties.
    """
    parameter_beam = cheetah.ParameterBeam.from_twiss(
        beta_x=torch.tensor(5.91253676811640894),
        alpha_x=torch.tensor(3.55631307633660354),
        emittance_x=torch.tensor(3.494768647122823e-09),
        beta_y=torch.tensor(5.91253676811640982),
        alpha_y=torch.tensor(2e-7),
        emittance_y=torch.tensor(3.497810737006068e-09),
        energy=torch.tensor(6e6),
    )
    particle_beam = cheetah.ParticleBeam.from_twiss(
        num_particles=torch.tensor(
            10_000_000
        ),  # Large number of particles reduces noise
        beta_x=torch.tensor(5.91253676811640894),
        alpha_x=torch.tensor(3.55631307633660354),
        emittance_x=torch.tensor(3.494768647122823e-09),
        beta_y=torch.tensor(5.91253676811640982),
        alpha_y=torch.tensor(2e-7),
        emittance_y=torch.tensor(3.497810737006068e-09),
        energy=torch.tensor(6e6),
    )

    assert np.isclose(parameter_beam.mu_x, particle_beam.mu_x, atol=1e-6)
    assert np.isclose(parameter_beam.mu_y, particle_beam.mu_y, atol=1e-6)
    assert np.isclose(parameter_beam.sigma_x, particle_beam.sigma_x, rtol=1e-3)
    assert np.isclose(parameter_beam.sigma_y, particle_beam.sigma_y, rtol=1e-3)
    assert np.isclose(parameter_beam.mu_xp, particle_beam.mu_xp, atol=1e-6)
    assert np.isclose(parameter_beam.mu_yp, particle_beam.mu_yp, atol=1e-6)
    assert np.isclose(parameter_beam.sigma_xp, particle_beam.sigma_xp, rtol=1e-3)
    assert np.isclose(parameter_beam.sigma_yp, particle_beam.sigma_yp, rtol=1e-3)
    assert np.isclose(parameter_beam.mu_s, particle_beam.mu_s)
    assert np.isclose(parameter_beam.sigma_s, particle_beam.sigma_s)
    assert np.isclose(parameter_beam.mu_p, particle_beam.mu_p)
    assert np.isclose(parameter_beam.sigma_p, particle_beam.sigma_p)


def test_drift():
    """Test that the drift output for both beam types is roughly the same."""

    # Set up lattice
    cheetah_drift = cheetah.Drift(length=torch.tensor(1.0))

    # Parameter beam
    incoming_parameter_beam = cheetah.ParameterBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )
    outgoing_parameter_beam = cheetah_drift.track(incoming_parameter_beam)

    # Particle beam
    incoming_particle_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )
    outgoing_particle_beam = cheetah_drift.track(incoming_particle_beam)

    # Compare
    assert np.isclose(outgoing_parameter_beam.energy, outgoing_particle_beam.energy)
    assert np.isclose(
        outgoing_parameter_beam.mu_x, outgoing_particle_beam.mu_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_y, outgoing_particle_beam.mu_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_x, outgoing_particle_beam.sigma_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_y, outgoing_particle_beam.sigma_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_xp, outgoing_particle_beam.mu_xp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_yp, outgoing_particle_beam.mu_yp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_xp, outgoing_particle_beam.sigma_xp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_yp, outgoing_particle_beam.sigma_yp, rtol=1e-2
    )


def test_quadrupole():
    """Test that the quadrupole output for both beam types is roughly the same."""

    # Set up lattice
    cheetah_quadrupole = cheetah.Quadrupole(
        length=torch.tensor(0.15), k1=torch.tensor(4.2)
    )

    # Parameter beam
    incoming_parameter_beam = cheetah.ParameterBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )
    outgoing_parameter_beam = cheetah_quadrupole.track(incoming_parameter_beam)

    # Particle beam
    incoming_particle_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )
    outgoing_particle_beam = cheetah_quadrupole.track(incoming_particle_beam)

    # Compare
    assert np.isclose(outgoing_parameter_beam.energy, outgoing_particle_beam.energy)
    assert np.isclose(
        outgoing_parameter_beam.mu_x, outgoing_particle_beam.mu_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_y, outgoing_particle_beam.mu_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_x, outgoing_particle_beam.sigma_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_y, outgoing_particle_beam.sigma_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_xp, outgoing_particle_beam.mu_xp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_yp, outgoing_particle_beam.mu_yp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_xp, outgoing_particle_beam.sigma_xp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_yp, outgoing_particle_beam.sigma_yp, rtol=1e-2
    )


def test_cavity_from_astra():
    """
    Test that the cavity output for both beam types is roughly the same. This test uses
    a beam converted from an ASTRA beam file.
    """

    # Set up lattice
    cheetah_cavity = cheetah.Cavity(
        length=torch.tensor(1.0377),
        voltage=torch.tensor(0.01815975e9),
        frequency=torch.tensor(1.3e9),
        phase=torch.tensor(0.0),
    )

    # Parameter beam
    incoming_parameter_beam = cheetah.ParameterBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )
    outgoing_parameter_beam = cheetah_cavity.track(incoming_parameter_beam)

    # Particle beam
    incoming_particle_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )
    outgoing_particle_beam = cheetah_cavity.track(incoming_particle_beam)

    # Compare
    assert np.isclose(
        outgoing_parameter_beam.beta_x, outgoing_particle_beam.beta_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.alpha_x, outgoing_particle_beam.alpha_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.beta_y, outgoing_particle_beam.beta_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.alpha_y, outgoing_particle_beam.alpha_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.emittance_x, outgoing_particle_beam.emittance_x
    )
    assert np.isclose(
        outgoing_parameter_beam.emittance_y, outgoing_particle_beam.emittance_y
    )
    assert np.isclose(outgoing_parameter_beam.energy, outgoing_particle_beam.energy)
    assert np.isclose(
        outgoing_parameter_beam.mu_x, outgoing_particle_beam.mu_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_y, outgoing_particle_beam.mu_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_x, outgoing_particle_beam.sigma_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_y, outgoing_particle_beam.sigma_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_xp, outgoing_particle_beam.mu_xp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_yp, outgoing_particle_beam.mu_yp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_xp, outgoing_particle_beam.sigma_xp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_yp, outgoing_particle_beam.sigma_yp, rtol=1e-2
    )


def test_cavity_from_twiss():
    """
    Test that the cavity output for both beam types is roughly the same. This test uses
    a beam generated from Twiss parameters.
    """

    # Set up lattice
    cheetah_cavity = cheetah.Cavity(
        length=torch.tensor(1.0377),
        voltage=torch.tensor(0.01815975e9),
        frequency=torch.tensor(1.3e9),
        phase=torch.tensor(0.0),
    )

    # Parameter beam
    incoming_parameter_beam = cheetah.ParameterBeam.from_twiss(
        beta_x=torch.tensor(5.91253677),
        alpha_x=torch.tensor(3.55631308),
        beta_y=torch.tensor(5.91253677),
        alpha_y=torch.tensor(3.55631308),
        emittance_x=torch.tensor(3.494768647122823e-09),
        emittance_y=torch.tensor(3.497810737006068e-09),
        energy=torch.tensor(6e6),
    )
    outgoing_parameter_beam = cheetah_cavity.track(incoming_parameter_beam)

    # Particle beam
    incoming_particle_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=torch.tensor(5.91253677),
        alpha_x=torch.tensor(3.55631308),
        beta_y=torch.tensor(5.91253677),
        alpha_y=torch.tensor(3.55631308),
        emittance_x=torch.tensor(3.494768647122823e-09),
        emittance_y=torch.tensor(3.497810737006068e-09),
        energy=torch.tensor(6e6),
    )
    outgoing_particle_beam = cheetah_cavity.track(incoming_particle_beam)

    # Compare
    assert np.isclose(
        outgoing_parameter_beam.beta_x, outgoing_particle_beam.beta_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.alpha_x, outgoing_particle_beam.alpha_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.beta_y, outgoing_particle_beam.beta_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.alpha_y, outgoing_particle_beam.alpha_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.emittance_x, outgoing_particle_beam.emittance_x
    )
    assert np.isclose(
        outgoing_parameter_beam.emittance_y, outgoing_particle_beam.emittance_y
    )
    assert np.isclose(outgoing_parameter_beam.energy, outgoing_particle_beam.energy)
    assert np.isclose(
        outgoing_parameter_beam.mu_x, outgoing_particle_beam.mu_x, atol=1e-6
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_y, outgoing_particle_beam.mu_y, atol=1e-6
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_x, outgoing_particle_beam.sigma_x, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_y, outgoing_particle_beam.sigma_y, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_xp, outgoing_particle_beam.mu_xp, atol=1e-6
    )
    assert np.isclose(
        outgoing_parameter_beam.mu_yp, outgoing_particle_beam.mu_yp, atol=1e-6
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_xp, outgoing_particle_beam.sigma_xp, rtol=1e-2
    )
    assert np.isclose(
        outgoing_parameter_beam.sigma_yp, outgoing_particle_beam.sigma_yp, rtol=1e-2
    )
