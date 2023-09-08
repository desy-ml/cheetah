"""
Tests that ensure that both beam types produce (roughly) the same results.
"""

import numpy as np

import cheetah


def test_cavity():
    """Test that the cavity output for both beam types is roughly the same."""

    # Set up lattice
    cheetah_cavity = cheetah.Cavity(
        length=1.0377, voltage=0.01815975e9, frequency=1.3e9, phase=0.0
    )

    # Parameter beam
    incoming_parameter_beam = cheetah.ParameterBeam.from_twiss(
        beta_x=5.91253677,
        alpha_x=3.55631308,
        beta_y=5.91253677,
        alpha_y=3.55631308,
        emittance_x=3.494768647122823e-09,
        emittance_y=3.497810737006068e-09,
        energy=6e6,
    )
    outgoing_parameter_beam = cheetah_cavity.track(incoming_parameter_beam)

    # Particle beam
    incoming_particle_beam = cheetah.ParticleBeam.from_twiss(
        beta_x=5.91253677,
        alpha_x=3.55631308,
        beta_y=5.91253677,
        alpha_y=3.55631308,
        emittance_x=3.494768647122823e-09,
        emittance_y=3.497810737006068e-09,
        energy=6e6,
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
