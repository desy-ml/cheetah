import numpy as np

import cheetah


def test_drift_end():
    """
    Test that at the end of a split drift the result is the same as at the end of the
    original drift.
    """
    original_drift = cheetah.Drift(length=2.0)
    split_drift = cheetah.Segment(original_drift.split(0.1))

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_drift.track(incoming_beam)
    outgoing_beam_split = split_drift.track(incoming_beam)

    assert np.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


def test_quadrupole_end():
    """
    Test that at the end of a split quadrupole the result is the same as at the end of
    the original quadrupole.
    """
    original_quadrupole = cheetah.Quadrupole(length=0.2, k1=4.2)
    split_quadrupole = cheetah.Segment(original_quadrupole.split(0.01))

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_quadrupole.track(incoming_beam)
    outgoing_beam_split = split_quadrupole.track(incoming_beam)

    assert np.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


def test_cavity_end():
    """
    Test that at the end of a split cavity the result is the same as at the end of
    the original cavity.
    """
    original_cavity = cheetah.Cavity(
        length=1.0377, voltage=0.01815975e9, frequency=1.3e9, phase=0.0
    )
    split_cavity = cheetah.Segment(original_cavity.split(0.1))

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_cavity.track(incoming_beam)
    outgoing_beam_split = split_cavity.track(incoming_beam)

    assert np.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)
