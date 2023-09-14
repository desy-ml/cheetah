import numpy as np
import pytest

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


def test_solenoid_end():
    """
    Test that at the end of a split solenoid the result is the same as at the end of
    the original solenoid.
    """
    original_solenoid = cheetah.Solenoid(length=0.2, k=4.2)
    split_solenoid = cheetah.Segment(original_solenoid.split(0.01))

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_solenoid.track(incoming_beam)
    outgoing_beam_split = split_solenoid.track(incoming_beam)

    assert np.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


def test_dipole_end():
    """
    Test that at the end of a split dipole the result is the same as at the end of
    the original dipole.
    """
    original_dipole = cheetah.Dipole(length=0.2, angle=4.2)
    split_dipole = cheetah.Segment(original_dipole.split(0.01))

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_dipole.track(incoming_beam)
    outgoing_beam_split = split_dipole.track(incoming_beam)

    assert np.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


def test_undulator_end():
    """
    Test that at the end of a split undulator the result is the same as at the end of
    the original undulator.
    """
    original_undulator = cheetah.Undulator(length=3.142)
    split_undulator = cheetah.Segment(original_undulator.split(0.1))

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_undulator.track(incoming_beam)
    outgoing_beam_split = split_undulator.track(incoming_beam)

    assert np.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


@pytest.mark.xfail  # TODO: Fix this
def test_horizontal_corrector_end():
    """
    Test that at the end of a split horizontal corrector the result is the same as at
    the end of the original horizontal corrector.
    """
    original_horizontal_corrector = cheetah.HorizontalCorrector(length=0.2, angle=4.2)
    split_horizontal_corrector = cheetah.Segment(
        original_horizontal_corrector.split(0.01)
    )

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_horizontal_corrector.track(incoming_beam)
    outgoing_beam_split = split_horizontal_corrector.track(incoming_beam)

    assert np.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


@pytest.mark.xfail  # TODO: Fix this
def test_vertical_corrector_end():
    """
    Test that at the end of a split vertical corrector the result is the same as at
    the end of the original vertical corrector.
    """
    original_vertical_corrector = cheetah.VerticalCorrector(length=0.2, angle=4.2)
    split_vertical_corrector = cheetah.Segment(original_vertical_corrector.split(0.01))

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_vertical_corrector.track(incoming_beam)
    outgoing_beam_split = split_vertical_corrector.track(incoming_beam)

    assert np.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)
