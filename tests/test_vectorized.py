import pytest
import torch

import cheetah

from .resources import ARESlatticeStage3v1_9 as ares


def test_segment_length_shape():
    """Test that the shape of a segment's length matches the input."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([0.6, 0.5])),
            cheetah.Quadrupole(
                length=torch.tensor([0.2, 0.25]), k1=torch.tensor([4.2, 4.2])
            ),
            cheetah.Drift(length=torch.tensor([0.4, 0.3])),
        ]
    )

    assert segment.length.shape == (2,)


def test_segment_length_shape_2d():
    """
    Test that the shape of a segment's length matches the input for a batch with
    multiple dimensions.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([[0.6, 0.5], [0.4, 0.3], [0.4, 0.3]])),
            cheetah.Quadrupole(
                length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.3, 0.35]]),
                k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.3, 4.3]]),
            ),
            cheetah.Drift(length=torch.tensor([[0.4, 0.3], [0.2, 0.1], [0.2, 0.1]])),
        ]
    )

    assert segment.length.shape == (3, 2)


@pytest.mark.parametrize("BeamClass", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_track_quadrupole_shape(BeamClass):
    """
    Test that the shape of a beam tracked through a single quadrupole element matches
    the input.
    """
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor([0.2, 0.25]), k1=torch.tensor([4.2, 4.2])
    )
    incoming = BeamClass.from_parameters(sigma_x=torch.tensor([1e-5, 2e-5]))

    outgoing = quadrupole.track(incoming)

    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particles.shape == (2, 100_000, 7)
    assert outgoing.mu_x.shape == (2,)
    assert outgoing.mu_px.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_py.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_px.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_py.shape == (2,)
    assert outgoing.sigma_tau.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (2, 100_000)


@pytest.mark.parametrize("BeamClass", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_track_quadrupole_shape_2d(BeamClass):
    """
    Test that the shape of a beam tracked through a single quadrupole element matches
    the input for an n-dimensional batch.
    """
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.4, 0.45]]),
        k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.4, 4.4]]),
    )
    incoming = BeamClass.from_parameters(
        sigma_x=torch.tensor([[1e-5, 2e-5], [2e-5, 3e-5], [3e-5, 4e-5]])
    )

    outgoing = quadrupole.track(incoming)

    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particles.shape == (3, 2, 100_000, 7)
    assert outgoing.mu_x.shape == (3, 2)
    assert outgoing.mu_px.shape == (3, 2)
    assert outgoing.mu_y.shape == (3, 2)
    assert outgoing.mu_py.shape == (3, 2)
    assert outgoing.sigma_x.shape == (3, 2)
    assert outgoing.sigma_px.shape == (3, 2)
    assert outgoing.sigma_y.shape == (3, 2)
    assert outgoing.sigma_py.shape == (3, 2)
    assert outgoing.sigma_tau.shape == (3, 2)
    assert outgoing.sigma_p.shape == (3, 2)
    assert outgoing.energy.shape == (3, 2)
    assert outgoing.total_charge.shape == (3, 2)
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (3, 2, 100_000)


@pytest.mark.parametrize("BeamClass", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_track_segment_shape(BeamClass):
    """
    Test that the shape of a beam tracked through a segment matches the input.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([0.6, 0.5])),
            cheetah.Quadrupole(
                length=torch.tensor([0.2, 0.25]), k1=torch.tensor([4.2, 4.2])
            ),
            cheetah.Drift(length=torch.tensor([0.4, 0.3])),
        ]
    )
    incoming = BeamClass.from_parameters(sigma_x=torch.tensor([1e-5, 2e-5]))

    outgoing = segment.track(incoming)

    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particles.shape == (2, 100_000, 7)
    assert outgoing.mu_x.shape == (2,)
    assert outgoing.mu_px.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_py.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_px.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_py.shape == (2,)
    assert outgoing.sigma_tau.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (2, 100_000)


@pytest.mark.parametrize("BeamClass", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_track_particle_segment_shape_2d(BeamClass):
    """
    Test that the shape of a particle beam tracked through a segment matches the input
    for the case of a multi-dimensional batch.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]])),
            cheetah.Quadrupole(
                length=torch.tensor([[0.2, 0.25], [0.3, 0.35], [0.4, 0.45]]),
                k1=torch.tensor([[4.2, 4.2], [4.3, 4.3], [4.4, 4.4]]),
            ),
            cheetah.Drift(length=torch.tensor([[0.4, 0.3], [0.6, 0.5], [0.8, 0.7]])),
        ]
    )
    incoming = BeamClass.from_parameters(
        sigma_x=torch.tensor([[1e-5, 2e-5], [2e-5, 3e-5], [3e-5, 4e-5]])
    )

    outgoing = segment.track(incoming)

    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particles.shape == (3, 2, 100_000, 7)
    assert outgoing.mu_x.shape == (3, 2)
    assert outgoing.mu_px.shape == (3, 2)
    assert outgoing.mu_y.shape == (3, 2)
    assert outgoing.mu_py.shape == (3, 2)
    assert outgoing.sigma_x.shape == (3, 2)
    assert outgoing.sigma_px.shape == (3, 2)
    assert outgoing.sigma_y.shape == (3, 2)
    assert outgoing.sigma_py.shape == (3, 2)
    assert outgoing.sigma_tau.shape == (3, 2)
    assert outgoing.sigma_p.shape == (3, 2)
    assert outgoing.energy.shape == (3, 2)
    assert outgoing.total_charge.shape == (3, 2)
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (3, 2, 100_000)


def test_enormous_through_ares():
    """
    Test ARES EA with a huge number of settings. This is a stress test and only run
    for `ParameterBeam` because `ParticleBeam` would require a lot of memory.
    """
    segment = cheetah.Segment.from_ocelot(ares.cell).subcell("AREASOLA1", "AREABSCR1")
    incoming = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    segment.AREAMQZM1.k1 = torch.linspace(-30.0, 30.0, 200_000).repeat(3, 1)

    outgoing = segment.track(incoming)

    assert outgoing.mu_x.shape == (3, 200_000)
    assert outgoing.mu_px.shape == (3, 200_000)
    assert outgoing.mu_y.shape == (3, 200_000)
    assert outgoing.mu_py.shape == (3, 200_000)
    assert outgoing.sigma_x.shape == (3, 200_000)
    assert outgoing.sigma_px.shape == (3, 200_000)
    assert outgoing.sigma_y.shape == (3, 200_000)
    assert outgoing.sigma_py.shape == (3, 200_000)
    assert outgoing.sigma_tau.shape == (3, 200_000)
    assert outgoing.sigma_p.shape == (3, 200_000)
    assert outgoing.energy.shape == (3, 200_000)
    assert outgoing.total_charge.shape == (3, 200_000)


@pytest.mark.parametrize("BeamClass", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_cavity_with_zero_and_non_zero_voltage(BeamClass):
    """
    Tests that if zero and non-zero voltages are passed to a cavity in a single batch,
    there are no errors. This test does NOT check physical correctness.
    """
    cavity = cheetah.Cavity(
        length=torch.tensor(3.0441),
        voltage=torch.tensor([0.0, 48198468.0, 0.0]),
        phase=torch.tensor(48198468.0),
        frequency=torch.tensor(2.8560e09),
        name="my_test_cavity",
    )
    incoming = BeamClass.from_parameters(sigma_x=torch.tensor(1e-5))

    outgoing = cavity.track(incoming)

    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particles.shape == (3, 100_000, 7)
    assert outgoing.mu_x.shape == (3,)
    assert outgoing.mu_px.shape == (3,)
    assert outgoing.mu_y.shape == (3,)
    assert outgoing.mu_py.shape == (3,)
    assert outgoing.sigma_x.shape == (3,)
    assert outgoing.sigma_px.shape == (3,)
    assert outgoing.sigma_y.shape == (3,)
    assert outgoing.sigma_py.shape == (3,)
    assert outgoing.sigma_tau.shape == (3,)
    assert outgoing.sigma_p.shape == (3,)
    assert outgoing.energy.shape == (3,)
    assert outgoing.total_charge.shape == (3,)
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (3, 100_000)


@pytest.mark.parametrize("BeamClass", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_vectorized_undulator(BeamClass):
    """Test that a vectorized `Undulator` is able to track a particle beam."""
    element = cheetah.Undulator(length=torch.tensor([0.4, 0.7]))
    incoming = BeamClass.from_parameters(sigma_x=torch.tensor(1e-5))

    outgoing = element.track(incoming)

    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particles.shape == (2, 100_000, 7)
    assert outgoing.mu_x.shape == (2,)
    assert outgoing.mu_px.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_py.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_px.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_py.shape == (2,)
    assert outgoing.sigma_tau.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (2, 100_000)


@pytest.mark.parametrize("BeamClass", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_vectorized_solenoid(BeamClass):
    """Test that a vectorized `Solenoid` is able to track a particle beam."""
    element = cheetah.Solenoid(
        length=torch.tensor([0.4, 0.7]), k=torch.tensor([4.2, 3.1])
    )
    incoming = BeamClass.from_parameters(sigma_x=torch.tensor(1e-5))

    outgoing = element.track(incoming)

    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particles.shape == (2, 100_000, 7)
    assert outgoing.mu_x.shape == (2,)
    assert outgoing.mu_px.shape == (2,)
    assert outgoing.mu_y.shape == (2,)
    assert outgoing.mu_py.shape == (2,)
    assert outgoing.sigma_x.shape == (2,)
    assert outgoing.sigma_px.shape == (2,)
    assert outgoing.sigma_y.shape == (2,)
    assert outgoing.sigma_py.shape == (2,)
    assert outgoing.sigma_tau.shape == (2,)
    assert outgoing.sigma_p.shape == (2,)
    assert outgoing.energy.shape == (2,)
    assert outgoing.total_charge.shape == (2,)
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (2, 100_000)
