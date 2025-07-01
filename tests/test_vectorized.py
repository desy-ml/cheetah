import numpy as np
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
    Test that the shape of a segment's length matches the input for a vectorisation with
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
    assert outgoing.energy.shape == torch.Size([])
    assert outgoing.total_charge.shape == torch.Size([])
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (100_000,)


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
    assert outgoing.energy.shape == torch.Size([])
    assert outgoing.total_charge.shape == torch.Size([])
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (100_000,)


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
    assert outgoing.energy.shape == torch.Size([])
    assert outgoing.total_charge.shape == torch.Size([])
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (100_000,)


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
    assert outgoing.energy.shape == torch.Size([])
    assert outgoing.total_charge.shape == torch.Size([])
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (100_000,)


@pytest.mark.filterwarnings("ignore::cheetah.utils.DefaultParameterWarning")
def test_enormous_through_ares_ea():
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
    assert outgoing.energy.shape == torch.Size([])
    assert outgoing.total_charge.shape == torch.Size([])


@pytest.mark.parametrize("BeamClass", [cheetah.ParticleBeam, cheetah.ParameterBeam])
def test_cavity_with_zero_and_non_zero_voltage(BeamClass):
    """
    Tests that if zero and non-zero voltages are passed to a cavity in a single batch,
    there are no errors. This test does NOT check physical correctness.
    """
    cavity = cheetah.Cavity(
        length=torch.tensor(3.0441),
        voltage=torch.tensor([0.0, 48_198_468.0, 0.0]),
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
    assert outgoing.total_charge.shape == torch.Size([])
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (100_000,)


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
    assert outgoing.energy.shape == torch.Size([])
    assert outgoing.total_charge.shape == torch.Size([])
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (100_000,)


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
    assert outgoing.energy.shape == torch.Size([])
    assert outgoing.total_charge.shape == torch.Size([])
    if BeamClass == cheetah.ParticleBeam:
        assert outgoing.particle_charges.shape == (100_000,)


@pytest.mark.parametrize("BeamClass", [cheetah.ParticleBeam])
@pytest.mark.parametrize("method", ["kde"])  # Currently only KDE supports vectorisation
def test_vectorized_screen_2d(BeamClass, method):
    """
    Test that a vectorized `Screen` is able to track a particle beam and produce a
    reading with 2 vector dimensions.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Screen(
                resolution=(100, 100),
                pixel_size=torch.tensor((1e-5, 1e-5)),
                misalignment=torch.tensor(
                    [
                        [[1e-4, 2e-4], [3e-4, 4e-4], [5e-4, 6e-4]],
                        [[-1e-4, -2e-4], [-3e-4, -4e-4], [-5e-4, -6e-4]],
                    ]
                ),
                is_active=True,
                method=method,
                name="my_screen",
            ),
        ],
        name="my_segment",
    )
    incoming = BeamClass.from_parameters(sigma_x=torch.tensor(1e-5))

    _ = segment.track(incoming)

    # Check the reading
    assert segment.my_screen.reading.shape == (2, 3, 100, 100)


@pytest.mark.for_every_element(
    "element_with_length",
    except_for=[
        cheetah.Aperture,
        cheetah.BPM,
        cheetah.CustomTransferMap,
        cheetah.Marker,
        cheetah.Screen,
        cheetah.Segment,
        cheetah.SpaceChargeKick,
    ],
)
def test_broadcasting_two_different_inputs(element_with_length):
    """
    Test that broadcasting rules are correctly applied to a elements with two different
    input shapes for elements that have a `length` attribute.

    Skipped for elements whose response is not influenced by their length.
    """
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, energy=torch.tensor([154e6, 14e9])
    )

    element_with_length.length = torch.tensor([[0.6], [0.5], [0.4]])
    outgoing = element_with_length.track(incoming)

    assert outgoing.particles.shape == (3, 2, 100_000, 7)
    assert outgoing.particle_charges.shape == (100_000,)
    assert outgoing.energy.shape == (2,)


@pytest.mark.parametrize(
    "ElementClass",
    [
        cheetah.Dipole,
        cheetah.Drift,
        cheetah.Quadrupole,
        cheetah.TransverseDeflectingCavity,
    ],
)
def test_broadcasting_two_different_inputs_bmadx(ElementClass):
    """
    Test that broadcasting rules are correctly applied to a elements with two different
    input shapes for elements that have a `"bmadx"` tracking method.
    """
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, energy=torch.tensor([154e6, 14e9])
    )
    element = ElementClass(
        tracking_method="bmadx", length=torch.tensor([[0.6], [0.5], [0.4]])
    )

    outgoing = element.track(incoming)

    assert outgoing.particles.shape == (3, 2, 100_000, 7)
    assert outgoing.particle_charges.shape == (100_000,)
    assert outgoing.energy.shape == (2,)


def test_vectorized_parameter_beam_creation():
    """
    Tests that creating a parameter beam with a few vectorised parameters works as
    expected.
    """
    beam = cheetah.ParameterBeam.from_parameters(
        mu_x=torch.tensor([2e-4, 3e-4]), sigma_x=torch.tensor([1e-5, 2e-5])
    )

    assert beam.mu_x.shape == (2,)
    assert torch.allclose(beam.mu_x, torch.tensor([2e-4, 3e-4]))
    assert beam.sigma_x.shape == (2,)
    assert torch.allclose(beam.sigma_x, torch.tensor([1e-5, 2e-5]))


@pytest.mark.parametrize(
    "ElementClass", [cheetah.HorizontalCorrector, cheetah.VerticalCorrector]
)
def test_broadcasting_corrector_angles(ElementClass):
    """Test that broadcasting rules are correctly applied to with corrector angles."""
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, energy=torch.tensor([154e6, 14e9])
    )
    element = ElementClass(
        length=torch.tensor(0.15), angle=torch.tensor([[1e-5], [2e-5], [3e-5]])
    )

    outgoing = element.track(incoming)

    assert outgoing.particles.shape == (3, 2, 100_000, 7)
    assert outgoing.particle_charges.shape == (100_000,)
    assert outgoing.energy.shape == (2,)


def test_broadcasting_solenoid_misalignment():
    """
    Test that broadcasting rules are correctly applied to the misalignment in solenoids.
    """
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, energy=torch.tensor([154e6, 14e9])
    )
    element = cheetah.Solenoid(
        length=torch.tensor(0.15),
        misalignment=torch.tensor(
            [
                [[1e-5, 2e-5], [2e-5, 3e-5]],
                [[3e-5, 4e-5], [4e-5, 5e-5]],
                [[5e-5, 6e-5], [6e-5, 7e-5]],
            ]
        ),
    )

    outgoing = element.track(incoming)

    assert outgoing.particles.shape == (3, 2, 100_000, 7)
    assert outgoing.particle_charges.shape == (100_000,)
    assert outgoing.energy.shape == (2,)


@pytest.mark.parametrize("aperture_shape", ["rectangular", "elliptical"])
def test_vectorized_aperture_broadcasting(aperture_shape):
    """
    Test that apertures work in a vectorised setting and that broadcasting rules are
    applied correctly.
    """
    incoming = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000,
        sigma_py=torch.tensor(1e-4),
        sigma_px=torch.tensor(2e-4),
        energy=torch.tensor([154e6, 14e9]),
    )
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.Aperture(
                x_max=torch.tensor([[1e-5], [2e-4], [3e-4]]),
                y_max=torch.tensor(2e-4),
                shape=aperture_shape,
            ),
            cheetah.Drift(length=torch.tensor(0.5)),
        ]
    )

    outgoing = segment.track(incoming)

    # Particle positions are unaffected by the aperture ... only their survival is
    assert outgoing.particles.shape == (2, 100_000, 7)
    assert outgoing.energy.shape == (2,)
    assert outgoing.particle_charges.shape == (100_000,)
    assert outgoing.survival_probabilities.shape == (3, 2, 100_000)

    if aperture_shape == "elliptical":
        assert np.allclose(
            outgoing.survival_probabilities.mean(dim=-1)[:, 0],
            [0.0235, 0.42, 0.552],
            atol=5e-3,  # Last digit off by five
        )
    elif aperture_shape == "rectangular":
        assert np.allclose(
            outgoing.survival_probabilities.mean(dim=-1)[:, 0],
            [0.029, 0.495, 0.629],
            atol=5e-3,  # Last digit off by five
        )
