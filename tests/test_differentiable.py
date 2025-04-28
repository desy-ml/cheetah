import pytest
import torch
from torch import nn

import cheetah

from .resources import ARESlatticeStage3v1_9 as ares


def test_simple_quadrupole():
    """
    Simple test on a [D, Q, D] lattice with the qudrupole's k1 requiring grad, checking
    if PyTorch tracked a grad_fn into the outgoing beam.
    """
    segment = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Quadrupole(
                length=torch.tensor(0.2),
                k1=nn.Parameter(torch.tensor(3.142)),
                name="my_quad",
            ),
            cheetah.Drift(length=torch.tensor(1.0)),
        ]
    )
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam = segment.track(incoming_beam)

    assert outgoing_beam.particles.grad_fn is not None


def test_ea_magnets():
    """
    Test that gradients are tracking when the magnet settings in the ARES experimental
    area require grad.
    """
    ea = cheetah.Segment.from_ocelot(ares.cell, warnings=False).subcell(
        "AREASOLA1", "AREABSCR1"
    )
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    ea.AREAMQZM2.k1 = nn.Parameter(ea.AREAMQZM2.k1)
    ea.AREAMQZM1.k1 = nn.Parameter(ea.AREAMQZM1.k1)
    ea.AREAMCVM1.angle = nn.Parameter(ea.AREAMCVM1.angle)
    ea.AREAMQZM3.k1 = nn.Parameter(ea.AREAMQZM3.k1)
    ea.AREAMCHM1.angle = nn.Parameter(ea.AREAMCHM1.angle)

    outgoing_beam = ea.track(incoming_beam)

    assert outgoing_beam.particles.grad_fn is not None


def test_ea_incoming_parameter_beam():
    """
    Test that gradients are tracking when incoming beam (being a `ParameterBeam`)
    requires grad.
    """
    ea = cheetah.Segment.from_ocelot(ares.cell, warnings=False).subcell(
        "AREASOLA1", "AREABSCR1"
    )
    incoming_beam = cheetah.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    incoming_beam.mu = nn.Parameter(incoming_beam.mu)
    incoming_beam.cov = nn.Parameter(incoming_beam.cov)

    outgoing_beam = ea.track(incoming_beam)

    assert outgoing_beam.mu.grad_fn is not None
    assert outgoing_beam.cov.grad_fn is not None


def test_ea_incoming_particle_beam():
    """
    Test that gradients are tracking when incoming beam (being a `ParticleBeam`)
    requires grad.
    """
    ea = cheetah.Segment.from_ocelot(ares.cell, warnings=False).subcell(
        "AREASOLA1", "AREABSCR1"
    )
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    incoming_beam.particles = nn.Parameter(incoming_beam.particles)

    outgoing_beam = ea.track(incoming_beam)

    assert outgoing_beam.particles.grad_fn is not None


@pytest.mark.parametrize(
    "ElementClass",
    [
        cheetah.Cavity,
        cheetah.Dipole,
        cheetah.Drift,
        cheetah.HorizontalCorrector,
        cheetah.Quadrupole,
        cheetah.RBend,
        cheetah.Solenoid,
        cheetah.TransverseDeflectingCavity,
        cheetah.Undulator,
        cheetah.VerticalCorrector,
    ],
)
def test_nonleaf_tracking(ElementClass):
    """
    Test that a beam with non-leaf tensors as elements can be tracked through elements
    with length parameter.
    """
    beam = cheetah.ParticleBeam.from_parameters()

    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0, requires_grad=True)),
            ElementClass(length=torch.tensor(2.0)),
        ]
    )
    segment.track(beam)


@pytest.mark.parametrize(
    "ElementClass", [cheetah.Aperture, cheetah.BPM, cheetah.Screen]
)
def test_nonleaf_lenghtless_elements(ElementClass):
    """
    Test that a beam with non-leaf tensors as elements can be tracked through elements
    without length parameter.

    The split into lengthless elements is necessary since there is no common constructor
    for all element classes. Some require a length, some cannot handle a length
    argument.
    """
    beam = cheetah.ParticleBeam.from_parameters()

    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0, requires_grad=True)),
            ElementClass(is_active=True),
        ]
    )
    segment.track(beam)


def test_parameters_at_initialization():
    """
    Test that passing a `torch.nn.Parameter` at initialization registeres the parameter
    in the same way as an assignment after initialization.
    """
    dipole_with_buffer = cheetah.Dipole(length=torch.tensor(1.0))

    # Dipole with buffer (without parameter) should not have any parameters
    assert len(list(dipole_with_buffer.parameters())) == 0

    # Create two dipoles with the same parameter, one passed at initialization and one
    # assigned after initialization.
    parameter = torch.nn.Parameter(torch.tensor(0.2))
    dipole_initial = cheetah.Dipole(length=torch.tensor(1.0), angle=parameter)
    dipole_assigned = cheetah.Dipole(length=torch.tensor(1.0))
    dipole_assigned.angle = parameter

    # Both dipoles should have the same parameter (the originally passed one and one in
    # total)
    assert list(dipole_initial.parameters()) == list(dipole_assigned.parameters())
    assert len(list(dipole_initial.parameters())) == 1
    assert parameter in dipole_initial.parameters()
