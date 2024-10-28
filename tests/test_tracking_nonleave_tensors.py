import pytest
import torch

import cheetah


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
def test_nonleave_tracking(ElementClass):
    """
    Test that a beam with non-leave tensors as elements can be tracked through
    elements with length parameter.
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
    "ElementClass",
    [
        cheetah.Aperture,
        cheetah.BPM,
        cheetah.Screen,
    ],
)
def test_nonleave_lenghtless_elements(ElementClass):
    """
    Test that a beam with non-leave tensors as elements can be tracked through
    elements without length parameter.

    The split into lengthless elements is necessary since there is no common
    constructor for all element classes. Some require a length, some cannot
    handle a length argument.
    """
    beam = cheetah.ParticleBeam.from_parameters()

    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0, requires_grad=True)),
            ElementClass(is_active=True),
        ]
    )
    segment.track(beam)
