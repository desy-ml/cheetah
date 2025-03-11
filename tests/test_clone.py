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
def test_element_buffer_contents_and_location(ElementClass):
    """
    Test that the buffers of cloned elements have the same content while not sharing the
    same memory location.
    """
    element = ElementClass(length=torch.tensor(1.0))
    clone = element.clone()

    for buffer, buffer_clone in zip(element.buffers(), clone.buffers()):
        assert torch.allclose(buffer, buffer_clone)
        assert not buffer.data_ptr() == buffer_clone.data_ptr()


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_beam_buffer_contents_and_location(BeamClass):
    """
    Test that the buffers of cloned beams have the same content while not sharing the
    same memory location.
    """
    beam = BeamClass.from_parameters(species=cheetah.Species("proton"))
    clone = beam.clone()

    for buffer, buffer_clone in zip(beam.buffers(), clone.buffers()):
        assert torch.allclose(buffer, buffer_clone)
        assert not buffer.data_ptr() == buffer_clone.data_ptr()

    assert beam.species.name == clone.species.name
    assert beam.species.num_elementary_charges == clone.species.num_elementary_charges
    assert beam.species.mass_eV == clone.species.mass_eV
    assert id(beam.species) != id(clone.species)
