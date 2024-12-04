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
def test_element_buffer_location(ElementClass):
    """Test that the buffers of cloned elements do not share memory."""
    element = ElementClass(length=torch.tensor(1.0))
    clone = element.clone()

    for buffer, buffer_clone in zip(element.buffers(), clone.buffers()):
        assert torch.allclose(buffer, buffer_clone)
        assert not buffer.data_ptr() == buffer_clone.data_ptr()


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_beam_buffer_location(BeamClass):
    """Test that the buffers of clones beams do not share memory."""
    beam = BeamClass.from_parameters()
    clone = beam.clone()

    for buffer, buffer_clone in zip(beam.buffers(), clone.buffers()):
        assert torch.allclose(buffer, buffer_clone)
        assert not buffer.data_ptr() == buffer_clone.data_ptr()
