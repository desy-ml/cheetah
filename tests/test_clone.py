import pytest
import torch

import cheetah


@pytest.mark.for_every_element("element")
def test_element_buffer_contents_and_location(element):
    """
    Test that the buffers of cloned elements have the same content while not sharing the
    same memory location.
    """
    clone = element.clone()

    for feature in element.defining_tensors:
        mwe_feature = getattr(element, feature)
        clone_feature = getattr(clone, feature)

        assert torch.allclose(mwe_feature, clone_feature, equal_nan=True)
        assert not mwe_feature.data_ptr() == clone_feature.data_ptr()


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
def test_beam_buffer_contents_and_location(BeamClass):
    """
    Test that the buffers of cloned beams have the same content while not sharing the
    same memory location.
    """
    beam = BeamClass.from_parameters(species=cheetah.Species("positron"))
    clone = beam.clone()

    for attribute in beam.UNVECTORIZED_NUM_ATTR_DIMS.keys():
        beam_attribute = getattr(beam, attribute)
        cloned_attribute = getattr(clone, attribute)

        assert torch.allclose(beam_attribute, cloned_attribute)
        assert not beam_attribute.data_ptr() == cloned_attribute.data_ptr()

    assert beam.species.name == clone.species.name
    assert beam.species.num_elementary_charges == clone.species.num_elementary_charges
    assert beam.species.mass_eV == clone.species.mass_eV
    assert id(beam.species) != id(clone.species)
