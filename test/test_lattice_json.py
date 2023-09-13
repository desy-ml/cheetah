import os
import test.ARESlatticeStage3v1_9 as ares

import pytest

from cheetah.accelerator import Segment
from cheetah.latticejson import load_cheetah_model, save_cheetah_model

cheetah_segment = Segment.from_ocelot(ares.cell, name="ARES_Segment")


@pytest.mark.skip(
    reason="Lattice JSON loading and saving is broken with torch.Tensors."
)
def test_save_cheetah():
    """Test that saving Cheetah segment to lattice JSON doesn't throw an error."""
    # TODO: Use temporary directory
    save_cheetah_model(
        cheetah_segment,
        "test/test_save_cheetah.json",
        metadata={
            "version": "1.0",
            "title": "ARES Lattice",
            "info": "JSON file for ARESlatticeStage3v1_9",
            "root": "cell",
        },
    )


@pytest.mark.skip(
    reason="Lattice JSON loading and saving is broken with torch.Tensors."
)
def test_load_cheetah():
    """
    Test that loading Cheetah segment reproduces the segment that was originally saved
    in the save test.
    """
    # TODO: Use pytest feature more properly
    cheetah_segment2 = load_cheetah_model(
        "test/test_save_cheetah.json", name="ARES_Segment"
    )
    for i, element in enumerate(cheetah_segment.elements):
        element2 = cheetah_segment2.elements[i]
        assert element.name == element2.name
    assert cheetah_segment2 == cheetah_segment

    # remove file
    os.remove("test/test_save_cheetah.json")
