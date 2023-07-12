import os
import test.ARESlatticeStage3v1_9 as ares

from cheetah.accelerator import Segment
from cheetah.utils import load_cheetah_model, save_cheetah_model

cheetah_segment = Segment.from_ocelot(ares.cell)


def test_save_cheetah():
    save_cheetah_model(cheetah_segment, "test/test_save_cheetah.json")


def test_load_cheetah():
    cheetah_segment2 = load_cheetah_model("test/test_save_cheetah.json")
    for i, element in enumerate(cheetah_segment.elements):
        element2 = cheetah_segment2.elements[i]
        assert element.name == element2.name

    # remove file
    os.remove("test/test_save_cheetah.json")
