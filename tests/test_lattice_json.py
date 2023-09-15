from cheetah.accelerator import Segment

from .resources import ARESlatticeStage3v1_9 as ares


def test_save_and_reload(tmp_path):
    """
    Test that saving Cheetah `Segment` to LatticeJSON works and that it can be reloaded
    correctly.
    """
    original_segment = Segment.from_ocelot(ares.cell, name="ARES_Segment")

    original_segment.to_lattice_json(
        str(tmp_path / "ares_lattice.json"),
        title="ARES LatticeJSON",
        info="Save and reload test for Cheetah using the ARES lattice",
    )

    reloaded_segment = Segment.from_lattice_json(str(tmp_path / "ares_lattice.json"))

    assert original_segment.name == reloaded_segment.name
    assert len(original_segment.elements) == len(reloaded_segment.elements)
    assert original_segment.length == reloaded_segment.length

    # TODO: Improve element comparison when equality is working again
    for original_element, reloaded_element in zip(
        original_segment.elements, reloaded_segment.elements
    ):
        assert original_element.name == reloaded_element.name
        assert original_element.__class__ == reloaded_element.__class__
