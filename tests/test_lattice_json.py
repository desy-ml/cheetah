import pytest
import torch

import cheetah

from .resources import ARESlatticeStage3v1_9 as ares


@pytest.mark.filterwarnings("ignore::cheetah.utils.DefaultParameterWarning")
def test_save_and_reload_ares_example(tmp_path):
    """
    Test that saving Cheetah `Segment` to LatticeJSON works and that it can be reloaded
    correctly at the example of the full ARES lattice.
    """
    original_segment = cheetah.Segment.from_ocelot(ares.cell, name="ARES_Segment")

    original_segment.to_lattice_json(
        str(tmp_path / "ares_lattice.json"),
        title="ARES LatticeJSON",
        info="Save and reload test for Cheetah using the ARES lattice",
    )

    reloaded_segment = cheetah.Segment.from_lattice_json(
        str(tmp_path / "ares_lattice.json")
    )

    assert original_segment.name == reloaded_segment.name
    assert len(original_segment.elements) == len(reloaded_segment.elements)
    assert original_segment.length == reloaded_segment.length

    # TODO: Improve element comparison when equality is working again
    for original_element, reloaded_element in zip(
        original_segment.elements, reloaded_segment.elements
    ):
        assert original_element.name == reloaded_element.name
        assert original_element.__class__ == reloaded_element.__class__


def test_save_and_reload_custom_transfer_map(tmp_path):
    """
    Test that saving and reloading a `CustomTransferMap` works. `CustomTransferMap`
    never appears in the ARES lattice and must therefore be tested separately.
    """
    custom_transfer_map_element = cheetah.CustomTransferMap(
        predefined_transfer_map=torch.eye(7, 7),
        length=torch.tensor(1.0),
        name="my_custom_transfer_map_element",
    )
    segment = cheetah.Segment([custom_transfer_map_element], name="test_segment")

    segment.to_lattice_json(
        str(tmp_path / "custom_transfer_map_lattice.json"),
        title="Custom Transfer Map LatticeJSON",
        info="Save and reload test for Cheetah using a custom transfer map",
    )

    reloaded_segment = cheetah.Segment.from_lattice_json(
        str(tmp_path / "custom_transfer_map_lattice.json")
    )

    # I really only care that the transfer map element is recovered correctly, the
    # segment was tested in a different test.
    reloaded_custom_transfer_map_element = reloaded_segment.elements[0]

    assert torch.allclose(
        custom_transfer_map_element.predefined_transfer_map,
        reloaded_custom_transfer_map_element.predefined_transfer_map,
    )
    assert torch.allclose(
        custom_transfer_map_element.length, reloaded_custom_transfer_map_element.length
    )
    assert custom_transfer_map_element.name == reloaded_custom_transfer_map_element.name


@pytest.mark.parametrize("desired_dtype", [torch.float32, torch.float64])
def test_desired_dtype(tmp_path, desired_dtype: torch.dtype):
    """
    Test that the lattice JSON import correctly interprets its optional dtype parameter.
    """
    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Dipole(length=torch.tensor(0.5), angle=torch.tensor(1e-3)),
            cheetah.Quadrupole(length=torch.tensor(0.3), k1=torch.tensor(31.415)),
        ]
    )
    original_segment.to_lattice_json(str(tmp_path / "dummy_lattice.json"))

    reloaded_segment = cheetah.Segment.from_lattice_json(
        str(tmp_path / "dummy_lattice.json"), dtype=desired_dtype
    )

    assert all(
        getattr(element, feature).dtype == desired_dtype
        for element in reloaded_segment.elements
        for feature in element.defining_tensors
    )
