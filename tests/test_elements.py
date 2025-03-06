import pytest
import torch

import cheetah

ELEMENT_CLASSES_REQUIRING_ARGS = {
    cheetah.Cavity: {"length": torch.tensor(1.0)},
    cheetah.CustomTransferMap: {"predefined_transfer_map": torch.eye(7)},
    cheetah.Drift: {"length": torch.tensor(1.0)},
    cheetah.Dipole: {"length": torch.tensor(1.0)},
    cheetah.HorizontalCorrector: {"length": torch.tensor(1.0)},
    cheetah.Quadrupole: {"length": torch.tensor(1.0)},
    cheetah.Segment: {"elements": [cheetah.Drift(length=torch.tensor(1.0))]},
    cheetah.Solenoid: {"length": torch.tensor(1.0)},
    cheetah.SpaceChargeKick: {"effect_length": torch.tensor(1.0)},
    cheetah.TransverseDeflectingCavity: {"length": torch.tensor(1.0)},
    cheetah.Undulator: {"length": torch.tensor(1.0)},
    cheetah.VerticalCorrector: {"length": torch.tensor(1.0)},
}


@pytest.mark.parametrize("ElementClass", cheetah.Element.__subclasses__())
def test_element_subclasses_is_active_boolean(ElementClass):
    """
    Test that the `is_active` property of all `Element` subclasses returns a boolean if
    the element class has an `is_active` property.
    """
    if ElementClass in ELEMENT_CLASSES_REQUIRING_ARGS:
        element = ElementClass(**ELEMENT_CLASSES_REQUIRING_ARGS[ElementClass])
    else:
        element = ElementClass()

    assert not hasattr(element, "is_active") or isinstance(element.is_active, bool)


@pytest.mark.parametrize("ElementClass", cheetah.Element.__subclasses__())
def test_all_element_subclasses_is_skippable_boolean(ElementClass):
    """
    Test that the `is_skippable` property of all `Element` subclasses returns a boolean.
    """
    if ElementClass in ELEMENT_CLASSES_REQUIRING_ARGS:
        element = ElementClass(**ELEMENT_CLASSES_REQUIRING_ARGS[ElementClass])
    else:
        element = ElementClass()

    assert isinstance(element.is_skippable, bool)
