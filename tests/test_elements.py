import pytest
import torch


@pytest.mark.for_every_mwe_element("mwe_element")
def test_element_subclasses_is_active_boolean(mwe_element):
    """
    Test that the `is_active` property of all `Element` subclasses returns a boolean if
    the element class has an `is_active` property.
    """
    assert not hasattr(mwe_element, "is_active") or isinstance(
        mwe_element.is_active, bool
    )


@pytest.mark.for_every_mwe_element("mwe_element")
def test_all_element_subclasses_is_skippable_boolean(mwe_element):
    """
    Test that the `is_skippable` property of all `Element` subclasses returns a boolean.
    """
    assert isinstance(mwe_element.is_skippable, bool)


@pytest.mark.for_every_mwe_element("mwe_element")
def test_defining_features_dtype(mwe_element):
    """
    Test that all defining features of `Element` subclasses that are `torch.Tensor`are
    properly converted between different dtypes. This transitively tests if all defining
    features are registered as pytorch buffers.
    """

    # Ensure all features have the same dtype initially
    for feature in mwe_element.defining_tensors:
        assert getattr(mwe_element, feature).dtype == torch.float32

    mwe_element.to(torch.float64)

    # Ensure all features have been converted to float64
    for feature in mwe_element.defining_tensors:
        assert getattr(mwe_element, feature).dtype == torch.float64
