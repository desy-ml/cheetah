import pytest


@pytest.mark.test_all_elements
def test_element_subclasses_is_active_boolean(mwe_cheetah_element):
    """
    Test that the `is_active` property of all `Element` subclasses returns a boolean if
    the element class has an `is_active` property.
    """
    assert not hasattr(mwe_cheetah_element, "is_active") or isinstance(
        mwe_cheetah_element.is_active, bool
    )


@pytest.mark.test_all_elements
def test_all_element_subclasses_is_skippable_boolean(mwe_cheetah_element):
    """
    Test that the `is_skippable` property of all `Element` subclasses returns a boolean.
    """
    assert isinstance(mwe_cheetah_element.is_skippable, bool)
