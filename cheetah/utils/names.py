import os


class UniqueNameGenerator:
    """Generates a unique name given a prefix."""

    def __init__(self, prefix: str):
        self._prefix = prefix
        self._counter = 0

    def __call__(self):
        name = f"{self._prefix}_{self._counter}"
        self._counter += 1
        return name


def merge_element_names(*names: str, use_shared_prefix: bool = True) -> str:
    """
    Determine the name for a merged element based on the names of the elements being
    merged.

    If element names share a common prefix followed by a suffix (e.g. enumeration or
    "in"/"out") and `use_shared_prefix=True`, the shared prefix is returned as the
    merged name. Otherwise, a concatenation of the names separated by `_` is returned.

    :param names: Names of elements to merge.
    :param use_shared_prefix: Whether to use the shared prefix if it exists.
    :return: Name for the merged element.
    """
    assert len(names) > 0, "At least one name must be provided."

    common_prefix = os.path.commonprefix(list(names))

    return (
        common_prefix
        if use_shared_prefix and len(common_prefix) > 0
        else "_".join(names)
    )
