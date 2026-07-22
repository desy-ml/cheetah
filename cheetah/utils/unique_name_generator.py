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


def merge_element_names(*names: str) -> str:
    """
    Determine the name for a merged element based on the names of the elements being
    merged.

    If elements share a long-enough common prefix followed by a short suffix (e.g.
    enumeration or "in"/"out"), the prefix is returned as the new name. Otherwise, a
    concatenation of the names is returned.

    :param names: Names of elements to merge.
    :return: Name for the merged element.
    """
    if not names:
        return ""
    if len(names) == 1:
        return names[0]

    common_prefix = os.path.commonprefix(list(names))
    clean_prefix = common_prefix.rstrip("_.- ")

    if len(clean_prefix) >= 1:
        is_valid_prefix = True
        for name in names:
            suffix = name[len(clean_prefix) :].strip("_.- ")
            if len(suffix) > 5:
                is_valid_prefix = False
                break
        if is_valid_prefix:
            return clean_prefix

    return "_".join(names)
