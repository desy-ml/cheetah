class PhysicsWarning(Warning):
    """
    Base class for all warnings in Cheetah that indicate a potentially incorrect
    physics.
    """

    ...


class DirtyNameWarning(Warning):
    """
    Warning raised when an element's name is not clean, i.e. it contains characters
    "that are not valid for use in Python variable names and therefore prevent the use "
    of the `segment.element_name` syntax to access elements in a segment.
    """

    ...
