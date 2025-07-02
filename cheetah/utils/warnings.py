class PhysicsWarning(Warning):
    """
    Base class for all warnings in Cheetah that indicate a potentially incorrect
    physics.
    """

    ...


class UnknownElementWarning(PhysicsWarning):
    """
    Warning raised when an element is encountered that is not understood by Cheetah.
    This can happen when importing a lattice from another code, like Bmad or Elegant,
    and the element is not supported by Cheetah.
    """

    ...


class NotUnderstoodPropertyWarning(PhysicsWarning):
    """
    Warning raised when a property not understood by Cheetah is encountered during
    reading of a Fortran Namelist file, like it is done in the Bmad and Elegant
    importers.
    """

    ...


class NoBeamPropertiesInLatticeWarning(PhysicsWarning):
    """
    Warning raised when a beam property is encountered in a Bamd or Elegant lattice. In
    Cheetah, beam properties are not stored in the lattice, but in the `Beam` class.
    """

    ...


class DefaultParameterWarning(PhysicsWarning):
    """
    Warning raised when a default parameters are assumed for an element that does not
    have certain parameters known to Cheetah in its original lattice format.
    """

    ...


class DirtyNameWarning(Warning):
    """
    Warning raised when an element's name is not clean, i.e. it contains characters
    "that are not valid for use in Python variable names and therefore prevent the use "
    of the `segment.element_name` syntax to access elements in a segment.
    """

    ...


class NoVisualizationWarning(Warning):
    """
    Warning raised when an element does not have a visualization method, i.e. it cannot
    be converted to a 3D mesh.
    """

    ...
