from . import converters  # noqa: F401

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except Exception:
        __version__ = "unknown"
from .accelerator import (  # noqa: F401
    BPM,
    Aperture,
    Cavity,
    CombinedCorrector,
    CustomTransferMap,
    Dipole,
    Drift,
    Element,
    HorizontalCorrector,
    Marker,
    Quadrupole,
    RBend,
    Screen,
    Segment,
    Sextupole,
    Solenoid,
    SpaceChargeKick,
    TransverseDeflectingCavity,
    Undulator,
    VerticalCorrector,
)
from .particles import Beam, ParameterBeam, ParticleBeam, Species  # noqa: F401
from .utils import (  # noqa: F401
    DefaultParameterWarning,
    DirtyNameWarning,
    NoBeamPropertiesInLatticeWarning,
    NotUnderstoodPropertyWarning,
    PhysicsWarning,
    UnknownElementWarning,
    VisualizationWarning,
)
