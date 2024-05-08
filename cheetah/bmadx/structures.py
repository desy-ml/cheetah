from collections import namedtuple
from cheetah.bmadx.constants import M_ELECTRON

Particle = namedtuple('Particle', 'x px y py z pz s p0c mc2')

Drift = namedtuple('Drift', 'L')

Quadrupole = namedtuple(
    'Quadrupole',
    [
        'L',
        'K1',
        'NUM_STEPS',
        'X_OFFSET',
        'Y_OFFSET',
        'TILT'
    ],
    defaults=(None, None, 1, 0.0, 0.0, 0.0)
)

Sextupole = namedtuple(
    'Sextupole',
    [
        'L',
        'K2',
        'NUM_STEPS',
        'X_OFFSET',
        'Y_OFFSET',
        'TILT'
    ],
    defaults=(None, None, 1, 0.0, 0.0, 0.0)
)

CrabCavity = namedtuple(
    'CrabCavity', 
    [
        'L', 
        'VOLTAGE',
        'PHI0',
        'RF_FREQUENCY',
        'NUM_STEPS',
        'X_OFFSET',
        'Y_OFFSET',
        'TILT'
    ],
    defaults=(None, None, None, None, 1, 0.0, 0.0, 0.0)
)

RFCavity = namedtuple(
    'RFCavity', 
    [
      'L', 
      'VOLTAGE',
      'PHI0',
      'RF_FREQUENCY',
      'NUM_STEPS',
      'X_OFFSET',
      'Y_OFFSET',
      'TILT'
    ],
    defaults=(None, None, None, None, 1, 0.0, 0.0, 0.0)
)

SBend = namedtuple(
    'SBend', 
    [
       'L',
       'P0C',
       'G',
       'DG',
       'E1',
       'E2',
       'FINT',
       'HGAP',
       'FINTX',
       'HGAPX',
       'FRINGE_AT',
       'FRINGE_TYPE',
       'TILT'
    ],
    defaults=(
       None,
       None, 
       None, 
       0.0, 
       0.0, 
       0.0, 
       0.0, 
       0.0, 
       0.0, 
       0.0, 
       "both_ends", 
       "linear_edge",
       0.0
    )
)