import sys
sys.path.append("C:/Users/ftheilen/Source/ocelot")

from cheetah import utils
from ocelot import (Drift, Monitor)

#Drift
drift_arlisolg1 = Drift(l=0.19600000000000006, eid="Drift_ARLISOLG1")

#Monitors
arlibscl1 = Monitor(eid="ARLIBSCL1")
arshscre1 = Monitor(eid="ARSHSCRE1")

#lattice
cell = (
    drift_arlisolg1,
    arlibscl1,
    arshscre1
)

segment = utils.ocelot2cheetah(cell)

print(segment)