import sys
sys.path.append("C:/Users/ftheilen/Source/ocelot")

from cheetah import accelerator
from ocelot import (Drift, Monitor)

#Drift
drift_arlisolg1 = Drift(l=0.19600000000000006, eid="Drift_ARLISOLG1")
drift_arliscrg1 = Drift(l=0.19600000000000006, eid="Drift_ARLISCRG1")

#Monitors
arlibscl1 = Monitor(eid="ARLIBSCL1")
arshscre1 = Monitor(eid="ARSHSCRE1")

#lattice
cell = (
    drift_arlisolg1,
    drift_arliscrg1,
    arlibscl1,
    arshscre1
)

segment = accelerator.Segment.from_ocelot(cell)

print(segment)