# Element Metadata

Every _Cheetah_ `Element` carries an optional `metadata` dictionary. It
is meant for arbitrary, serializable annotations that are **not** used during simulation,
such as control-system process variable (PV) names, hardware identifiers, or any other
bookkeeping information you want to keep alongside the lattice.

```{warning}
The simulation never reads `metadata`. It does not influence tracking, transfer maps, or
any physics. Do not store quantities that affect the simulation here — use the element's
defining features (e.g. `k1`, `length`) for those.
```

## Setting metadata

You can set `metadata` either via the constructor or as an attribute after creation:

```python
import torch
import cheetah

q1 = cheetah.Quadrupole(
    length=torch.tensor(0.1),
    k1=torch.tensor(4.2),
    name="Q1",
    metadata={
        "control_system": {
            "pv_base": "A:Q1:",
            "setpoint": "SetCurrent",
            "readback": "MeasCurrent",
        },
    },
)

# Or assign/extend later
q1.metadata["comment"] = "This magnet is installed 2026-06"
```

By default, `metadata` is an empty dictionary (`{}`).

## Schema

_Cheetah_ does **not** enforce a schema on `metadata`. You are free to structure it
however suits your application. The only requirement is that the contents should be
JSON-serializable (e.g. `str`, `int`, `float`, `bool`, `list`, and nested `dict`) so that
it survives a [LatticeJSON](latticejson.rst) save/load round trip.

A suggested structure for control-system integration:

```python
# For a Quadrupole
q1.metadata = {
    "control_system": {
        "pv_base": "A:Q1:PS",       # common PV prefix
        "setpoint": "SetCurrent", # write PV (relative to pv_base)
        "readback": "MeasCurrent" # read PV (relative to pv_base)
    }
}
# For a Screen with EPICS AreaDetector Module
screen.metadata = {
    "control_system": {
        "pv_base": "CAM1:image1:",
        "readback":"ArrayData",  # image PV
        "callback": "EnableCallbacks",  # enable/disable camera data streaming
    }
}

# For a hv-BPM
ap1.metadata = {
    "control_system": {
        "pv_base": "A:P1:",
        "readback": ["X", "Y"], # X and Y positions
    }
}

# For a segment
segment.metadata = {
    "control_system": {
        "type": "EPICS",  # define global control system protocol
    }
}
```

## Persistence

`metadata` is preserved by `Element.clone()` (deep-copied, so clones do not share the
same dictionary) and is written to / read from LatticeJSON files. Elements with an empty
`metadata` dictionary omit the field from the saved JSON.
