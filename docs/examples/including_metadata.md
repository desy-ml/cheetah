# Including metadata with your lattice

In _Cheetah_, elements carry a `metadata` dictionary that allows you to attach
arbitrary, serialisable annotations that are **not** used for the simulation itself.
This is useful for storing information such as control-system addresses and process
variable (PV) names, hardware identifiers, or any other bookkeeping information you want
to keep alongside the lattice.

## Setting metadata

You can set `metadata` either via the constructor or as an attribute after creation:

```python
# Metadata added when the element is created
qf_2124_t1 = cheetah.Quadrupole(
    length=torch.tensor(0.1),
    k1=torch.tensor(4.2),
    name="QF.2124.T1",
    metadata={
        "control_system": {
            "type": "DOOCS",   # Define global control system protocol
            "facility": "XFEL.MAGNETS",   # DOOCS facility name
            "device": "MAGNET.ML",   # DOOCS device name
            "location": "QF.2124.T1",   # DOOCS location name
            "properties": {
                "k1": {   # Name of the corresponding property of the Cheetah element
                    "setpoint": "K1.SP",   # DOOCS property name for setpoint
                    "readback": "K1.RBV",   # DOOCS property name for readback
                },
            },
        },
    },
)

# Metadata added after the element is created
qf_2124_t1.metadata["comment"] = "This magnet was installed 2026-06-02."
```

By default, `metadata` is an empty dictionary (`{}`).

## Schema

JSON-serializable (e.g. `str`, `int`, `float`, `bool`, `list` and nested `dict`) so that
it survives a [LatticeJSON](../latticejson.rst) save/load round trip.

One of the original intents of `metadata` was to store control-system information. In
the following there are two good practices for structuring control-system metadata for
two different control systems: DOOCS and EPICS.

### DOOCS

```python
# Example quadrupole
qf_2124_t1.metadata = {
    "control_system": {
        "type": "DOOCS",   # Define global control system protocol
        "facility": "XFEL.MAGNETS",   # DOOCS facility name
        "device": "MAGNET.ML",   # DOOCS device name
        "location": "QF.2124.T1",   # DOOCS location name
        "properties": {
            "k1": {   # Name of the corresponding property of the Cheetah element
                "setpoint": "K1.SP",   # DOOCS property name for setpoint
                "readback": "K1.RBV",   # DOOCS property name for readback
            },
        },
    },
}
```

### EPICS

A suggested structure for control-system integration:

```python
# Example quadrupole
q1.metadata = {
    "control_system": {
        "pv_base": "A:Q1:PS",   # Common PV prefix
        "setpoint": "SetCurrent",   # Write PV (relative to pv_base)
        "readback": "MeasCurrent",   # Read PV (relative to pv_base)
    },
}

# Example screen with EPICS AreaDetector Module
screen.metadata = {
    "control_system": {
        "pv_base": "CAM1:image1:",   # Common PV prefix
        "readback": "ArrayData",   # Image PV
        "callback": "EnableCallbacks",   # Enable/disable camera data streaming
    },
}

# Example hv-BPM
ap1.metadata = {
    "control_system": {
        "pv_base": "A:P1:",   # Common PV prefix
        "readback": ["X", "Y"],   # X and Y positions
    },
}

# Example segment
segment.metadata = {
    "control_system": {
        "type": "EPICS",  # Define global control system protocol
    },
}
```
