# Control Systems Integration

Cheetah elements can carry a `metadata` dictionary. This dictionary allows you to attach arbitrary, serialisable annotations that are **not** used by the physics simulation itself, but are crucial for online control and integration with control systems like EPICS or DOOCS.

## Setting Metadata

Metadata can be set either via the constructor when creating an element or as an attribute after creation:

```python
import torch
import cheetah

# Set metadata during instantiation
quadrupole = cheetah.Quadrupole(
    length=torch.tensor(0.1),
    k1=torch.tensor(4.2),
    name="QF.2124.T1",
    metadata={
        "control_system": {
            "type": "DOOCS",
            "facility": "XFEL.MAGNETS",
            "device": "MAGNET.ML",
            "location": "QF.2124.T1",
            "properties": {
                "STRENGTH.SP": "k1",  # Setpoint property
                "STRENGTH.RBV": "k1",  # Readback value property
                "PS_ON": "is_active",  # Power supply state property
            },
        },
    },
)

# Modify metadata later
quadrupole.metadata["comment"] = "Calibrated on 2026-07-16."
```

## Recommended Schemas

To maintain clean and standardised integration, the following schemas are recommended for DOOCS and EPICS.

### DOOCS Schema

```python
metadata = {
    "control_system": {
        "type": "DOOCS",
        "facility": "XFEL.MAGNETS",
        "device": "MAGNET.ML",
        "location": "QF.2124.T1",
        "properties": {
            "STRENGTH.SP": "k1",  # Setpoint property
            "STRENGTH.RBV": "k1",  # Readback value property
            "PS_ON": "is_active",  # Power supply state property
        },
    },
}
```

### EPICS Schema

For EPICS integration, you can specify PV bases and properties:

```python
metadata = {
    "control_system": {
        "type": "EPICS",
        "pv_base": "A:Q1:PS:",
        "properties": {
            "SetCurrent": "k1",  # EPICS setpoint maps to quadrupole k1
            "MeasCurrent": "k1",  # EPICS readback maps to quadrupole k1
        },
    },
}
```

## Online Integration Example

Below is a simple conceptual example of how a control script can use Cheetah's metadata to read magnet settings from DOOCS (using the `doocs4py` package) to synchronise a Cheetah-based digital twin with the live accelerator.

```python
import cheetah
import doocs4py
import torch

# Load your lattice
segment = cheetah.Segment.from_lattice_json("my_lattice.json")

# 1. Read live values from the control system and update Cheetah elements
for element in segment.elements:
    control_system = element.metadata.get("control_system", {})
    properties = control_system.get("properties", {})
    for property_name, element_attr in properties.items():
        if property_name.endswith(".RBV"):
            facility = control_system["facility"]
            device = control_system["device"]
            location = control_system["location"]

            # Construct DOOCS address: FACILITY/DEVICE/LOCATION/PROPERTY
            address = f"{facility}/{device}/{location}/{property_name}"

            # Read live value from DOOCS
            live_value = doocs4py.get(address).get_data()

            # Update Cheetah element attribute
            setattr(element, element_attr, torch.tensor(live_value))

# 2. Run the simulation with live settings
outgoing_beam = segment.track(incoming_beam)
```
