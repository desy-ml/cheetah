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
            "pv_base": "XFEL:QF:2124:T1:",
            "properties": {
                "STRENGTH.SP": "k1",
                "STRENGTH.RBV": "k1",
            },
        }
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
            "STRENGTH.SP": "k1",    # Setpoint property
            "STRENGTH.RBV": "k1",   # Readback value property
            "PS_ON": "is_active",   # Power supply state property
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
            "SetCurrent": "k1",     # EPICS setpoint maps to quadrupole k1
            "MeasCurrent": "k1",    # EPICS readback maps to quadrupole k1
        },
    },
}
```

## Online Integration Example

Below is a simple conceptual example of how a control script can use Cheetah's metadata to read magnet settings from DOOCS (using the `pydoocs` library), run a simulation, and write updated settings back.

```python
import torch
import cheetah
import pydoocs  # Assuming pydoocs is installed

# Load your lattice
segment = cheetah.Segment.from_lattice_json("my_lattice.json")

# 1. Read live values from the control system and update Cheetah elements
for element in segment.elements:
    if "control_system" in element.metadata:
        cs = element.metadata["control_system"]
        if cs.get("type") == "DOOCS":
            facility = cs.get("facility")
            device = cs.get("device")
            location = cs.get("location")

            # Map DOOCS properties to Cheetah element attributes
            for prop_name, element_attr in cs.get("properties", {}).items():
                if prop_name == "STRENGTH.RBV":
                    # Construct DOOCS address: FACILITY/DEVICE/LOCATION/PROPERTY
                    address = f"{facility}/{device}/{location}/{prop_name}"

                    # Read live value from DOOCS (value is returned in a dict)
                    live_data = pydoocs.read(address)
                    live_value = live_data["data"]

                    # Update Cheetah element attribute
                    setattr(element, element_attr, torch.tensor(live_value))

# 2. Run the simulation with live settings
outgoing_beam = segment.track(incoming_beam)

# 3. Perform optimisation/tuning and write setpoints back
# (e.g. update quadrupole settings to new values computed by an optimiser)
new_k1 = torch.tensor(4.5)
segment.elements[0].k1 = new_k1

# Write setpoint back to DOOCS
cs = segment.elements[0].metadata["control_system"]
address_sp = f"{cs['facility']}/{cs['device']}/{cs['location']}/STRENGTH.SP"
pydoocs.write(address_sp, new_k1.item())
```
