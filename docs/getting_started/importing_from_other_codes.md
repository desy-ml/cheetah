# Importing from Other Codes

Cheetah provides utilities to import both accelerator lattices and particle beams from other simulation frameworks and file formats.

## Lattices

Cheetah supports converting lattices from other tracking codes and file formats into Cheetah segments.

### Ocelot

Lattices defined as Ocelot cells (lists of Ocelot elements) can be converted directly in memory using `cheetah.Segment.from_ocelot()`:

```python
import ocelot
import cheetah

# Define an Ocelot cell
ocelot_cell = [
    ocelot.Drift(l=1.0),
    ocelot.Quadrupole(l=0.2, k1=4.2),
    ocelot.Drift(l=1.0),
]

# Convert to a Cheetah Segment
segment = cheetah.Segment.from_ocelot(ocelot_cell)
```

### Bmad

To convert a Bmad lattice file (`.bmad`), pass the file path to `cheetah.Segment.from_bmad()`:

```python
import cheetah

segment = cheetah.Segment.from_bmad("path/to/lattice.bmad")
```

Bmad lattices define parameters such as reference particle type and reference energy/momentum, which will be parsed. Any properties or elements that are not currently understood or supported by Cheetah will trigger a warning, but the conversion will proceed.

### Elegant

To convert an Elegant lattice file (`.lte`), pass the file path to `cheetah.Segment.from_elegant()`:

```python
import cheetah

segment = cheetah.Segment.from_elegant("path/to/lattice.lte", name="ares_line")
```

### NX Tables

Cheetah also supports importing from NX Tables format (a CSV-based tabular lattice representation) via `cheetah.Segment.from_nx_tables()`:

```python
import cheetah

segment = cheetah.Segment.from_nx_tables("path/to/nxtables.csv")
```

### LatticeJSON (Native Serialisation)

Cheetah has native support for LatticeJSON, a standardised, human-readable format for exchange of accelerator lattice descriptions.

#### Saving a Lattice

You can save a Cheetah `Segment` to a `.json` file:

```python
import cheetah

# Save using the segment method
segment.to_lattice_json("my_lattice.json")

# Or save using the helper function
from cheetah.latticejson import save_cheetah_model
save_cheetah_model(segment, "my_lattice.json", title="My Lattice Model", info="Optional description")
```

#### Loading a Lattice

To load a saved LatticeJSON file back into Cheetah:

```python
import cheetah

# Load using the segment class method
segment = cheetah.Segment.from_lattice_json("my_lattice.json")

# Or load using the helper function
from cheetah.latticejson import load_cheetah_model
segment = load_cheetah_model("my_lattice.json")
```

## Beams

Cheetah supports importing particle beams and beam parameter distributions from several simulation codes and standard formats.

### Astra

To import a beam from an Astra output file, use `from_astra()` on `cheetah.ParticleBeam` or `cheetah.ParameterBeam`:

```python
import cheetah

beam = cheetah.ParticleBeam.from_astra("path/to/astra_beam.txt")
```

### Elegant

To import a beam from an Elegant output file, use `from_elegant()` on `cheetah.ParticleBeam`:

```python
import cheetah

beam = cheetah.ParticleBeam.from_elegant("path/to/elegant_beam.txt")
```

### Ocelot

To import a beam from an Ocelot beam object, use `from_ocelot()` on `cheetah.ParticleBeam` or `cheetah.ParameterBeam`:

```python
import cheetah

beam = cheetah.ParticleBeam.from_ocelot(ocelot_beam)
```

### openPMD

Cheetah supports the openPMD standard (via the `openpmd-beamphysics` library) to load standard beam distribution datasets:

```python
import torch
import cheetah

# Load from an openPMD HDF5/JSON file
beam = cheetah.ParticleBeam.from_openpmd_file(
    "path/to/beam.h5",
    energy=torch.tensor(1e7)
)

# Load directly from an openpmd-beamphysics ParticleGroup object
beam = cheetah.ParticleBeam.from_openpmd_particlegroup(
    particle_group,
    energy=torch.tensor(1e7)
)
```

*Note: Using openPMD requires the `[openpmd]` extra dependency. Install it using `pip install "cheetah-accelerator[openpmd]"`.*

### Twiss Parameters & Standard Distributions

You can also generate beams from Twiss parameters or multi-dimensional parameters:

```python
import torch
import cheetah

# Create a parameter beam from Twiss parameters
beam_param = cheetah.ParameterBeam.from_twiss(
    beta_x=torch.tensor(3.14),
    beta_y=torch.tensor(3.14),
    energy=torch.tensor(1e7)
)

# Create a particle beam from Twiss parameters
beam_part = cheetah.ParticleBeam.from_twiss(
    num_particles=10000,
    beta_x=torch.tensor(3.14),
    beta_y=torch.tensor(3.14),
    energy=torch.tensor(1e7)
)
```
