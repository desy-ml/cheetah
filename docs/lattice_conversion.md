# Lattice Conversion & Import/Export

Cheetah supports importing and converting lattices from several popular accelerator simulation codes. This makes it easy to bring existing lattice designs into Cheetah for high-speed, differentiable tracking.

## Ocelot

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

## Bmad

To convert a Bmad lattice file (`.bmad`), pass the file path to `cheetah.Segment.from_bmad()`:

```python
import cheetah

segment = cheetah.Segment.from_bmad("path/to/lattice.bmad")
```

Bmad lattices define parameters such as reference particle type and reference energy/momentum, which will be parsed. Any properties or elements that are not currently understood or supported by Cheetah will trigger a warning, but the conversion will proceed.

## Elegant

To convert an Elegant lattice file (`.lte`), pass the file path to `cheetah.Segment.from_elegant()`:

```python
import cheetah

segment = cheetah.Segment.from_elegant("path/to/lattice.lte")
```

## NX Tables

Cheetah also supports importing from NX Tables format (a CSV-based tabular lattice representation) via `cheetah.Segment.from_nx_tables()`:

```python
import cheetah

segment = cheetah.Segment.from_nx_tables("path/to/nxtables.csv")
```

## LatticeJSON (Native Serialisation)

Cheetah has native support for LatticeJSON, a standardised, human-readable format for exchange of accelerator lattice descriptions.

### Saving a Lattice

You can save a Cheetah `Segment` to a `.json` file:

```python
import cheetah

# Save using the segment method
segment.to_lattice_json("my_lattice.json")

# Or save using the helper function
cheetah.save_cheetah_model(segment, "my_lattice.json", title="My Lattice Model", info="Optional description")
```

### Loading a Lattice

To load a saved LatticeJSON file back into Cheetah:

```python
import cheetah

# Load using the segment class method
segment = cheetah.Segment.from_lattice_json("my_lattice.json")

# Or load using the helper function
segment = cheetah.load_cheetah_model("my_lattice.json")
```
