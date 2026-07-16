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

### Bmad Syntax Example

Below is an example of a simple Bmad lattice file that Cheetah can parse and convert:

```none
! Lattice file: simple.bmad
beginning[beta_a] = 10. ! m a-mode beta function
beginning[beta_b] = 10. ! m b-mode beta function
beginning[e_tot] = 10e6 ! eV reference energy

parameter[geometry] = open
parameter[particle] = electron ! Reference particle

abs = -0.6

d: drift, L = 0.5 * (0.3 + 0.7)
b: sbend, L = 0.6  -0.1, g = 1, e1 = 0.1, dg = sqrt(0.000001)
n: drift, L = -0.4
q: quadrupole, L = abs(abs), k1 = 0.23
s: sextupole, tilt = -0.1, L = 0.3, k2 = 0.42
v: drift, l = -q[l]

lat: line = (d, b, n, q, s, v) ! List of lattice elements
use, lat ! Line used to construct the lattice
```

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

## LatticeJSON (Native Serialization)

Cheetah has native support for LatticeJSON, a standardized, human-readable format for exchange of accelerator lattice descriptions.

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
