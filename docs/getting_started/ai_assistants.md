# Using Cheetah with AI assistants

Use `segment.summary()` and `beam.summary()` to inspect a simulation in Python.
They return dictionaries with scalar values or nested lists for vectorised tensors.
Pass `as_string=True` for a readable report. Use `segment.flattened().summary()`
to see individual elements inside nested segments.

## Agent skill

Copy the [`skills/cheetah` directory](https://github.com/desy-ml/cheetah/tree/master/skills/cheetah)
into your assistant's skills directory. The skill covers the coordinate system,
tracking and gradient-based matching. The [`llms.txt` index](../llms.txt) links to
the relevant documentation and examples.

## MCP server

Install the optional MCP extra in the Python environment that will run the server.

```sh
python -m pip install "cheetah-accelerator[mcp]"
```

Configure your MCP client to run the server over stdio. For clients with an
`mcpServers` JSON setting, use the following configuration with the absolute
path to your Python executable.

```json
{
  "mcpServers": {
    "cheetah": {
      "command": "/path/to/environment/bin/python",
      "args": ["-m", "cheetah.mcp"]
    }
  }
}
```

Each tool accepts either an absolute `lattice_path` to a local LatticeJSON file
or an inline `lattice_dict` using the [LatticeJSON format](../latticejson.rst).
The server runs with the permissions of its local process. Use trusted lattice
files and only share files you intend to make available to the MCP client.

1. Call `cheetah_inspect_lattice` to find element names, positions, and settings.
2. Call `cheetah_track_beam` with the same lattice and the entrance Twiss
   parameters. The default is an electron `ParameterBeam` at 100 MeV total
   reference energy. Use `beam_type: "particle"` for macroparticle tracking.
3. Call `cheetah_match_beam` with the same entrance parameters, the unique
   `quad_names` to tune, and `target_sigma_x` and `target_sigma_y` in metres.
   Matching requires a non-vectorised lattice. Compare the returned beam sizes
   with your targets and use the fitted `k1` values to update the simulation in Python.
   The optimizer works on an in-memory model and does not write to the lattice file.

## Inspect, track, match, and verify

Run this script with the Python environment that has the MCP extra installed.
It starts the server, sets beam-size targets from a reference quadrupole
strength, and verifies the matched sizes by tracking again. All sizes are in
metres and quadrupole strengths are in inverse square metres.

```python
import asyncio
import sys

from mcp import Client, StdioServerParameters


async def main():
    """Inspect a lattice and verify a beam match through MCP."""
    lattice = {
        "root": "line",
        "lattices": {"line": ["d1", "q1", "d2", "SCR1"]},
        "elements": {
            "d1": ["Drift", {"length": 0.5}],
            "q1": ["Quadrupole", {"length": 0.2, "k1": 2.0}],
            "d2": ["Drift", {"length": 0.5}],
            "SCR1": ["Screen", {"is_active": False}],
        },
    }
    beam_args = {
        "lattice_dict": lattice,
        "energy": 1e8,
        "beta_x": 2.0,
        "beta_y": 1.0,
        "alpha_x": 0.0,
        "alpha_y": -0.5,
        "emittance_x": 1e-6,
        "emittance_y": 1e-6,
    }
    server = StdioServerParameters(command=sys.executable, args=["-m", "cheetah.mcp"])
    async with Client(server) as client:

        async def call(name, arguments):
            """Return the tool's structured result or raise with its error."""
            result = await client.call_tool(name, arguments)
            if result.is_error:
                raise RuntimeError(result.content)
            return result.structured_content

        inspection = await call("cheetah_inspect_lattice", {"lattice_dict": lattice})
        print([element["name"] for element in inspection["elements"]])

        target = await call("cheetah_track_beam", {**beam_args, "end_element": "SCR1"})
        settings = lattice["elements"]["q1"][1]
        settings["k1"] = 0.0
        match = await call(
            "cheetah_match_beam",
            {
                **beam_args,
                "quad_names": ["q1"],
                "steps": 100,
                "target_sigma_x": target["sigma_x"],
                "target_sigma_y": target["sigma_y"],
            },
        )
        settings["k1"] = match["optimized_parameters"]["q1"]
        verified = await call("cheetah_track_beam", {**beam_args, "end_element": "SCR1"})
        for key in ("sigma_x", "sigma_y"):
            print(f"{key}: target {target[key]:.3e} m, fitted {verified[key]:.3e} m")
            assert abs(verified[key] - target[key]) < 1e-5


asyncio.run(main())
```

Use `end_element` to track through one uniquely named element from the inspection
output. Omit it to track to the lattice exit. In this example, `SCR1` is at the
exit, so the matching tool and the verification track use the same observation
point. The tracking tool returns beam properties at the selected point.
