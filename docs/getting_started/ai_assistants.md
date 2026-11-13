# Using Cheetah with AI assistants

Use `segment.summary()` and `beam.summary()` to inspect a simulation in Python.
They return dictionaries with scalar values or nested lists for batched tensors.
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
   Matching requires an unbatched lattice. Compare the returned beam sizes with
   your targets and use the fitted `k1` values to update the simulation in Python.
   The optimizer works on an in-memory model and does not write to the lattice file.
