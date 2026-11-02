"""
Model Context Protocol (MCP) server for Cheetah accelerator physics simulations.

Provides stdio-based MCP tools for:
- cheetah_inspect_lattice: Inspects lattice elements, cumulative positions,
  and properties.
- cheetah_track_beam: Tracks a parameter or particle beam through an
  accelerator lattice.
- cheetah_match_beam: Optimizes quadrupole strengths using PyTorch autograd
  gradients.
"""

import json
import sys
from typing import Any

import torch

import cheetah
from cheetah import latticejson


def load_lattice(
    lattice_path: str | None = None, lattice_dict: dict | None = None
) -> cheetah.Segment:
    """Load a Cheetah segment from either a file path or a lattice dictionary."""
    if lattice_path is not None:
        return cheetah.Segment.from_lattice_json(lattice_path)
    if lattice_dict is not None:
        root_lattice = lattice_dict.get("root", next(iter(lattice_dict["lattices"])))
        return latticejson.parse_segment(root_lattice, lattice_dict)
    raise ValueError("Either lattice_path or lattice_dict must be provided.")


def create_beam(
    beam_type: str = "parameter",
    energy: float = 1e8,
    beta_x: float = 1.0,
    beta_y: float = 1.0,
    alpha_x: float = 0.0,
    alpha_y: float = 0.0,
    emittance_x: float = 1e-6,
    emittance_y: float = 1e-6,
    num_particles: int = 1000,
) -> cheetah.Beam:
    """Create a beam object from Twiss parameters."""
    kwargs = {
        "energy": torch.tensor(energy),
        "beta_x": torch.tensor(beta_x),
        "beta_y": torch.tensor(beta_y),
        "alpha_x": torch.tensor(alpha_x),
        "alpha_y": torch.tensor(alpha_y),
        "emittance_x": torch.tensor(emittance_x),
        "emittance_y": torch.tensor(emittance_y),
    }
    if beam_type.lower() == "particle":
        return cheetah.ParticleBeam.from_twiss(num_particles=num_particles, **kwargs)
    return cheetah.ParameterBeam.from_twiss(**kwargs)


def tool_inspect_lattice(arguments: dict[str, Any]) -> dict[str, Any]:
    """Inspect elements and properties of a lattice."""
    segment = load_lattice(
        lattice_path=arguments.get("lattice_path"),
        lattice_dict=arguments.get("lattice_dict"),
    )
    return segment.summary(as_string=False)


def tool_track_beam(arguments: dict[str, Any]) -> dict[str, Any]:
    """Track a beam through a lattice and return the beam summary."""
    segment = load_lattice(
        lattice_path=arguments.get("lattice_path"),
        lattice_dict=arguments.get("lattice_dict"),
    )
    beam = create_beam(
        beam_type=arguments.get("beam_type", "parameter"),
        energy=float(arguments.get("energy", 1e8)),
        beta_x=float(arguments.get("beta_x", 1.0)),
        beta_y=float(arguments.get("beta_y", 1.0)),
        alpha_x=float(arguments.get("alpha_x", 0.0)),
        alpha_y=float(arguments.get("alpha_y", 0.0)),
        emittance_x=float(arguments.get("emittance_x", 1e-6)),
        emittance_y=float(arguments.get("emittance_y", 1e-6)),
        num_particles=int(arguments.get("num_particles", 1000)),
    )
    outgoing_beam = segment.track(beam)
    return outgoing_beam.summary(as_string=False)


def tool_match_beam(arguments: dict[str, Any]) -> dict[str, Any]:
    """Optimize quadrupole strengths to match target beam sizes."""
    segment = load_lattice(
        lattice_path=arguments.get("lattice_path"),
        lattice_dict=arguments.get("lattice_dict"),
    )
    target_sigma_x = float(arguments["target_sigma_x"])
    target_sigma_y = float(arguments["target_sigma_y"])
    quad_names = arguments["quad_names"]
    steps = int(arguments.get("steps", 50))
    lr = float(arguments.get("lr", 0.1))

    # Collect parameters to optimize
    params = []
    for name in quad_names:
        quad = getattr(segment, name)
        quad.k1 = torch.nn.Parameter(quad.k1.clone())
        params.append(quad.k1)

    optimizer = torch.optim.Adam(params, lr=lr)

    loss_val = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        beam = create_beam(
            beam_type="parameter",
            energy=float(arguments.get("energy", 1e8)),
            beta_x=float(arguments.get("beta_x", 1.0)),
            beta_y=float(arguments.get("beta_y", 1.0)),
            alpha_x=float(arguments.get("alpha_x", 0.0)),
            alpha_y=float(arguments.get("alpha_y", 0.0)),
            emittance_x=float(arguments.get("emittance_x", 1e-6)),
            emittance_y=float(arguments.get("emittance_y", 1e-6)),
        )
        out = segment.track(beam)
        loss = (out.sigma_x - target_sigma_x) ** 2 + (out.sigma_y - target_sigma_y) ** 2
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())

    final_k1 = {name: float(getattr(segment, name).k1.item()) for name in quad_names}
    eval_beam = create_beam(
        beam_type="parameter",
        energy=float(arguments.get("energy", 1e8)),
        beta_x=float(arguments.get("beta_x", 1.0)),
        beta_y=float(arguments.get("beta_y", 1.0)),
        alpha_x=float(arguments.get("alpha_x", 0.0)),
        alpha_y=float(arguments.get("alpha_y", 0.0)),
        emittance_x=float(arguments.get("emittance_x", 1e-6)),
        emittance_y=float(arguments.get("emittance_y", 1e-6)),
    )
    final_beam = segment.track(eval_beam)
    return {
        "optimized_parameters": final_k1,
        "loss": loss_val,
        "final_beam": final_beam.summary(as_string=False),
    }


TOOLS = [
    {
        "name": "cheetah_inspect_lattice",
        "description": (
            "Inspects lattice elements, cumulative positions, and properties of an "
            "accelerator beamline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lattice_path": {
                    "type": "string",
                    "description": "Path to a LatticeJSON file.",
                },
                "lattice_dict": {
                    "type": "object",
                    "description": "LatticeJSON dictionary definition.",
                },
            },
        },
    },
    {
        "name": "cheetah_track_beam",
        "description": (
            "Tracks a particle or parameter beam through a Cheetah accelerator "
            "lattice and returns beam properties."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lattice_path": {
                    "type": "string",
                    "description": "Path to LatticeJSON file.",
                },
                "lattice_dict": {
                    "type": "object",
                    "description": "LatticeJSON dictionary definition.",
                },
                "beam_type": {
                    "type": "string",
                    "enum": ["parameter", "particle"],
                    "default": "parameter",
                },
                "energy": {
                    "type": "number",
                    "default": 1e8,
                    "description": "Beam energy in eV.",
                },
                "beta_x": {"type": "number", "default": 1.0},
                "beta_y": {"type": "number", "default": 1.0},
                "alpha_x": {"type": "number", "default": 0.0},
                "alpha_y": {"type": "number", "default": 0.0},
                "emittance_x": {"type": "number", "default": 1e-6},
                "emittance_y": {"type": "number", "default": 1e-6},
                "num_particles": {"type": "integer", "default": 1000},
            },
        },
    },
    {
        "name": "cheetah_match_beam",
        "description": (
            "Optimizes quadrupole strengths via PyTorch automatic differentiation "
            "to match target beam sizes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lattice_path": {
                    "type": "string",
                    "description": "Path to LatticeJSON file.",
                },
                "lattice_dict": {
                    "type": "object",
                    "description": "LatticeJSON dictionary definition.",
                },
                "target_sigma_x": {
                    "type": "number",
                    "description": ("Target horizontal beam size sigma_x in meters."),
                },
                "target_sigma_y": {
                    "type": "number",
                    "description": ("Target vertical beam size sigma_y in meters."),
                },
                "quad_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of quadrupole element names to tune.",
                },
                "energy": {
                    "type": "number",
                    "default": 1e8,
                    "description": "Beam energy in eV.",
                },
                "steps": {"type": "integer", "default": 50},
                "lr": {"type": "number", "default": 0.1},
            },
            "required": ["target_sigma_x", "target_sigma_y", "quad_names"],
        },
    },
]

TOOL_HANDLERS = {
    "cheetah_inspect_lattice": tool_inspect_lattice,
    "cheetah_track_beam": tool_track_beam,
    "cheetah_match_beam": tool_match_beam,
}


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Process a JSON-RPC MCP request and return the JSON-RPC response."""
    req_id = request.get("id")
    method = request.get("method")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "cheetah-mcp", "version": cheetah.__version__},
                "capabilities": {"tools": {}},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Tool '{tool_name}' not found."},
            }
        try:
            result = handler(arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                },
            }

    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not supported."},
        }
    return None


def run_stdio_server() -> None:
    """Run the stdio MCP server loop."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
