"""Local stdio MCP tools for inspecting, tracking, and matching Cheetah models."""

from pathlib import Path
from typing import Annotated, Any, Literal, cast

import torch

import cheetah
from cheetah import latticejson
from cheetah.particles.species import electron_mass_eV

try:
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from pydantic import Field
except ImportError as exc:
    raise ImportError(
        'Install MCP with pip install "cheetah-accelerator[mcp]".'
    ) from exc

server = MCPServer("cheetah-mcp", version=cheetah.__version__)

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
ElectronEnergy = Annotated[float, Field(gt=electron_mass_eV, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


def _load_lattice(
    lattice_path: str | None, lattice_dict: dict[str, Any] | None
) -> cheetah.Segment:
    """Load a segment from exactly one LatticeJSON source."""
    if lattice_path is not None and lattice_dict is not None:
        raise ToolError("Provide exactly one of lattice_path and lattice_dict.")
    try:
        if lattice_path is not None:
            if not Path(lattice_path).is_file():
                raise ToolError(
                    "lattice_path must point to a regular LatticeJSON file."
                )
            return cheetah.Segment.from_lattice_json(lattice_path, sanitize_names=False)
        if lattice_dict is not None:
            return latticejson.parse_segment(
                lattice_dict["root"], lattice_dict, sanitize_names=False
            )
    except (
        AttributeError,
        KeyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise ToolError(f"Could not load LatticeJSON: {exc}") from exc
    raise ToolError("Provide exactly one of lattice_path and lattice_dict.")


def _twiss(
    energy: float,
    beta_x: float,
    beta_y: float,
    alpha_x: float,
    alpha_y: float,
    emittance_x: float,
    emittance_y: float,
) -> dict[str, torch.Tensor]:
    """Convert scalar beam inputs to tensors for the beam constructors."""
    return {
        "energy": torch.tensor(energy),
        "beta_x": torch.tensor(beta_x),
        "beta_y": torch.tensor(beta_y),
        "alpha_x": torch.tensor(alpha_x),
        "alpha_y": torch.tensor(alpha_y),
        "emittance_x": torch.tensor(emittance_x),
        "emittance_y": torch.tensor(emittance_y),
    }


@server.tool()
def cheetah_inspect_lattice(
    lattice_path: str | None = None,
    lattice_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect element names, positions in metres, and settings in a LatticeJSON model.

    Provide exactly one of lattice_path (a local file) and lattice_dict (the inline
    LatticeJSON object). Nested segments are expanded in tracking order.
    """
    segment = _load_lattice(lattice_path, lattice_dict).flattened()
    return cast(dict[str, Any], segment.summary())


@server.tool()
@torch.no_grad()
def cheetah_track_beam(
    lattice_path: str | None = None,
    lattice_dict: dict[str, Any] | None = None,
    beam_type: Literal["parameter", "particle"] = "parameter",
    energy: ElectronEnergy = 1e8,
    beta_x: PositiveFloat = 1.0,
    beta_y: PositiveFloat = 1.0,
    alpha_x: FiniteFloat = 0.0,
    alpha_y: FiniteFloat = 0.0,
    emittance_x: PositiveFloat = 1e-6,
    emittance_y: PositiveFloat = 1e-6,
    num_particles: Annotated[int, Field(ge=7)] = 1000,
) -> dict[str, Any]:
    """Track an electron beam from entrance Twiss parameters and report exit optics.

    Provide exactly one LatticeJSON source, a local lattice_path or inline lattice_dict.
    Energy is total reference energy in eV, beta functions are in metres, alpha
    functions are dimensionless, and geometric emittances are in metre-radians.
    num_particles applies to particle beams.
    """
    segment = _load_lattice(lattice_path, lattice_dict)
    twiss = _twiss(energy, beta_x, beta_y, alpha_x, alpha_y, emittance_x, emittance_y)
    beam = (
        cheetah.ParticleBeam.from_twiss(num_particles=num_particles, **twiss)
        if beam_type == "particle"
        else cheetah.ParameterBeam.from_twiss(**twiss)
    )
    return cast(dict[str, Any], segment.track(beam).summary())


@server.tool()
def cheetah_match_beam(
    target_sigma_x: PositiveFloat,
    target_sigma_y: PositiveFloat,
    quad_names: Annotated[list[str], Field(min_length=1)],
    lattice_path: str | None = None,
    lattice_dict: dict[str, Any] | None = None,
    energy: ElectronEnergy = 1e8,
    beta_x: PositiveFloat = 1.0,
    beta_y: PositiveFloat = 1.0,
    alpha_x: FiniteFloat = 0.0,
    alpha_y: FiniteFloat = 0.0,
    emittance_x: PositiveFloat = 1e-6,
    emittance_y: PositiveFloat = 1e-6,
    steps: Annotated[int, Field(gt=0)] = 50,
    lr: PositiveFloat = 0.1,
) -> dict[str, Any]:
    """Tune uniquely named scalar quadrupoles to approach exit RMS sizes in metres.

    Provide exactly one LatticeJSON source, a local lattice_path or inline lattice_dict.
    The incoming electron beam uses the same energy and Twiss units as tracking.
    Return the fitted k1 values in 1/m², final beam summary, and squared error in m².
    """
    segment = _load_lattice(lattice_path, lattice_dict)
    elements = segment.flattened().elements
    params = {}
    for name in dict.fromkeys(quad_names):
        matches = [element for element in elements if element.name == name]
        if len(matches) != 1 or not isinstance(matches[0], cheetah.Quadrupole):
            raise ToolError(f"'{name}' must identify exactly one quadrupole.")
        quad = matches[0]
        if quad.k1.ndim != 0:
            raise ToolError(f"'{name}' must have a scalar k1.")
        quad.k1 = torch.nn.Parameter(quad.k1)
        params[name] = quad.k1

    beam = cheetah.ParameterBeam.from_twiss(
        **_twiss(energy, beta_x, beta_y, alpha_x, alpha_y, emittance_x, emittance_y)
    )
    target_scale = max(target_sigma_x, target_sigma_y)
    optimizer = torch.optim.Adam(list(params.values()), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        out = segment.track(beam)
        if out.sigma_x.ndim != 0 or out.sigma_y.ndim != 0:
            raise ToolError("Matching requires an unbatched lattice.")
        loss = ((out.sigma_x - target_sigma_x) / target_scale).square() + (
            (out.sigma_y - target_sigma_y) / target_scale
        ).square()
        if not loss.isfinite() or not loss.requires_grad:
            raise ToolError(
                "Matching requires finite beam sizes and a differentiable lattice."
            )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_beam = segment.track(beam)
        final_loss = (final_beam.sigma_x - target_sigma_x).square() + (
            final_beam.sigma_y - target_sigma_y
        ).square()
        if not final_loss.isfinite():
            raise ToolError("Matching produced non-finite beam sizes.")
        return {
            "optimized_parameters": {
                name: param.item() for name, param in params.items()
            },
            "loss": final_loss.item(),
            "final_beam": final_beam.summary(),
        }


if __name__ == "__main__":
    server.run(transport="stdio")
