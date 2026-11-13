"""Local stdio MCP tools for inspecting, tracking, and matching Cheetah models."""

from pathlib import Path
from typing import Annotated, Any, Literal, cast

import torch
from torch import nn

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
    if (lattice_path is None) == (lattice_dict is None):
        raise ToolError("Provide exactly one of lattice_path and lattice_dict.")
    if lattice_path is not None and not Path(lattice_path).is_file():
        raise ToolError("lattice_path must point to a regular LatticeJSON file.")
    try:
        if lattice_dict is not None:
            return latticejson.parse_segment(
                lattice_dict["root"], lattice_dict, sanitize_names=False
            )
        assert lattice_path is not None
        return cheetah.Segment.from_lattice_json(lattice_path, sanitize_names=False)
    except (
        AttributeError,
        KeyError,
        OSError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise ToolError(f"Could not load LatticeJSON: {exc}") from exc


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
    """
    Inspect a LatticeJSON model with nested segments expanded in tracking order.
    Provide exactly one lattice source.

    :param lattice_path: Path to a local LatticeJSON file.
    :param lattice_dict: LatticeJSON data supplied inline.
    :return: Summary of element names, positions in metres, and settings.
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
    end_element: str | None = None,
) -> dict[str, Any]:
    """
    Track an electron beam from entrance Twiss parameters. Provide exactly one
    lattice source.

    :param lattice_path: Path to a local LatticeJSON file.
    :param lattice_dict: LatticeJSON data supplied inline.
    :param beam_type: Track a `ParameterBeam` or a `ParticleBeam`.
    :param energy: Total reference energy in eV.
    :param beta_x: Horizontal beta function in metres.
    :param beta_y: Vertical beta function in metres.
    :param alpha_x: Dimensionless horizontal alpha function.
    :param alpha_y: Dimensionless vertical alpha function.
    :param emittance_x: Horizontal geometric emittance in metre-radians.
    :param emittance_y: Vertical geometric emittance in metre-radians.
    :param num_particles: Number of macroparticles for a `ParticleBeam`.
    :param end_element: Unique element name from the flattened lattice inspection.
        Track through that element, or through the full lattice if omitted.
    :return: Summary of the outgoing beam.
    """
    segment = _load_lattice(lattice_path, lattice_dict)
    if end_element is not None:
        segment = segment.flattened()
        if segment.element_names.count(end_element) != 1:
            raise ToolError(f"'{end_element}' must identify exactly one element.")
        segment = segment.subcell(end=end_element)
    twiss = _twiss(energy, beta_x, beta_y, alpha_x, alpha_y, emittance_x, emittance_y)
    incoming_beam: cheetah.Beam
    if beam_type == "particle":
        incoming_beam = cheetah.ParticleBeam.from_twiss(
            num_particles=num_particles, **twiss
        )
    else:
        incoming_beam = cheetah.ParameterBeam.from_twiss(**twiss)

    outgoing_beam = segment.track(incoming_beam)
    return cast(dict[str, Any], outgoing_beam.summary())


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
    """
    Tune scalar quadrupoles to match the exit RMS sizes of an electron beam.
    Provide exactly one lattice source.

    :param target_sigma_x: Target horizontal RMS beam size in metres.
    :param target_sigma_y: Target vertical RMS beam size in metres.
    :param quad_names: Names of quadrupoles to tune, each unique within the lattice.
    :param lattice_path: Path to a local LatticeJSON file.
    :param lattice_dict: LatticeJSON data supplied inline.
    :param energy: Total reference energy in eV.
    :param beta_x: Horizontal beta function in metres.
    :param beta_y: Vertical beta function in metres.
    :param alpha_x: Dimensionless horizontal alpha function.
    :param alpha_y: Dimensionless vertical alpha function.
    :param emittance_x: Horizontal geometric emittance in metre-radians.
    :param emittance_y: Vertical geometric emittance in metre-radians.
    :param steps: Number of optimiser steps.
    :param lr: Learning rate for the Adam optimiser.
    :return: Fitted `k1` values in 1/m², the outgoing beam summary, and the final
        squared error in m².
    """
    segment = _load_lattice(lattice_path, lattice_dict)
    elements = segment.flattened().elements
    quadrupole_strengths: dict[str, nn.Parameter] = {}
    for name in dict.fromkeys(quad_names):
        matches = [element for element in elements if element.name == name]
        if len(matches) != 1 or not isinstance(matches[0], cheetah.Quadrupole):
            raise ToolError(f"'{name}' must identify exactly one quadrupole.")
        quadrupole = matches[0]
        if quadrupole.k1.ndim != 0:
            raise ToolError(f"'{name}' must have a scalar k1.")
        quadrupole.k1 = nn.Parameter(quadrupole.k1)
        quadrupole_strengths[name] = quadrupole.k1

    incoming_beam = cheetah.ParameterBeam.from_twiss(
        **_twiss(energy, beta_x, beta_y, alpha_x, alpha_y, emittance_x, emittance_y)
    )
    target_scale = max(target_sigma_x, target_sigma_y)
    optimizer = torch.optim.Adam(list(quadrupole_strengths.values()), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        outgoing_beam = segment.track(incoming_beam)
        sigma_x, sigma_y = outgoing_beam.sigma_x, outgoing_beam.sigma_y
        if sigma_x.ndim != 0 or sigma_y.ndim != 0:
            raise ToolError("Matching requires a non-vectorised lattice.")
        loss = ((sigma_x - target_sigma_x) / target_scale).square() + (
            (sigma_y - target_sigma_y) / target_scale
        ).square()
        if not loss.isfinite() or not loss.requires_grad:
            raise ToolError(
                "Matching requires finite beam sizes and a differentiable lattice."
            )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_beam = segment.track(incoming_beam)
        final_loss = (final_beam.sigma_x - target_sigma_x).square() + (
            final_beam.sigma_y - target_sigma_y
        ).square()
        if not final_loss.isfinite():
            raise ToolError("Matching produced non-finite beam sizes.")
        return {
            "optimized_parameters": {
                name: strength.item() for name, strength in quadrupole_strengths.items()
            },
            "loss": final_loss.item(),
            "final_beam": final_beam.summary(),
        }


if __name__ == "__main__":
    server.run(transport="stdio")
