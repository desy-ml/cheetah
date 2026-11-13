import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_mcp_stdio(tmp_path, mode):
    """Discover and use all three tools through a real stdio client."""
    lattice = {
        "root": "line",
        "lattices": {"line": ["d1", "cell", "d2"], "cell": ["q1"]},
        "elements": {
            "d1": ["Drift", {"length": 0.5}],
            "q1": ["Quadrupole", {"length": 0.2, "k1": 0.0}],
            "d2": ["Drift", {"length": 0.5}],
        },
    }
    path = tmp_path / "lattice.json"
    path.write_text(json.dumps(lattice), encoding="utf-8")

    async def exercise_tools():
        """Check discovery, tracking, matching, and error reporting."""
        env = {
            name: os.environ[name]
            for name in ("PYTHONPATH", "MPLCONFIGDIR")
            if name in os.environ
        }
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "cheetah.mcp"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
        )
        async with Client(server, mode=mode) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            assert tools == {
                "cheetah_inspect_lattice",
                "cheetah_track_beam",
                "cheetah_match_beam",
            }
            inspect = await client.call_tool(
                "cheetah_inspect_lattice", {"lattice_path": str(path)}
            )
            assert not inspect.is_error, inspect.content
            data = inspect.structured_content
            assert data["num_elements"] == 3
            assert data["total_length"] == pytest.approx(1.2)
            assert data["elements"][1]["name"] == "q1"

            beam_args = {"lattice_dict": lattice, "beta_x": 2.0, "alpha_y": -0.5}
            initial = await client.call_tool("cheetah_track_beam", beam_args)
            assert not initial.is_error, initial.content
            initial_beam = initial.structured_content
            assert initial_beam["s"] == pytest.approx(1.2)
            particle = await client.call_tool(
                "cheetah_track_beam",
                {**beam_args, "beam_type": "particle", "num_particles": 50},
            )
            assert not particle.is_error, particle.content
            assert particle.structured_content["num_particles"] == 50

            lattice["elements"]["q1"][1]["k1"] = 2.0
            target = await client.call_tool("cheetah_track_beam", beam_args)
            assert not target.is_error, target.content
            target_x = target.structured_content["sigma_x"]
            target_y = target.structured_content["sigma_y"]
            lattice["elements"]["q1"][1]["k1"] = 0.0

            match = await client.call_tool(
                "cheetah_match_beam",
                {
                    **beam_args,
                    "target_sigma_x": target_x,
                    "target_sigma_y": target_y,
                    "quad_names": ["q1"],
                },
            )
            assert not match.is_error, match.content
            result = match.structured_content
            lattice["elements"]["q1"][1]["k1"] = result["optimized_parameters"]["q1"]
            tracked = await client.call_tool("cheetah_track_beam", beam_args)
            assert not tracked.is_error, tracked.content
            final = tracked.structured_content
            assert result["final_beam"]["sigma_x"] == pytest.approx(final["sigma_x"])
            assert result["final_beam"]["sigma_y"] == pytest.approx(final["sigma_y"])
            expected_loss = (final["sigma_x"] - target_x) ** 2 + (
                final["sigma_y"] - target_y
            ) ** 2
            assert result["loss"] == pytest.approx(expected_loss, rel=1e-5, abs=1e-16)
            initial_loss = (initial_beam["sigma_x"] - target_x) ** 2 + (
                initial_beam["sigma_y"] - target_y
            ) ** 2
            assert result["loss"] < initial_loss / 100

            bad_source = await client.call_tool("cheetah_inspect_lattice", {})
            assert bad_source.is_error
            bad_beam = await client.call_tool(
                "cheetah_track_beam", {**beam_args, "beta_x": -1.0}
            )
            assert bad_beam.is_error
            lattice["elements"]["q1"][0] = "Species"
            bad_lattice = await client.call_tool(
                "cheetah_inspect_lattice", {"lattice_dict": lattice}
            )
            assert bad_lattice.is_error

    asyncio.run(asyncio.wait_for(exercise_tools(), timeout=45))
