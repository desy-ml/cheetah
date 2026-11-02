import json

import pytest

from cheetah.mcp import handle_request


@pytest.fixture
def sample_lattice_dict():
    """Return a minimal valid LatticeJSON dictionary."""
    return {
        "root": "line",
        "lattices": {"line": ["d1", "q1", "d2"]},
        "elements": {
            "d1": ["Drift", {"length": 0.5}],
            "q1": ["Quadrupole", {"length": 0.2, "k1": 5.0}],
            "d2": ["Drift", {"length": 0.5}],
        },
    }


def test_mcp_initialize():
    """Test MCP initialize handshake."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    }
    response = handle_request(request)
    assert response is not None
    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "cheetah-mcp"


def test_mcp_tools_list():
    """Test MCP tools/list returns registered tools."""
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    response = handle_request(request)
    assert response is not None
    tools = response["result"]["tools"]
    tool_names = [tool["name"] for tool in tools]
    assert "cheetah_inspect_lattice" in tool_names
    assert "cheetah_track_beam" in tool_names
    assert "cheetah_match_beam" in tool_names


def test_mcp_inspect_lattice(sample_lattice_dict):
    """Test cheetah_inspect_lattice tool call."""
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "cheetah_inspect_lattice",
            "arguments": {"lattice_dict": sample_lattice_dict},
        },
    }
    response = handle_request(request)
    assert response is not None
    content = response["result"]["content"]
    assert len(content) > 0
    data = json.loads(content[0]["text"])
    assert data["num_elements"] == 3
    assert pytest.approx(data["total_length"]) == 1.2


def test_mcp_track_beam(sample_lattice_dict):
    """Test cheetah_track_beam tool call."""
    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "cheetah_track_beam",
            "arguments": {
                "lattice_dict": sample_lattice_dict,
                "beam_type": "parameter",
                "energy": 1e8,
                "beta_x": 2.0,
                "beta_y": 2.0,
            },
        },
    }
    response = handle_request(request)
    assert response is not None
    content = response["result"]["content"]
    data = json.loads(content[0]["text"])
    assert "sigma_x" in data
    assert "sigma_y" in data


def test_mcp_match_beam(sample_lattice_dict):
    """Test cheetah_match_beam optimization tool call."""
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "cheetah_match_beam",
            "arguments": {
                "lattice_dict": sample_lattice_dict,
                "target_sigma_x": 0.0001,
                "target_sigma_y": 0.0001,
                "quad_names": ["q1"],
                "steps": 5,
                "lr": 0.05,
            },
        },
    }
    response = handle_request(request)
    assert response is not None
    content = response["result"]["content"]
    data = json.loads(content[0]["text"])
    assert "optimized_parameters" in data
    assert "q1" in data["optimized_parameters"]
