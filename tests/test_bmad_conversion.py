import torch

import cheetah


def test_bmad_tutorial():
    """Test importing the lattice example file from the Bmad and Tao tutorial."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"
    converted = cheetah.Segment.from_bmad(file_path)
    converted.name = "bmad_tutorial"

    correct = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(0.5), name="d"),
            cheetah.Dipole(
                length=torch.tensor(0.5), e1=torch.tensor(0.1), name="b"
            ),  # TODO: What are g and dg?
            cheetah.Quadrupole(
                length=torch.tensor(0.6), k1=torch.tensor(0.23), name="q"
            ),
        ],
        name="bmad_tutorial",
    )

    assert converted.name == correct.name
    assert [element.name for element in converted.elements] == [
        element.name for element in correct.elements
    ]
    assert converted.d.length == correct.d.length
    assert converted.b.length == correct.b.length
    assert converted.b.e1 == correct.b.e1
    assert converted.q.length == correct.q.length
    assert converted.q.k1 == correct.q.k1


def test_device_passing():
    """Test that the device is passed correctly."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"

    # Check if CUDA or MPS are available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Convert the lattice while passing the device
    converted = cheetah.Segment.from_bmad(file_path, device=device)

    # Check that the properties of the loaded elements are on the correct device
    assert converted.d.length.device == torch.device("cpu")
    assert converted.b.length.device == torch.device("cpu")
    assert converted.b.e1.device == torch.device("cpu")
    assert converted.q.length.device == torch.device("cpu")
    assert converted.q.k1.device == torch.device("cpu")
