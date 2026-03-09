from cheetah.accelerator import SuperimposedElement, BPM, Quadrupole, Drift, Segment
from cheetah.latticejson import convert_segment, parse_segment
from cheetah.particles import ParticleBeam
import torch

def test_superimposed_bpm():
    """
    Test that a superimposed BPM element correctly tracks particles through the segment.
    """

    # Create a base drift element
    quad = Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(1.0), name="Quad")

    # Create a BPM element to be superimposed
    bpm = BPM(name="BPM1", is_active=False)

    # Create a superimposed segment
    superimposed_segment = SuperimposedElement(
        base_element=quad,
        superimposed_element=bpm,
        name="SuperimposedBPM"
    )

    # make sure the elements are as expected
    assert isinstance(superimposed_segment.subelements[0], Quadrupole)
    assert superimposed_segment.subelements[0].name == "Quad#0"
    assert superimposed_segment.subelements[0].length == quad.length / 2
    assert isinstance(superimposed_segment.subelements[1], BPM)
    assert superimposed_segment.subelements[1].name == "BPM1"
    assert isinstance(superimposed_segment.subelements[2], Quadrupole)
    assert superimposed_segment.subelements[2].name == "Quad#1"
    assert superimposed_segment.subelements[2].length == quad.length / 2

    # Create an incoming particle beam
    incoming_beam = ParticleBeam.from_twiss(
        beta_x=torch.tensor(10.0),
        alpha_x=torch.tensor(0.0),
        beta_y=torch.tensor(10.0),
        alpha_y=torch.tensor(0.0),
    )

    # Track the beam through the superimposed segment
    outgoing_beam = superimposed_segment.track(incoming_beam)

    # Check that the outgoing beam has the same number of particles as the incoming beam
    assert outgoing_beam.particles.shape[0] == incoming_beam.particles.shape[0]

    # check the names of the elements in the superimposed segment
    assert superimposed_segment.subelements[0].k1 == quad.k1

    # check to make sure setting the strength of the quadrupole in the superimposed segment works
    superimposed_segment.base_element.k1 = torch.tensor(2.0)
    assert superimposed_segment.subelements[0].k1 == torch.tensor(2.0)
    assert superimposed_segment.subelements[2].k1 == torch.tensor(2.0)
    superimposed_segment.base_element.k1 = torch.tensor(1.0)

    # check the transfer map
    energy = torch.tensor(1.0e9)
    species = incoming_beam.species
    tm = superimposed_segment.first_order_transfer_map(energy, species)
    tm_expected = quad.first_order_transfer_map(energy, species)
    assert torch.allclose(tm, tm_expected)

    # set the quadrupole strength through the superimposed segment and ensure it propagates
    superimposed_segment.base_element.k1 = torch.tensor(3.0)
    quad.k1 = torch.tensor(3.0)

    tm = superimposed_segment.first_order_transfer_map(energy, species)
    tm_expected = quad.first_order_transfer_map(energy, species)
    assert torch.allclose(tm, tm_expected)

    # set the BPM to active and ensure tracking still works
    superimposed_segment.superimposed_element.is_active = True
    superimposed_segment.track(incoming_beam)
    assert torch.allclose(
        superimposed_segment.superimposed_element.reading,
        torch.zeros(2)
    )


def test_in_lattice():
    drift = Drift(length=torch.tensor(1.0), name="Drift")
    quad = Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(1.0), name="Quad")
    bpm = BPM(name="BPM1", is_active=False)

    superimposed_segment = SuperimposedElement(
        base_element=quad,
        superimposed_element=bpm,
        name="SuperimposedBPM"
    )
    full_segment = Segment([
        drift,
        superimposed_segment,
        drift
    ])
    assert full_segment.element_names == ["Drift", "SuperimposedBPM", "Drift"]

    # Create an incoming particle beam
    incoming_beam = ParticleBeam.from_twiss(
        beta_x=torch.tensor(10.0),
        alpha_x=torch.tensor(0.0),
        beta_y=torch.tensor(10.0),
        alpha_y=torch.tensor(0.0),
    )
    # Track the beam through the full segment
    full_segment.track(incoming_beam)

    # test flattening
    flattened = full_segment.flattened()
    assert flattened.element_names == ["Drift", "Quad#0", "BPM1", "Quad#1", "Drift"]


def test_to_json():
    """
    Test that a superimposed segment can be correctly serialized to and from JSON.
    """
    drift = Drift(length=torch.tensor(1.0), name="Drift")
    quad = Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(1.0), name="Quad")
    bpm = BPM(name="BPM1", is_active=False)

    superimposed_element = SuperimposedElement(
        base_element=quad,
        superimposed_element=bpm,
        name="SuperimposedBPM"
    )
    full_segment = Segment([
        drift,
        superimposed_element,
        drift
    ], name="FullSegment")

    # test conversion to dict
    elements, lattices = convert_segment(full_segment)
    segment_dict = {
        "elements": elements,
        "lattices": lattices
    }
    
    # test conversion back to segment
    reconstructed_segment = parse_segment(
        "FullSegment",
        segment_dict
    )
    assert reconstructed_segment.SuperimposedBPM.base_element.k1 == superimposed_element.base_element.k1

    full_segment.to_lattice_json("full_segment.json")





