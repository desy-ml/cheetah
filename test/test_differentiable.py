import torch

import cheetah


def test_simple_quadrupole():
    """
    Simple test on a [D, Q, D] lattice with the qudrupole's k1 requiring grad, checking
    if PyTorch tracked a grad_fn into the outgoing beam.
    """
    segment = cheetah.Segment(
        [
            cheetah.Drift(torch.tensor(1.0)),
            cheetah.Quadrupole(torch.tensor(0.2), k1=torch.tensor(1.0), name="my_quad"),
            cheetah.Drift(1.0),
        ]
    )
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    segment.my_quad.k1.requires_grad = True

    outgoing_beam = segment.track(incoming_beam)

    assert hasattr(outgoing_beam, "grad_fn")
