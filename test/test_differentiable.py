import torch
from torch import nn

import cheetah


def test_simple_quadrupole():
    """
    Simple test on a [D, Q, D] lattice with the qudrupole's k1 requiring grad, checking
    if PyTorch tracked a grad_fn into the outgoing beam.
    """
    segment = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Quadrupole(
                length=torch.tensor(0.2),
                k1=nn.Parameter(torch.tensor(3.142)),
                name="my_quad",
            ),
            cheetah.Drift(length=torch.tensor(1.0)),
        ]
    )
    incoming_beam = cheetah.ParticleBeam.from_astra(
        "benchmark/astra/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam = segment.track(incoming_beam)

    assert outgoing_beam.particles.grad_fn is not None
