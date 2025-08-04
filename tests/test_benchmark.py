import pytest

import cheetah

from .resources import ARESlatticeStage3v1_9 as ares


@pytest.mark.parametrize(
    ("beam_cls", "tracking_method"),
    [
        (cheetah.ParticleBeam, "linear"),
        (cheetah.ParticleBeam, "second_order"),
        (cheetah.ParticleBeam, "drift_kick_drift"),
        (cheetah.ParameterBeam, "linear"),
    ],
)
def test_benchmark_ares_lattice(benchmark, beam_cls, tracking_method):
    """Benchmark for tracking through the ARES lattice with fixed beam."""
    incoming = beam_cls.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    segment = cheetah.Segment.from_ocelot(ares.cell)
    segment.set_attrs_on_every_element(tracking_method=tracking_method, num_steps=5)

    benchmark(segment.track, incoming=incoming)
