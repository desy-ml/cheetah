import test.ARESlatticeStage3v1_9 as ares

from cheetah import ParameterBeam, ParticleBeam, Segment


def test_ares_ocelot_import():
    segment = Segment.from_ocelot(ares.cell)
    segment.plot_overview()


def test_astra_beam_import():
    parameter_beam = ParameterBeam.from_astra("benchmark/astra/ACHIP_EA1_2021.1351.001")
    particle_beam = ParticleBeam.from_astra("benchmark/astra/ACHIP_EA1_2021.1351.001")

    assert isinstance(parameter_beam, ParameterBeam)
    assert isinstance(particle_beam, ParticleBeam)
