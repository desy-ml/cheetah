import ARESlatticeStage3v1_9 as lattice

import cheetah

cell = cheetah.utils.subcell_of_ocelot(lattice.cell, "AREASOLA1", "ARMRBSCR1")
segment = cheetah.Segment.from_ocelot(cell)
segment.ARMRBSCR1.is_active = True

segment.AREAMQZM2.misalignment = (0.0000005, 0.0)

segment.AREAMQZM1.k1 = 4.5
segment.AREAMQZM2.k1 = -9.0
segment.AREAMQZM3.k1 = 4.5

segment.plot_overview(resolution=0.01)

def test_always_passes():
    assert True