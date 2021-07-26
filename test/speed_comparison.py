import time

from accelerator_environments import utils
import cheetah
import joss
import numpy as np
import ocelot as oc
import torch

import ARESlatticeStage3v1_9 as ares


def time_ocelot():
    measurements = []

    for _ in range(n):
        cell = utils.subcell_of(ares.cell, "AREASOLA1", "AREABSCR1")
        ares.areamqzm1.k1 = 5.5
        ares.areamqzm2.k1 = -5.5
        ares.areamqzm3.k1 = 0.0

        lattice = oc.MagneticLattice(cell)

        particles = oc.generate_parray(nparticles=int(1e+5), sigma_x=175e-6, sigma_y=175e-6)

        t1 = time.time()

        navigator = oc.Navigator(lattice)
        navigator.unit_step = lattice.totalLen
        _, particles = oc.track(lattice, particles, navigator, print_progress=False)

        t2 = time.time()

        measurements.append(t2 - t1)
    
    return sum(measurements) / n


def time_ocelot_screen():
    measurements = []

    for _ in range(n):
        binning = 4
        screen_resolution = (int(2448 / binning), int(2040 / binning))
        meter_per_pixel = (3.5488e-6 * binning, 2.5003e-6 * binning)
        screen_bin_edges = (np.linspace(-screen_resolution[0] * meter_per_pixel[0] / 2,
                                        screen_resolution[0] * meter_per_pixel[0] / 2,
                                        screen_resolution[0] + 1),
                            np.linspace(-screen_resolution[1] * meter_per_pixel[1] / 2,
                                        screen_resolution[1] * meter_per_pixel[1] / 2,
                                        screen_resolution[1] + 1))

        cell = utils.subcell_of(ares.cell, "AREASOLA1", "AREABSCR1")
        ares.areamqzm1.k1 = 5.5
        ares.areamqzm2.k1 = -5.5
        ares.areamqzm3.k1 = 0.0

        lattice = oc.MagneticLattice(cell)

        particles = oc.generate_parray(nparticles=int(1e+5), sigma_x=175e-6, sigma_y=175e-6)

        t1 = time.time()

        navigator = oc.Navigator(lattice)
        navigator.unit_step = lattice.totalLen
        _, particles = oc.track(lattice, particles, navigator, print_progress=False)
        image = np.histogram2d(particles.x(), particles.y(), bins=screen_bin_edges)[0].transpose()

        t2 = time.time()

        measurements.append(t2 - t1)
    
    return sum(measurements) / n


def time_joss():
    measurements = []

    for _ in range(n):
        cell = utils.subcell_of(ares.cell, "AREASOLA1", "AREABSCR1")

        segment = joss.Segment.from_ocelot(cell)
        segment.AREABSCR1.is_active = False  # Turn screen on and off

        particles = joss.Beam.make_random(n=int(1e+5), sigma_x=175e-6, sigma_y=175e-6)

        t1 = time.time()

        _ = segment(particles)

        t2 = time.time()

        measurements.append(t2 - t1)
    
    return sum(measurements) / n


def time_joss_screen():
    measurements = []

    for _ in range(n):
        cell = utils.subcell_of(ares.cell, "AREASOLA1", "AREABSCR1")

        segment = joss.Segment.from_ocelot(cell)
        segment.AREABSCR1.is_active = True  # Turn screen on and off

        particles = joss.Beam.make_random(n=int(1e+5), sigma_x=175e-6, sigma_y=175e-6)

        t1 = time.time()

        _ = segment(particles)
        image = segment.AREABSCR1.reading

        t2 = time.time()

        measurements.append(t2 - t1)
    
    return sum(measurements) / n


def time_cheetah_cpu():
    measurements = []

    for _ in range(n):
        cell = utils.subcell_of(ares.cell, "AREASOLA1", "AREABSCR1")

        segment = cheetah.Segment.from_ocelot(cell, device="cpu")
        segment.AREABSCR1.is_active = False  # Turn screen on and off

        particles = cheetah.Beam.make_random(n=int(1e+5), sigma_x=175e-6, sigma_y=175e-6, device="cpu")

        t1 = time.time()

        _ = segment(particles)

        t2 = time.time()

        measurements.append(t2 - t1)
    
    return sum(measurements) / n


def time_cheetah_cpu_screen():
    measurements = []

    for _ in range(n):
        cell = utils.subcell_of(ares.cell, "AREASOLA1", "AREABSCR1")

        segment = cheetah.Segment.from_ocelot(cell, device="cpu")
        segment.AREABSCR1.is_active = True  # Turn screen on and off

        particles = cheetah.Beam.make_random(n=int(1e+5), sigma_x=175e-6, sigma_y=175e-6, device="cpu")

        t1 = time.time()

        _ = segment(particles)
        image = segment.AREABSCR1.reading

        t2 = time.time()

        measurements.append(t2 - t1)
    
    return sum(measurements) / n


def time_cheetah_gpu():
    measurements = []

    for _ in range(n):
        cell = utils.subcell_of(ares.cell, "AREASOLA1", "AREABSCR1")

        segment = cheetah.Segment.from_ocelot(cell, device="cuda")
        segment.AREABSCR1.is_active = False  # Turn screen on and off

        particles = cheetah.Beam.make_random(n=int(1e+5), sigma_x=175e-6, sigma_y=175e-6, device="cuda")

        t1 = time.time()

        _ = segment(particles)

        t2 = time.time()

        measurements.append(t2 - t1)
    
    return sum(measurements) / n


def time_cheetah_gpu_screen():
    measurements = []

    for _ in range(n):
        cell = utils.subcell_of(ares.cell, "AREASOLA1", "AREABSCR1")

        segment = cheetah.Segment.from_ocelot(cell, device="cuda")
        segment.AREABSCR1.is_active = True  # Turn screen on and off

        particles = cheetah.Beam.make_random(n=int(1e+5), sigma_x=175e-6, sigma_y=175e-6, device="cuda")

        t1 = time.time()

        _ = segment(particles)
        image = segment.AREABSCR1.reading

        t2 = time.time()

        measurements.append(t2 - t1)
    
    return sum(measurements) / n


n = 3

toc = time_ocelot()
tjo = time_joss()
tcc = time_cheetah_cpu()
tcg = time_cheetah_gpu() if torch.cuda.is_available() else None

toc_scr = time_ocelot_screen()
tjo_scr = time_joss_screen()
tcc_scr = time_cheetah_cpu_screen()
tcg_scr = time_cheetah_gpu_screen() if torch.cuda.is_available() else None

print("")
print(f"            SCREEN OFF             ")
print(f"Simulation Code | Avrg. Time of {n}")
print(f"----------------------------------")
print(f"Ocelot          |   {toc:11.4f} s")
print(f"JOSS            |   {tjo:11.4f} s")
print(f"Cheetah (CPU)   |   {tcc:11.4f} s")
print(f"Cheetah (GPU)   |   {tcg:11.4f} s" if tcg is not None else "Cheetah (GPU)   |             N/A")

print("")
print(f"            SCREEN ON              ")
print(f"Simulation Code | Avrg. Time of {n}")
print(f"----------------------------------")
print(f"Ocelot          |   {toc_scr:11.4f} s")
print(f"JOSS            |   {tjo_scr:11.4f} s")
print(f"Cheetah (CPU)   |   {tcc_scr:11.4f} s")
print(f"Cheetah (GPU)   |   {tcg_scr:11.4f} s" if tcg_scr is not None else "Cheetah (GPU)   |             N/A")
