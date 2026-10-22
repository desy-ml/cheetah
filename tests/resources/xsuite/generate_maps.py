"""Generate independent second-order body maps with Xsuite (not needed by pytest)."""

from pathlib import Path

import numpy as np
from scipy.constants import physical_constants

ENERGY = 1e9
# length [m], k1 [m^-2], k2 [m^-3], curvature [m^-1]
CASES = {
    "combined_short": (0.7, 0.3, 0.8, 0.5),
    "combined_long": (1.3, 0.3, 0.8, 0.5),
    "defocusing": (1.3, -0.8, 0.8, 0.5),
    "zero_horizontal_focusing": (1.3, -0.25, 0.8, 0.5),
    "resonance": (1.3, -0.2, 0.8, 1.0),
    "near_resonance": (1.3, -0.196, 0.8, 1.0),
}


def generate_map(length, k1, k2, hx, kicks):
    import xtrack as xt
    from xtrack.tpsa import ParticlesTpsa

    mass = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
    particles = ParticlesTpsa(order=2, mass0=mass, energy0=ENERGY)
    beta = particles.beta0
    # Seed Cheetah's (x, px, y, py, tau, p): zeta = -beta0*tau,
    # delta = sqrt(1 + 2*p/beta0 + p**2) - 1, through second order.
    particles.set_jacobian(np.diag([1.0, 1.0, 1.0, 1.0, -beta, 1.0 / beta]))
    particles.set_coefficient("delta", (0, 0, 0, 0, 0, 2), 0.5 * (1.0 - beta**-2))
    bend = xt.Bend(
        length=length,
        angle=hx * length,
        k1=k1,
        k2=k2,
        model="bend-kick-bend",
        integrator="yoshida-6",
        num_multipole_kicks=kicks,
        edge_entry_active=False,
        edge_exit_active=False,
    )
    bend.track(particles)
    outputs = [
        particles.x,
        particles.px,
        particles.y,
        particles.py,
        -particles.zeta / beta,
        beta * particles.delta + 0.5 * beta * (1.0 - beta**2) * particles.delta**2,
    ]
    tensor = np.zeros((7, 7, 7))
    for j in range(6):
        for k in range(j, 6):
            powers = [0] * 6
            powers[j] += 1
            powers[k] += 1
            # Polynomial coefficients already combine both mixed-term permutations.
            tensor[:6, j, k] = [output.coefficient(powers) for output in outputs]
    return tensor


if __name__ == "__main__":
    maps = {}
    for name, parameters in CASES.items():
        coarse = generate_map(*parameters, kicks=256)
        fine = generate_map(*parameters, kicks=512)
        # More kicks alone do not guarantee convergence: check that doubling the
        # integration resolution barely changes the map before saving it as a reference.
        np.testing.assert_allclose(coarse, fine, rtol=0.0, atol=1e-10)
        maps[name] = fine
        print(f"{name}: max integration difference {np.max(np.abs(coarse - fine)):.2e}")
    np.savez(Path(__file__).with_name("second_order_maps.npz"), **maps)
