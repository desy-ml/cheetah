import numpy as np
import pytest
import torch

import cheetah
from cheetah.accelerator.wakefields import (
    SPEED_OF_LIGHT,
    gaussian_resonant_wake_potential,
    reference_momentum,
)


def test_assert_ei_greater_zero():
    """
    Reproduces

    ```
       1127 Ef = (energy + delta_energy) / particle_mass_eV
       1128 Ep = (Ef - Ei) / self.length  # Derivative of the energy
    -> 1129 assert Ei > 0, "Initial energy must be larger than 0"
       1131 alpha = torch.sqrt(eta / 8) / torch.cos(phi) * torch.log(Ef / Ei)
       1133 r11 = torch.cos(alpha) - torch.sqrt(2 / eta) * torch.cos(phi) * torch.sin(alpha)   # noqa: E501

    RuntimeError: Boolean value of Tensor with more than one value is ambiguous
    ```
    """
    cavity = cheetah.Cavity(
        length=torch.tensor([3.0441, 3.0441, 3.0441]),
        voltage=torch.tensor([48198468.0, 48198468.0, 48198468.0]),
        phase=torch.tensor([48198468.0, 48198468.0, 48198468.0]),
        frequency=torch.tensor([2.8560e09, 2.8560e09, 2.8560e09]),
        name="k26_2a",
    )
    beam = cheetah.ParticleBeam.from_parameters(
        num_particles=100_000, sigma_x=torch.tensor(1e-5)
    )

    _ = cavity.track(beam)


@pytest.mark.parametrize(
    ("voltage", "phase"),
    [
        (torch.tensor(0.0), torch.tensor([-90.0, 90.0])),
        (torch.tensor([0.0, 1e6]), torch.tensor([[-90.0], [0.0], [90.0], [180.0]])),
        (torch.tensor(1e6), torch.tensor([0.0, 180.0])),
    ],
    ids=["off", "mixed", "on"],
)
@pytest.mark.parametrize("cavity_type", ["standing_wave", "traveling_wave"])
def test_vectorized_inactive_cavity(cavity_type, voltage, phase):
    """
    Tests that a vectorised cavity with zero voltage or off-crest phase does not produce
    NaNs and that switched-off cavities can be vectorized with switched-on.

    This was a bug introduced during the vectorisation of Cheetah, when the special
    case of zero was removed and the `_cavity_rmatrix` method was also used in the case
    of zero voltage or off-crest phase. The latter produced NaNs in the transfer matrix.
    """
    cavity = cheetah.Cavity(
        cavity_type=cavity_type,
        length=torch.tensor(3.0441),
        voltage=voltage,
        phase=phase,
        frequency=torch.tensor(2.8560e09),
    ).to(torch.float64)
    incoming = cheetah.ParameterBeam.from_parameters(
        sigma_x=torch.tensor(4.8492e-06),
        sigma_px=torch.tensor(1.5603e-07),
        sigma_y=torch.tensor(4.1209e-07),
        sigma_py=torch.tensor(1.1035e-08),
        sigma_tau=torch.tensor(1.0000e-10),
        sigma_p=torch.tensor(1.0000e-06),
        energy=torch.tensor(8.0000e09),
        total_charge=torch.tensor(0.0),
    ).to(torch.float64)

    outgoing = cavity.track(incoming)

    assert (
        not cavity.first_order_transfer_map(incoming.energy, incoming.species)
        .isnan()
        .any()
    )

    assert not outgoing.sigma_x.isnan().any()
    assert not outgoing.sigma_y.isnan().any()
    assert not outgoing.beta_x.isnan().any()
    assert not outgoing.beta_y.isnan().any()

# ---------------------------------------------------------------------------
# Cavity with wake=True: three checks
#
# 1. slice statistics vs the closed-form Gaussian wake potential
# 2. RF chirp with the wake off, vs the analytic V k_rf / p0c
# 3. RF and wake together: beam loading, spread, and nothing else moving
#
# The kernel-level tests live in test_wakefields.py.
# ---------------------------------------------------------------------------

DTYPE = torch.float64
FREQUENCY = 1.3e9
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY
REFERENCE_ENERGY = 100e6
CAVITY_LENGTH = 1.038
CAVITY_VOLTAGE = 16e6 * CAVITY_LENGTH
TOTAL_CHARGE = 1e-9

# The notebook's amplitude.  It was applied to a beam whose particle_charges were
# ones -- i.e. 1e4 coulombs -- so the absolute numbers there are not physical;
# only the shape was.  Here the same amplitude acts on a real 1 nC bunch.
WAKE_STRENGTH = 35_000.0


def make_beam(sigma_tau, *, total_charge=TOTAL_CHARGE, num_particles=200_000, seed=0):
    """Gaussian `ParticleBeam` like the one used in `newWake.ipynb`."""
    torch.manual_seed(seed)
    return cheetah.ParticleBeam.from_twiss(
        num_particles=num_particles,
        beta_x=torch.tensor(10.0, dtype=DTYPE),
        beta_y=torch.tensor(10.0, dtype=DTYPE),
        emittance_x=torch.tensor(2e-9, dtype=DTYPE),
        emittance_y=torch.tensor(2e-9, dtype=DTYPE),
        sigma_tau=torch.tensor(sigma_tau, dtype=DTYPE),
        sigma_p=torch.tensor(1e-3, dtype=DTYPE),
        energy=torch.tensor(REFERENCE_ENERGY, dtype=DTYPE),
        total_charge=torch.tensor(total_charge, dtype=DTYPE),
        dtype=DTYPE,
    )


def cavity(**kwargs):
    """`Cavity` with the test's RF settings, plus any wake keyword arguments."""
    return cheetah.Cavity(
        length=torch.tensor(CAVITY_LENGTH, dtype=DTYPE),
        voltage=torch.tensor(CAVITY_VOLTAGE, dtype=DTYPE),
        phase=torch.tensor(kwargs.pop("phase", 0.0), dtype=DTYPE),
        frequency=torch.tensor(FREQUENCY, dtype=DTYPE),
        **kwargs,
    )


def slice_means(tau, values, n_bins=100, min_count=50):
    """Mean of `values` in each `tau` bin with at least `min_count` particles."""
    edges = np.linspace(tau.min(), tau.max(), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    index = np.clip(np.digitize(tau, edges) - 1, 0, n_bins - 1)

    counts = np.bincount(index, minlength=n_bins)
    totals = np.bincount(index, weights=values, minlength=n_bins)

    populated = counts >= min_count
    return centres[populated], totals[populated] / counts[populated]


# ---------------------------------------------------------------- 1. slice stats


def test_slice_statistics_match_the_analytical_gaussian_wake():
    """Slice-averaged wake kick matches the closed-form Gaussian wake potential."""
    beam = make_beam(3e-2)
    element = cheetah.Cavity(
        length=torch.tensor(0.0, dtype=DTYPE),
        frequency=torch.tensor(FREQUENCY, dtype=DTYPE),
        wake=True,
        wake_strength=WAKE_STRENGTH,
        dtype=DTYPE,
    )
    outgoing = element.track(beam)

    p0c = float(reference_momentum(beam.energy, beam.species.mass_eV))
    tau = beam.tau.detach().numpy()
    delta_energy = (outgoing.p - beam.p).detach().numpy() * p0c

    centres, measured = slice_means(tau, delta_energy)

    # The closed form wants the bunch's actual sigma, not the requested one.
    sigma_tau = float(beam.tau.std())
    analytic = -gaussian_resonant_wake_potential(
        centres,
        sigma_tau=sigma_tau,
        wavelength=WAVELENGTH,
        total_charge=float(beam.particle_charges.sum()),
        wake_strength=WAKE_STRENGTH,
    )

    scale = np.sqrt(np.mean(analytic**2))
    relative_rms = np.sqrt(np.mean((measured - analytic) ** 2)) / scale
    assert relative_rms < 0.02, f"slice means vs closed form: {relative_rms:.2%}"

    # The wake must actually oscillate over the bunch, otherwise the agreement
    # above is trivially satisfied by a flat line through zero.
    k_rf = 2.0 * np.pi / WAVELENGTH
    assert k_rf * sigma_tau > 0.5
    assert measured.min() < 0.0 < measured.max()

    # And the FFT grid the element actually convolved on must agree too.
    diagnostics = element.cached_wake_diagnostics
    on_grid = -np.interp(
        centres,
        diagnostics["z_grid"].detach().numpy(),
        diagnostics["W_grid"].detach().numpy(),
    )
    assert np.sqrt(np.mean((on_grid - analytic) ** 2)) / scale < 0.05


# ------------------------------------------------------------------- 2. RF chirp


def test_rf_chirp_slope_matches_v_k_over_p0c():
    """With the wake off at zero crossing, the RF chirp slope is `V k / p0c`."""
    beam = make_beam(3e-4)
    outgoing = cavity(phase=90.0, wake=False).track(beam)

    tau = beam.tau.detach().numpy()
    delta = (outgoing.p - beam.p).detach().numpy()
    slope = float(np.polyfit(tau, delta, 1)[0])

    p0c = float(reference_momentum(beam.energy, beam.species.mass_eV))
    k_rf = 2.0 * np.pi * FREQUENCY / SPEED_OF_LIGHT
    expected = CAVITY_VOLTAGE * k_rf / p0c

    relative_error = abs(abs(slope) - expected) / expected
    assert relative_error < 0.01, (
        f"measured chirp {slope:.4f} /m vs theory {expected:.4f} /m "
        f"({relative_error:.2%})"
    )

    # A chirp is a correlation, not a mean kick: on the zero crossing the bunch
    # centroid gains nothing to first order.
    assert abs(float(outgoing.p.mean())) < 0.1 * float(outgoing.p.std())

    # The relation must be linear -- a bad transfer map would still fit a slope.
    residual = delta - np.polyval(np.polyfit(tau, delta, 1), tau)
    assert np.sqrt(np.mean(residual**2)) < 1e-3 * float(np.abs(delta).max())


# ----------------------------------------------------------------- 3. RF + wake


def test_rf_plus_wake_loads_the_beam_and_adds_spread():
    """RF plus wake loads the beam and adds energy spread."""
    beam = make_beam(3e-4)
    rf_only = cavity(wake=False).track(beam)
    rf_wake = cavity(wake=True, wake_strength=35e12).track(beam)

    # The RF accelerates: the reference energy goes up.
    assert float(rf_only.energy) > float(beam.energy)

    # The wake takes energy back out -- beam loading.  This is
    # the sign that was previously wrong.
    assert float(rf_wake.p.mean()) < float(rf_only.p.mean())

    # ... and it adds energy spread, since head and tail are loaded differently.
    # Comparing the two total spreads directly is a near-useless assertion: the
    # intrinsic sigma_p of 1e-3 swamps the wake, which moves the total by 0.1%.
    # Isolate the wake's own contribution instead.
    induced = rf_wake.p - rf_only.p
    assert float(induced.std()) > 0.02 * float(rf_only.p.std())
    assert float(rf_wake.p.std()) > float(rf_only.p.std())

    # A wake is a thin longitudinal kick: nothing else may move.
    assert torch.allclose(rf_wake.tau, rf_only.tau)
    assert torch.allclose(rf_wake.x, rf_only.x)
    assert torch.allclose(rf_wake.px, rf_only.px)
    assert torch.allclose(rf_wake.y, rf_only.y)
    assert torch.allclose(rf_wake.py, rf_only.py)
    assert float(rf_wake.energy) == float(rf_only.energy)
