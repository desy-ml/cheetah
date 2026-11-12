import torch


def lab_to_element_coordinates(
    x_offset: torch.Tensor,
    y_offset: torch.Tensor,
    tilt: torch.Tensor,
    x: torch.Tensor,
    px: torch.Tensor,
    y: torch.Tensor,
    py: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transforms particle coordinates from lab to element frame.

    :param x_offset: Element horizontal offset [m].
    :param y_offset: Element vertical offset [m].
    :param tilt: Tilt angle [rad].
    :param x: Horizontal coordinate in lab frame [m].
    :param px: Horizontal momentum in lab frame.
    :param y: Vertical coordinate in lab frame [m].
    :param py: Vertical momentum in lab frame.
    :return: Transformed coordinates (x, px, y, py) in element frame.
    """
    sin_tilt = tilt.sin().unsqueeze(-1)
    cos_tilt = tilt.cos().unsqueeze(-1)
    x_shifted = x - x_offset.unsqueeze(-1)
    y_shifted = y - y_offset.unsqueeze(-1)

    x_element = x_shifted * cos_tilt + y_shifted * sin_tilt
    y_element = -x_shifted * sin_tilt + y_shifted * cos_tilt
    px_element = px * cos_tilt + py * sin_tilt
    py_element = -px * sin_tilt + py * cos_tilt

    return x_element, px_element, y_element, py_element


def element_to_lab_coordinates(
    x_offset: torch.Tensor,
    y_offset: torch.Tensor,
    tilt: torch.Tensor,
    x: torch.Tensor,
    px: torch.Tensor,
    y: torch.Tensor,
    py: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transforms particle coordinates from element to lab frame.

    :param x_offset: Element horizontal offset [m].
    :param y_offset: Element vertical offset [m].
    :param tilt: Tilt angle [rad].
    :param x: Horizontal coordinate in element frame [m].
    :param px: Horizontal momentum in element frame.
    :param y: Vertical coordinate in element frame [m].
    :param py: Vertical momentum in element frame.
    :return: Transformed coordinates (x, px, y, py) in lab frame.
    """
    sin_tilt = tilt.sin().unsqueeze(-1)
    cos_tilt = tilt.cos().unsqueeze(-1)

    x_lab = x * cos_tilt - y * sin_tilt + x_offset.unsqueeze(-1)
    y_lab = x * sin_tilt + y * cos_tilt + y_offset.unsqueeze(-1)
    px_lab = px * cos_tilt - py * sin_tilt
    py_lab = px * sin_tilt + py * cos_tilt

    return x_lab, px_lab, y_lab, py_lab


def low_energy_z_correction(
    pz: torch.Tensor,
    p0c: torch.Tensor,
    mc2: torch.Tensor,
    ds: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the longitudinal coordinate correction for particles with speed < c.

    For small momentum deviations `pz`, a Taylor expansion is used to avoid
    loss of precision from catastrophic cancellation when computing `beta - beta0`.

    :param pz: Relative longitudinal momentum deviation delta = (p - p0) / p0.
    :param p0c: Reference particle momentum times speed of light [eV].
    :param mc2: Particle rest mass energy [eV].
    :param ds: Step length along the reference trajectory [m].
    :return: Longitudinal position change dz [m].
    """
    p0c_expanded = p0c.unsqueeze(-1)
    beta0 = p0c / (p0c.square() + mc2.square()).sqrt()
    e_tot = (p0c.square() + mc2.square()).sqrt()

    beta0_expanded = beta0.unsqueeze(-1)
    beta0_sq = beta0_expanded.square()
    e_tot_expanded = e_tot.unsqueeze(-1)
    mass_ratio_sq = (mc2 / e_tot_expanded).square()

    # Relativistic beta for each particle
    p_particle = (1.0 + pz) * p0c_expanded
    beta = p_particle / (p_particle.square() + mc2.square()).sqrt()

    # Threshold criterion for Taylor expansion regime
    criterion = mc2 * (beta0_expanded * pz).square()
    threshold = 3.0e-7 * e_tot_expanded

    dz_taylor = (
        pz
        * (
            1.0
            - 1.5 * pz * beta0_sq
            + pz.square() * beta0_sq * (2.0 * beta0_sq - 0.5 * mass_ratio_sq)
        )
        * mass_ratio_sq
    )
    dz_exact = (beta - beta0_expanded) / beta0_expanded

    return ds.unsqueeze(-1) * torch.where(criterion < threshold, dz_taylor, dz_exact)


def quadrupole_coefficients(
    k1: torch.Tensor,
    length: torch.Tensor,
    p_norm: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Returns 2x2 transfer matrix elements aij and the coefficients to compute the
    change in z position.

    NOTE: Accumulated error due to machine epsilon.

    :param k1: Quadrupole strength (k1 > 0 ==> defocus).
    :param length: Quadrupole length.
    :param p_norm: Normalized momentum P/P0.
    :return: Tuple of transfer matrix elements and coefficients.
        a11, a12, a21, a22: Transfer matrix elements.
        c1, c2, c3: Second order derivatives of z such that
            z = c1 * x_0^2 + c2 * x_0 * px_0 + c3 * px_0^2.
    """
    # The sign of k1 is flipped compared to track_methods by convention
    kx = (torch.complex(-k1, torch.zeros_like(k1))).sqrt()
    cx = (kx * length.unsqueeze(-1)).cos().real
    sx = ((kx * length.unsqueeze(-1) / torch.pi).sinc() * length.unsqueeze(-1)).real

    a11 = cx
    a12 = sx / p_norm
    a21 = k1 * sx * p_norm
    a22 = cx

    c1 = k1 * (-cx * sx + length.unsqueeze(-1)) / 4.0
    c2 = -k1 * sx.square() / (2.0 * p_norm)
    c3 = -(cx * sx + length.unsqueeze(-1)) / (4.0 * p_norm.square())

    return [[a11, a12], [a21, a22]], [c1, c2, c3]


def sqrt_one(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate `sqrt(1 + x) - 1` to machine precision.

    :param x: Input tensor.
    :return: Resulting tensor.
    """
    sq = (1.0 + x).sqrt()
    rad = sq + 1.0

    return x / rad


def track_a_drift(
    length: torch.Tensor,
    x_in: torch.Tensor,
    px_in: torch.Tensor,
    y_in: torch.Tensor,
    py_in: torch.Tensor,
    tau_in: torch.Tensor,
    delta_in: torch.Tensor,
    p0c: torch.Tensor,
    mc2: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact drift tracking in Cheetah coordinates.

    :param length: Drift length.
    :param x_in: Input x-coordinate.
    :param px_in: Input x-momentum.
    :param y_in: Input y-coordinate.
    :param py_in: Input y-momentum.
    :param tau_in: Input tau coordinate.
    :param delta_in: Input delta momentum.
    :param p0c: Reference particle momentum in eV.
    :param mc2: Particle mass in eV.
    :return: Output x, y, and tau coordinates.
    """
    e_tot = (p0c.square() + mc2.square()).sqrt()
    energy = e_tot.unsqueeze(-1) + delta_in * p0c.unsqueeze(-1)
    p = (energy.square() - mc2.square()).sqrt()
    beta = p / energy
    p_norm = p / p0c.unsqueeze(-1)
    pz = p_norm - 1.0

    Px = px_in / p_norm
    Py = py_in / p_norm
    Pxy2 = Px.square() + Py.square()
    Pl = (1.0 - Pxy2).sqrt()

    dz = length.unsqueeze(-1) * (
        sqrt_one(
            (mc2.square() * (2.0 * pz + pz.square()))
            / ((p0c.unsqueeze(-1) * p_norm).square() + mc2.square())
        )
        + sqrt_one(-Pxy2) / Pl
    )

    x_out = x_in + length.unsqueeze(-1) * Px / Pl
    y_out = y_in + length.unsqueeze(-1) * Py / Pl
    tau_out = tau_in - dz / beta

    return x_out, y_out, tau_out
