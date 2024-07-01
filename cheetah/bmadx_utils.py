import torch


def cheetah_to_bmad_coords(
    cheetah_coords: torch.Tensor, ref_energy: torch.Tensor, mc2: torch.Tensor
) -> torch.Tensor:
    """
    Transforms Cheetah coordinates to Bmad coordinates.
    :param cheetah_coords: 7-dimensional particle vectors in Cheetah coordinates.
    :param ref_energy: reference energy in eV.
    """
    # initialize Bmad coordinates:
    bmad_coords = cheetah_coords[..., :6].clone()

    # Cheetah longitudinal coords:
    tau = cheetah_coords[..., 4]
    delta = cheetah_coords[..., 5]

    # compute p0c and bmad z, pz
    p0c = torch.sqrt(ref_energy**2 - mc2**2)
    energy = ref_energy + delta * p0c
    p = torch.sqrt(energy**2 - mc2**2)
    beta = p / energy
    z = -beta * tau
    pz = (p - p0c) / p0c

    # Bmad coords:
    bmad_coords[..., 4] = z
    bmad_coords[..., 5] = pz

    return bmad_coords, p0c


def bmad_to_cheetah_coords(
    bmad_coords: torch.Tensor, p0c: torch.Tensor, mc2: torch.Tensor
) -> torch.Tensor:
    """
    Transforms Bmad coordinates to Cheetah coordinates.
    :param bmad_coords: 6-dimensional particle vectors in Bmad coordinates.
    :param p0c: reference momentum in eV/c.
    """
    # initialize Cheetah coordinates:
    cheetah_coords = torch.ones(
        (*bmad_coords.shape[:-1], 7), dtype=bmad_coords.dtype, device=bmad_coords.device
    )
    cheetah_coords[..., :6] = bmad_coords.clone()

    # Bmad longitudinal coords:
    z = bmad_coords[..., 4]
    pz = bmad_coords[..., 5]

    # compute ref_energy and Cheetah tau, delta
    ref_energy = torch.sqrt(p0c**2 + mc2**2)
    p = (1 + pz) * p0c
    energy = torch.sqrt(p**2 + mc2**2)
    beta = p / energy
    tau = -z / beta
    delta = (energy - ref_energy) / p0c

    # Cheetah coords:
    cheetah_coords[..., 4] = tau
    cheetah_coords[..., 5] = delta

    return cheetah_coords, ref_energy


def offset_particle_set(
    x_offset: torch.Tensor,
    y_offset: torch.Tensor,
    tilt: torch.Tensor,
    x_lab: torch.Tensor,
    px_lab: torch.Tensor,
    y_lab: torch.Tensor,
    py_lab: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Transforms particle coordinates from lab to element frame.
    :param x_offset: element x-coordinate offset.
    :param y_offset: element y-coordinate offset.
    :param tilt: tilt angle (rad).
    :param x_lab: x-coordinate in lab frame.
    :param px_lab: x-momentum in lab frame.
    :param y_lab: y-coordinate in lab frame.
    :param py_lab: y-momentum in lab frame.
    :return: x, px, y, py coordinates in element frame.
    """
    s = torch.sin(tilt)
    c = torch.cos(tilt)
    x_ele_int = x_lab - x_offset
    y_ele_int = y_lab - y_offset
    x_ele = x_ele_int * c + y_ele_int * s
    y_ele = -x_ele_int * s + y_ele_int * c
    px_ele = px_lab * c + py_lab * s
    py_ele = -px_lab * s + py_lab * c

    return x_ele, px_ele, y_ele, py_ele


def offset_particle_unset(
    x_offset: torch.Tensor,
    y_offset: torch.Tensor,
    tilt: torch.Tensor,
    x_ele: torch.Tensor,
    px_ele: torch.Tensor,
    y_ele: torch.Tensor,
    py_ele: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Transforms particle coordinates from element to lab frame.
    :param x_offset: element x-coordinate offset.
    :param y_offset: element y-coordinate offset.
    :param tilt: tilt angle (rad).
    :param x_ele: x-coordinate in element frame.
    :param px_ele: x-momentum in element frame.
    :param y_ele: y-coordinate in element frame.
    :param py_ele: y-momentum in element frame.
    :return: x, px, y, py coordinates in lab frame.
    """
    s = torch.sin(tilt)
    c = torch.cos(tilt)
    x_lab_int = x_ele * c - y_ele * s
    y_lab_int = x_ele * s + y_ele * c
    x_lab = x_lab_int + x_offset
    y_lab = y_lab_int + y_offset
    px_lab = px_ele * c - py_ele * s
    py_lab = px_ele * s + py_ele * c

    return x_lab, px_lab, y_lab, py_lab


def low_energy_z_correction(
    pz: torch.Tensor, p0c: torch.Tensor, mc2: torch.Tensor, ds: torch.Tensor
) -> torch.Tensor:
    """Corrects the change in z-coordinate due to speed < c_light.
    Input:
        p0c -- reference particle momentum in eV
        mc2 -- particle mass in eV
        ds -- drift length
    Output:
        dz -- dz=(ds-d_particle) + ds*(beta - beta_ref)/beta_ref
    """
    beta = (1 + pz) * p0c / torch.sqrt(((1 + pz) * p0c) ** 2 + mc2**2)
    beta0 = p0c / torch.sqrt(p0c**2 + mc2**2)
    e_tot = torch.sqrt(p0c**2 + mc2**2)

    evaluation = mc2 * (beta0 * pz) ** 2
    dz = ds * pz * (
        1
        - 3 * (pz * beta0**2) / 2
        + pz**2 * beta0**2 * (2 * beta0**2 - (mc2 / e_tot) ** 2 / 2)
    ) * (mc2 / e_tot) ** 2 * (evaluation < 3e-7 * e_tot) + (
        ds * (beta - beta0) / beta0
    ) * (
        evaluation >= 3e-7 * e_tot
    )

    return dz


def calculate_quadrupole_coefficients(
    k1: torch.Tensor, length: torch.Tensor, rel_p: torch.Tensor
) -> torch.Tensor:
    """Returns 2x2 transfer matrix elements aij and the
    coefficients to calculate the change in z position.
    Input:
        k1_ref -- Quad strength: k1 > 0 ==> defocus
        length -- Quad length
        rel_p -- Relative momentum P/P0
    Output:
        a11, a12, a21, a22 -- transfer matrix elements
        c1, c2, c3 -- second order derivatives of z such that
                    z = c1 * x_0^2 + c2 * x_0 * px_0 + c3* px_0^2
    **NOTE**: accumulated error due to machine epsilon. REVISIT
    """
    eps = 2.220446049250313e-16  # machine epsilon to double precission

    sqrt_k = torch.sqrt(torch.absolute(k1) + eps)
    sk_l = sqrt_k * length

    cx = torch.cos(sk_l) * (k1 <= 0) + torch.cosh(sk_l) * (k1 > 0)
    sx = (torch.sin(sk_l) / (sqrt_k)) * (k1 <= 0) + (torch.sinh(sk_l) / (sqrt_k)) * (
        k1 > 0
    )

    a11 = cx
    a12 = sx / rel_p
    a21 = k1 * sx * rel_p
    a22 = cx

    c1 = k1 * (-cx * sx + length) / 4
    c2 = -k1 * sx**2 / (2 * rel_p)
    c3 = -(cx * sx + length) / (4 * rel_p**2)

    return [[a11, a12], [a21, a22]], [c1, c2, c3]
