import numpy as np
import torch

from cheetah import accelerator as acc

# electron mass in eV
m_e_eV = 510998.8671


def from_astrabeam(path: str):
    """
    Read from a ASTRA beam distribution, and prepare for conversion to a Cheetah
    ParticleBeam or ParameterBeam.

    Adapted from the implementation in ocelot:
    https://github.com/ocelot-collab/ocelot/blob/master/ocelot/adaptors/astra2ocelot.py

    Parameters
    ----------
    path : str
        Path to the ASTRA beam distribution file.

    Returns
    -------
    particles : np.ndarray
        Particle 6D phase space information
    energy : float
        Mean energy of the particle beam

    """
    P0 = np.loadtxt(path)

    # remove lost particles
    inds = np.argwhere(P0[:, 9] > 0)
    inds = inds.reshape(inds.shape[0])

    P0 = P0[inds, :]
    n_particles = P0.shape[0]

    # s_ref = P0[0, 2]
    Pref = P0[0, 5]

    xp = P0[:, :6]
    xp[0, 2] = 0.0
    xp[0, 5] = 0.0

    gamref = np.sqrt((Pref / m_e_eV) ** 2 + 1)
    # energy in eV: E = gamma * m_e
    energy = gamref * m_e_eV

    n_particles = xp.shape[0]
    particles = np.zeros((n_particles, 6))

    u = np.c_[xp[:, 3], xp[:, 4], xp[:, 5] + Pref]
    gamma = np.sqrt(1 + np.sum(u * u, 1) / m_e_eV**2)
    beta = np.sqrt(1 - gamma**-2)
    betaref = np.sqrt(1 - gamref**-2)

    p0 = np.linalg.norm(u, 2, 1).reshape((n_particles, 1))

    u = u / p0
    cdt = -xp[:, 2] / (beta * u[:, 2])
    particles[:, 0] = xp[:, 0] + beta * u[:, 0] * cdt
    particles[:, 2] = xp[:, 1] + beta * u[:, 1] * cdt
    particles[:, 4] = cdt
    particles[:, 1] = xp[:, 3] / Pref
    particles[:, 3] = xp[:, 4] / Pref
    particles[:, 5] = (gamma / gamref - 1) / betaref

    return particles, energy


def ocelot2cheetah(element, warnings=True):
    """
    Translate an Ocelot element to a Cheetah element.

    Parameters
    ----------
    element : ocelot.Element
        Ocelot element object representing an element of particle accelerator.

    Returns
    -------
    cheetah.Element
        Cheetah element object representing an element of particle accelerator.

    Notes
    -----
    Object not supported by Cheetah are translated to drift sections. Screen objects are
    created only from `ocelot.Monitor` objects when the string "SCR" in their `id`
    attribute. Their screen properties are always set to default values and most likely
    need adjusting afterwards. BPM objects are only created from `ocelot.Monitor`
    objects when their id has a substring "BPM".
    """
    try:
        import ocelot as oc
    except ImportError:
        raise ImportError(
            """To use the ocelot2cheetah lattice converter, Ocelot must be first 
        installed, see https://github.com/ocelot-collab/ocelot """
        )

    if isinstance(element, oc.Drift):
        return acc.Drift(element.l, name=element.id)
    elif isinstance(element, oc.Quadrupole):
        return acc.Quadrupole(element.l, element.k1, name=element.id)
    elif isinstance(element, oc.Hcor):
        return acc.HorizontalCorrector(element.l, element.angle, name=element.id)
    elif isinstance(element, oc.Vcor):
        return acc.VerticalCorrector(element.l, element.angle, name=element.id)
    elif isinstance(element, oc.Cavity):
        return acc.Cavity(element.l, name=element.id)
    elif isinstance(element, oc.Monitor) and "BSC" in element.id:
        if warnings:
            print(
                "WARNING: Diagnostic screen was converted with default screen properties."
            )
        return acc.Screen((2448, 2040), (3.5488e-6, 2.5003e-6), name=element.id)
    elif isinstance(element, oc.Monitor) and "BPM" in element.id:
        return acc.BPM(name=element.id)
    elif isinstance(element, oc.Undulator):
        return acc.Undulator(element.l, name=element.id)
    else:
        if warnings:
            print(
                f"WARNING: Unknown element {element.id}, replacing with drift section."
            )
        return acc.Drift(element.l, name=element.id)


def subcell_of_ocelot(cell, start, end):
    """Extract a subcell `[start, end]` from an Ocelot cell."""
    subcell = []
    is_in_subcell = False
    for el in cell:
        if el.id == start:
            is_in_subcell = True
        if is_in_subcell:
            subcell.append(el)
        if el.id == end:
            break

    return subcell


_range = range


def histogramdd(sample, bins=None, range=None, weights=None, remove_overflow=True):
    """
    Pytorch version of n-dimensional histogram.

    Taken from https://github.com/miranov25/RootInteractive/blob/b54446e09072e90e17f3da72d5244a20c8fdd209/RootInteractive/Tools/Histograms/histogramdd.py
    """
    edges = None
    device = None
    custom_edges = False
    D, N = sample.shape
    if device == None:
        device = sample.device
    if bins == None:
        if edges == None:
            bins = 10
            custom_edges = False
        else:
            try:
                bins = edges.size(1) - 1
            except AttributeError:
                bins = torch.empty(D)
                for i in _range(len(edges)):
                    bins[i] = edges[i].size(0) - 1
                bins = bins.to(device)
            custom_edges = True
    try:
        M = bins.size(0)
        if M != D:
            raise ValueError(
                "The dimension of bins must be equal to the dimension of sample x."
            )
    except AttributeError:
        # bins is either an integer or a list
        if type(bins) == int:
            bins = torch.full([D], bins, dtype=torch.long, device=device)
        elif torch.is_tensor(bins[0]):
            custom_edges = True
            edges = bins
            bins = torch.empty(D, dtype=torch.long)
            for i in _range(len(edges)):
                bins[i] = edges[i].size(0) - 1
            bins = bins.to(device)
        else:
            bins = torch.as_tensor(bins)
    if bins.dim() == 2:
        custom_edges = True
        edges = bins
        bins = torch.full([D], bins.size(1) - 1, dtype=torch.long, device=device)
    if custom_edges:
        use_old_edges = False
        if not torch.is_tensor(edges):
            use_old_edges = True
            edges_old = edges
            m = max(i.size(0) for i in edges)
            tmp = torch.empty([D, m], device=edges[0].device)
            for i in _range(D):
                s = edges[i].size(0)
                tmp[i, :] = edges[i][-1]
                tmp[i, :s] = edges[i][:]
            edges = tmp.to(device)
        k = torch.searchsorted(edges, sample)
        k = torch.min(k, (bins + 1).reshape(-1, 1))
        if use_old_edges:
            edges = edges_old
        else:
            edges = torch.unbind(edges)
    else:
        if range == None:  # range is not defined
            range = torch.empty(2, D, device=device)
            if N == 0:  # Empty histogram
                range[0, :] = 0
                range[1, :] = 1
            else:
                range[0, :] = torch.min(sample, 1)[0]
                range[1, :] = torch.max(sample, 1)[0]
        elif not torch.is_tensor(range):  # range is a tuple
            r = torch.empty(2, D)
            for i in _range(D):
                if range[i] is not None:
                    r[:, i] = torch.as_tensor(range[i])
                else:
                    if N == 0:  # Edge case: empty histogram
                        r[0, i] = 0
                        r[1, i] = 1
                    r[0, i] = torch.min(sample[:, i])[0]
                    r[1, i] = torch.max(sample[:, i])[0]
            range = r.to(device=device, dtype=sample.dtype)
        singular_range = torch.eq(
            range[0], range[1]
        )  # If the range consists of only one point, pad it up.
        range[0, singular_range] -= 0.5
        range[1, singular_range] += 0.5
        edges = [
            torch.linspace(range[0, i], range[1, i], bins[i] + 1)
            for i in _range(len(bins))
        ]
        tranges = torch.empty_like(range)
        tranges[1, :] = bins / (range[1, :] - range[0, :])
        tranges[0, :] = 1 - range[0, :] * tranges[1, :]
        k = torch.addcmul(
            tranges[0, :].reshape(-1, 1), sample, tranges[1, :].reshape(-1, 1)
        ).long()  # Get the right index
        k = torch.max(
            k, torch.zeros([], device=device, dtype=torch.long)
        )  # Underflow bin
        k = torch.min(k, (bins + 1).reshape(-1, 1))

    multiindex = torch.ones_like(bins)
    multiindex[1:] = torch.cumprod(torch.flip(bins[1:], [0]) + 2, -1).long()
    multiindex = torch.flip(multiindex, [0])
    l = torch.sum(k * multiindex.reshape(-1, 1), 0)
    hist = torch.bincount(
        l, minlength=(multiindex[0] * (bins[0] + 2)).item(), weights=weights
    )
    hist = hist.reshape(tuple(bins + 2))
    if remove_overflow:
        core = D * (slice(1, -1),)
        hist = hist[core]
    return hist, edges
