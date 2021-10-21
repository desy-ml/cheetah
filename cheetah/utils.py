import ocelot as oc
import torch

from cheetah import accelerator as acc


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
    Object not supported by Cheetah are translated to drift sections. Screen objects are created only
    from `ocelot.Monitor` objects when the string "SCR" in their `id` attribute. Their screen
    properties are always set to default values and most likely need adjusting afterwards. BPM
    objects are only created from `ocelot.Monitor` objects when their id has a substring "BPM".
    """
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
    elif isinstance(element, oc.Monitor) and "SCR" in element.id:
        if warnings:
            print("WARNING: Diagnostic screen was converted with default screen properties.")
        return acc.Screen((2448,2040), (3.5488e-6,2.5003e-6), name=element.id)
    elif isinstance(element, oc.Monitor) and "BPM" in element.id:
        return acc.BPM(name=element.id)
    elif isinstance(element, oc.Undulator):
        return acc.Undulator(element.l, name=element.id)
    else:
        return acc.Drift(element.l, name=element.id)


def subcell_of(cell, start, end):
    """Extract a subcell `[start, end]` from an Ocelot cell."""
    subcell = []
    is_in_subcell = False
    for el in cell:
        if el.id == start: is_in_subcell = True
        if is_in_subcell: subcell.append(el)
        if el.id == end: break
    
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
            raise ValueError("The dimension of bins must be equal to the dimension of sample x.")
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
        bins = torch.full([D], bins.size(1)-1, dtype=torch.long, device=device)
    if custom_edges:
        use_old_edges = False
        if not torch.is_tensor(edges):
            use_old_edges = True
            edges_old = edges
            m = max(i.size(0) for i in edges)
            tmp = torch.empty([D,m], device=edges[0].device)
            for i in _range(D):
                s = edges[i].size(0)
                tmp[i,:] = edges[i][-1]
                tmp[i,:s] = edges[i][:]
            edges = tmp.to(device)
        k = torch.searchsorted(edges, sample)
        k = torch.min(k, (bins+1).reshape(-1,1))
        if use_old_edges:
            edges = edges_old
        else:
            edges = torch.unbind(edges)
    else:
            if range == None: # range is not defined
                range = torch.empty(2, D, device=device)
                if N == 0: # Empty histogram
                    range[0,:] = 0
                    range[1,:] = 1
                else:
                    range[0,:]=torch.min(sample, 1)[0]
                    range[1,:]=torch.max(sample, 1)[0]
            elif not torch.is_tensor(range): #range is a tuple
                r = torch.empty(2, D)
                for i in _range(D):
                    if range[i] is not None:
                        r[:,i] = torch.as_tensor(range[i])
                    else:
                        if N == 0: # Edge case: empty histogram
                            r[0,i] = 0
                            r[1,i] = 1
                        r[0,i] = torch.min(sample[:,i])[0]
                        r[1,i] = torch.max(sample[:,i])[0]
                range = r.to(device=device, dtype=sample.dtype)
            singular_range = torch.eq(range[0], range[1]) # If the range consists of only one point, pad it up.
            range[0,singular_range] -= 0.5
            range[1,singular_range] += 0.5
            edges = [torch.linspace(range[0,i], range[1,i], bins[i]+1) for i in _range(len(bins))]
            tranges = torch.empty_like(range)
            tranges[1,:] = bins / (range[1,:] - range[0,:])
            tranges[0,:] = 1 - range[0,:] * tranges[1,:]
            k = torch.addcmul(tranges[0,:].reshape(-1,1), sample, tranges[1,:].reshape(-1,1)).long() # Get the right index
            k = torch.max(k, torch.zeros([], device=device, dtype=torch.long)) # Underflow bin
            k = torch.min(k, (bins+1).reshape(-1,1))

    multiindex = torch.ones_like(bins)
    multiindex[1:] = torch.cumprod(torch.flip(bins[1:],[0])+2, -1).long()
    multiindex = torch.flip(multiindex, [0])
    l = torch.sum(k * multiindex.reshape(-1,1), 0)
    hist = torch.bincount(l, minlength=(multiindex[0]*(bins[0]+2)).item(), weights=weights)
    hist = hist.reshape(tuple(bins+2))
    if remove_overflow:
        core = D * (slice(1, -1),)
        hist = hist[core]
    return hist, edges
