import ocelot as oc

from joss import accelerator


def ocelot2joss(element):
    """
    Translate an Ocelot element to a JOSS element.

    Parameters
    ----------
    element : ocelot.Element
        Ocelot element object representing an element of particle accelerator.
    
    Returns
    -------
    joss.Element
        JOSS element object representing an element of particle accelerator.
    
    Notes
    -----
    Object not supported by JOSS are translated to drift sections. Screen objects are created only
    from `ocelot.Monitor` objects when the string "SCR" in their `id` attribute.
    """
    if element.__class__ is oc.Drift:
        return accelerator.Drift(element.l, name=element.id)
    elif element.__class__ is oc.Quadrupole:
        return accelerator.Quadrupole(element.l, element.k1, name=element.id)
    elif element.__class__ is oc.Hcor:
        return accelerator.HorizontalCorrector(element.l, element.angle, name=element.id)
    elif element.__class__ is oc.Vcor:
        return accelerator.VerticalCorrector(element.l, element.angle, name=element.id)
    elif element.__class__ is oc.Cavity:
        return accelerator.Cavity(element.l, name=element.id)
    elif element.__class__ is oc.Monitor and "SCR" in element.id:
        print("WARNING: A diagnostic screen was converted from Ocelot and its resolution and pixel size were set to default values. You may need to adjust these!")
        return accelerator.Screen((2448,2040), (3.5488e-6,2.5003e-6), name=element.id)
    elif element.__class__ is oc.Monitor and "BPM" in element.id:
        return accelerator.BPM(name=element.id)
    elif element.__class__ is oc.Undulator:
        return accelerator.Undulator(element.l, name=element.id)
    else:
        return accelerator.Drift(element.l, name=element.id)
