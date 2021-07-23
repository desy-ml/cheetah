import ocelot as oc

from cheetah import accelerator as acc


def ocelot2cheetah(element):
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
        print("WARNING: Diagnostic screen was converted with default screen properties.")
        return acc.Screen((2448,2040), (3.5488e-6,2.5003e-6), name=element.id)
    elif isinstance(element, oc.Monitor) and "BPM" in element.id:
        return acc.BPM(name=element.id)
    elif isinstance(element, oc.Undulator):
        return acc.Undulator(element.l, name=element.id)
    else:
        return acc.Drift(element.l, name=element.id)
