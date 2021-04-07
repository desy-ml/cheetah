import ocelot as oc

from joss import accelerator


def ocelot2joss(element):
    if element.__class__ is oc.Drift:
        return accelerator.Drift(element.l, name=element.id)
    elif element.__class__ is oc.Quadrupole:
        return accelerator.Quadrupole(element.l, element.k1, name=element.id)
    elif element.__class__ is oc.Hcor:
        return accelerator.HorizontalCorrector(element.l, element.angle, name=element.id)
    elif element.__class__ is oc.Vcor:
        return accelerator.VerticalCorrector(element.l, element.angle, name=element.id)
    elif element.__class__ is oc.Monitor and "SCR" in element.id:
        return accelerator.Screen(name=element.id)
    else:
        return accelerator.Drift(element.l, name=element.id)
