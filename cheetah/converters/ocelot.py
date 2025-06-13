import warnings

import torch

import cheetah
from cheetah.utils import DefaultParameterWarning, UnknownElementWarning


def convert_element_to_cheetah(
    element,
    sanitize_name: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """
    Translate an Ocelot element to a Cheetah element.

    NOTE Object not supported by Cheetah are translated to drift sections. Screen
    objects are created only from `ocelot.Monitor` objects when the string "BSC" is
    contained in their `id` attribute. Their screen properties are always set to default
    values and most likely need adjusting afterwards. BPM objects are only created from
    `ocelot.Monitor` objects when their id has a substring "BPM".

    :param element: Ocelot element object representing an element of particle
        accelerator.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable "
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    :return: Cheetah element object representing an element of particle accelerator.
    """
    try:
        import ocelot
    except ImportError:
        raise ImportError(
            """To use the ocelot2cheetah lattice converter, Ocelot must be first
        installed, see https://github.com/ocelot-collab/ocelot """
        )

    factory_kwargs = {
        "device": device or torch.get_default_device(),
        "dtype": dtype or torch.get_default_dtype(),
    }

    if isinstance(element, ocelot.Drift):
        return cheetah.Drift(
            length=torch.tensor(element.l, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Quadrupole):
        return cheetah.Quadrupole(
            length=torch.tensor(element.l, **factory_kwargs),
            k1=torch.tensor(element.k1, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Sextupole):
        return cheetah.Sextupole(
            length=torch.tensor(element.l, **factory_kwargs),
            k2=torch.tensor(element.k2, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Solenoid):
        return cheetah.Solenoid(
            length=torch.tensor(element.l, **factory_kwargs),
            k=torch.tensor(element.k, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Hcor):
        return cheetah.HorizontalCorrector(
            length=torch.tensor(element.l, **factory_kwargs),
            angle=torch.tensor(element.angle, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Vcor):
        return cheetah.VerticalCorrector(
            length=torch.tensor(element.l, **factory_kwargs),
            angle=torch.tensor(element.angle, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Bend):
        return cheetah.Dipole(
            length=torch.tensor(element.l, **factory_kwargs),
            angle=torch.tensor(element.angle, **factory_kwargs),
            dipole_e1=torch.tensor(element.e1, **factory_kwargs),
            dipole_e2=torch.tensor(element.e2, **factory_kwargs),
            tilt=torch.tensor(element.tilt, **factory_kwargs),
            fringe_integral=torch.tensor(element.fint, **factory_kwargs),
            fringe_integral_exit=torch.tensor(element.fintx, **factory_kwargs),
            gap=torch.tensor(element.gap, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.SBend):
        return cheetah.Dipole(
            length=torch.tensor(element.l, **factory_kwargs),
            angle=torch.tensor(element.angle, **factory_kwargs),
            dipole_e1=torch.tensor(element.e1, **factory_kwargs),
            dipole_e2=torch.tensor(element.e2, **factory_kwargs),
            tilt=torch.tensor(element.tilt, **factory_kwargs),
            fringe_integral=torch.tensor(element.fint, **factory_kwargs),
            fringe_integral_exit=torch.tensor(element.fintx, **factory_kwargs),
            gap=torch.tensor(element.gap, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.RBend):
        return cheetah.RBend(
            length=torch.tensor(element.l, **factory_kwargs),
            angle=torch.tensor(element.angle, **factory_kwargs),
            rbend_e1=torch.tensor(element.e1, **factory_kwargs) - element.angle / 2,
            rbend_e2=torch.tensor(element.e2, **factory_kwargs) - element.angle / 2,
            tilt=torch.tensor(element.tilt, **factory_kwargs),
            fringe_integral=torch.tensor(element.fint, **factory_kwargs),
            fringe_integral_exit=torch.tensor(element.fintx, **factory_kwargs),
            gap=torch.tensor(element.gap, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Cavity):
        return cheetah.Cavity(
            length=torch.tensor(element.l, **factory_kwargs),
            voltage=torch.tensor(element.v, **factory_kwargs) * 1e9,
            frequency=torch.tensor(element.freq, **factory_kwargs),
            phase=torch.tensor(element.phi, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.TDCavity):
        # TODO: Better replacement at some point?
        return cheetah.Cavity(
            length=torch.tensor(element.l, **factory_kwargs),
            voltage=torch.tensor(element.v, **factory_kwargs) * 1e9,
            frequency=torch.tensor(element.freq, **factory_kwargs),
            phase=torch.tensor(element.phi, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Monitor) and ("BSC" in element.id):
        # NOTE This pattern is very specific to ARES and will need a more complex
        # solution for other accelerators
        warnings.warn(
            "Diagnostic screen was converted with default screen properties.",
            category=DefaultParameterWarning,
            stacklevel=2,
        )
        return cheetah.Screen(
            resolution=(2448, 2040),
            pixel_size=torch.tensor([3.5488e-6, 2.5003e-6], **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Monitor) and "BPM" in element.id:
        return cheetah.BPM(name=element.id, sanitize_name=sanitize_name)
    elif isinstance(element, ocelot.Marker):
        return cheetah.Marker(name=element.id, sanitize_name=sanitize_name)
    elif isinstance(element, ocelot.Monitor):
        return cheetah.Marker(name=element.id, sanitize_name=sanitize_name)
    elif isinstance(element, ocelot.Undulator):
        return cheetah.Undulator(
            torch.tensor(element.l, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )
    elif isinstance(element, ocelot.Aperture):
        shape_translation = {"rect": "rectangular", "elip": "elliptical"}
        return cheetah.Aperture(
            x_max=torch.tensor(element.xmax, **factory_kwargs),
            y_max=torch.tensor(element.ymax, **factory_kwargs),
            shape=shape_translation[element.type],
            is_active=True,
            name=element.id,
            sanitize_name=sanitize_name,
        )
    else:
        warnings.warn(
            f"Unknown element {element.id} of type {type(element)}, replacing with "
            "drift section.",
            category=UnknownElementWarning,
            stacklevel=2,
        )
        return cheetah.Drift(
            length=torch.tensor(element.l, **factory_kwargs),
            name=element.id,
            sanitize_name=sanitize_name,
        )


def subcell_of_ocelot(cell: list, start: str, end: str) -> list:
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
