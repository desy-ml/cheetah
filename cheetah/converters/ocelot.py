import torch

import cheetah


def convert_element_to_cheetah(
    element,
    warnings: bool = True,
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
    :param warnings: Whether to print warnings when elements might not be converted as
        expected.
    :return: Cheetah element object representing an element of particle accelerator.
    """
    try:
        import ocelot
    except ImportError:
        raise ImportError(
            """To use the ocelot2cheetah lattice converter, Ocelot must be first
        installed, see https://github.com/ocelot-collab/ocelot """
        )

    device = device or torch.get_default_device()
    dtype = dtype or torch.get_default_dtype()

    if isinstance(element, ocelot.Drift):
        return cheetah.Drift(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.Quadrupole):
        return cheetah.Quadrupole(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            k1=torch.tensor(element.k1, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.Solenoid):
        return cheetah.Solenoid(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            k=torch.tensor(element.k, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.Hcor):
        return cheetah.HorizontalCorrector(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            angle=torch.tensor(element.angle, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.Vcor):
        return cheetah.VerticalCorrector(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            angle=torch.tensor(element.angle, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.Bend):
        return cheetah.Dipole(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            angle=torch.tensor(element.angle, device=device, dtype=dtype),
            dipole_e1=torch.tensor(element.e1, device=device, dtype=dtype),
            dipole_e2=torch.tensor(element.e2, device=device, dtype=dtype),
            tilt=torch.tensor(element.tilt, device=device, dtype=dtype),
            fringe_integral=torch.tensor(element.fint, device=device, dtype=dtype),
            fringe_integral_exit=torch.tensor(
                element.fintx, device=device, dtype=dtype
            ),
            gap=torch.tensor(element.gap, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.SBend):
        return cheetah.Dipole(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            angle=torch.tensor(element.angle, device=device, dtype=dtype),
            dipole_e1=torch.tensor(element.e1, device=device, dtype=dtype),
            dipole_e2=torch.tensor(element.e2, device=device, dtype=dtype),
            tilt=torch.tensor(element.tilt, device=device, dtype=dtype),
            fringe_integral=torch.tensor(element.fint, device=device, dtype=dtype),
            fringe_integral_exit=torch.tensor(
                element.fintx, device=device, dtype=dtype
            ),
            gap=torch.tensor(element.gap, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.RBend):
        return cheetah.RBend(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            angle=torch.tensor(element.angle, device=device, dtype=dtype),
            rbend_e1=torch.tensor(element.e1, device=device, dtype=dtype)
            - element.angle / 2,
            rbend_e2=torch.tensor(element.e2, device=device, dtype=dtype)
            - element.angle / 2,
            tilt=torch.tensor(element.tilt, device=device, dtype=dtype),
            fringe_integral=torch.tensor(element.fint, device=device, dtype=dtype),
            fringe_integral_exit=torch.tensor(
                element.fintx, device=device, dtype=dtype
            ),
            gap=torch.tensor(element.gap, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.Cavity):
        return cheetah.Cavity(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            voltage=torch.tensor(element.v, device=device, dtype=dtype) * 1e9,
            frequency=torch.tensor(element.freq, device=device, dtype=dtype),
            phase=torch.tensor(element.phi, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.TDCavity):
        # TODO: Better replacement at some point?
        return cheetah.Cavity(
            length=torch.tensor(element.l, device=device, dtype=dtype),
            voltage=torch.tensor(element.v, device=device, dtype=dtype) * 1e9,
            frequency=torch.tensor(element.freq, device=device, dtype=dtype),
            phase=torch.tensor(element.phi, device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.Monitor) and ("BSC" in element.id):
        # NOTE This pattern is very specific to ARES and will need a more complex
        # solution for other accelerators
        if warnings:
            print(
                "WARNING: Diagnostic screen was converted with default screen"
                " properties."
            )
        return cheetah.Screen(
            resolution=(2448, 2040),
            pixel_size=torch.tensor([3.5488e-6, 2.5003e-6], device=device, dtype=dtype),
            name=element.id,
        )
    elif isinstance(element, ocelot.Monitor) and "BPM" in element.id:
        return cheetah.BPM(name=element.id)
    elif isinstance(element, ocelot.Marker):
        return cheetah.Marker(name=element.id)
    elif isinstance(element, ocelot.Monitor):
        return cheetah.Marker(name=element.id)
    elif isinstance(element, ocelot.Undulator):
        return cheetah.Undulator(
            torch.tensor(element.l, device=device, dtype=dtype), name=element.id
        )
    elif isinstance(element, ocelot.Aperture):
        shape_translation = {"rect": "rectangular", "elip": "elliptical"}
        return cheetah.Aperture(
            x_max=torch.tensor(element.xmax, device=device, dtype=dtype),
            y_max=torch.tensor(element.ymax, device=device, dtype=dtype),
            shape=shape_translation[element.type],
            is_active=True,
            name=element.id,
        )
    else:
        if warnings:
            print(
                f"WARNING: Unknown element {element.id} of type {type(element)},"
                " replacing with drift section."
            )
        return cheetah.Drift(
            length=torch.tensor(element.l, device=device, dtype=dtype), name=element.id
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
