import torch

import lynx


def ocelot2cheetah(
    element, warnings: bool = True, device=None, dtype=torch.float32
) -> "lynx.Element":
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

    if isinstance(element, ocelot.Drift):
        return lynx.Drift(
            length=torch.tensor([element.l], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Quadrupole):
        return lynx.Quadrupole(
            length=torch.tensor([element.l], dtype=torch.float32),
            k1=torch.tensor([element.k1], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Solenoid):
        return lynx.Solenoid(
            length=torch.tensor([element.l], dtype=torch.float32),
            k=torch.tensor([element.k], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Hcor):
        return lynx.HorizontalCorrector(
            length=torch.tensor([element.l], dtype=torch.float32),
            angle=torch.tensor([element.angle], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Vcor):
        return lynx.VerticalCorrector(
            length=torch.tensor([element.l], dtype=torch.float32),
            angle=torch.tensor([element.angle], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Bend):
        return lynx.Dipole(
            length=torch.tensor([element.l], dtype=torch.float32),
            angle=torch.tensor([element.angle], dtype=torch.float32),
            e1=torch.tensor([element.e1], dtype=torch.float32),
            e2=torch.tensor([element.e2], dtype=torch.float32),
            tilt=torch.tensor([element.tilt], dtype=torch.float32),
            fringe_integral=torch.tensor([element.fint], dtype=torch.float32),
            fringe_integral_exit=torch.tensor([element.fintx], dtype=torch.float32),
            gap=torch.tensor([element.gap], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.SBend):
        return lynx.Dipole(
            length=torch.tensor([element.l], dtype=torch.float32),
            angle=torch.tensor([element.angle], dtype=torch.float32),
            e1=torch.tensor([element.e1], dtype=torch.float32),
            e2=torch.tensor([element.e2], dtype=torch.float32),
            tilt=torch.tensor([element.tilt], dtype=torch.float32),
            fringe_integral=torch.tensor([element.fint], dtype=torch.float32),
            fringe_integral_exit=torch.tensor([element.fintx], dtype=torch.float32),
            gap=torch.tensor([element.gap], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.RBend):
        return lynx.RBend(
            length=torch.tensor([element.l], dtype=torch.float32),
            angle=torch.tensor([element.angle], dtype=torch.float32),
            e1=torch.tensor([element.e1], dtype=torch.float32) - element.angle / 2,
            e2=torch.tensor([element.e2], dtype=torch.float32) - element.angle / 2,
            tilt=torch.tensor([element.tilt], dtype=torch.float32),
            fringe_integral=torch.tensor([element.fint], dtype=torch.float32),
            fringe_integral_exit=torch.tensor([element.fintx], dtype=torch.float32),
            gap=torch.tensor([element.gap], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Cavity):
        return lynx.Cavity(
            length=torch.tensor([element.l], dtype=torch.float32),
            voltage=torch.tensor([element.v], dtype=torch.float32) * 1e9,
            frequency=torch.tensor([element.freq], dtype=torch.float32),
            phase=torch.tensor([element.phi], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.TDCavity):
        # TODO: Better replacement at some point?
        return lynx.Cavity(
            length=torch.tensor([element.l], dtype=torch.float32),
            voltage=torch.tensor([element.v], dtype=torch.float32) * 1e9,
            frequency=torch.tensor([element.freq], dtype=torch.float32),
            phase=torch.tensor([element.phi], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Monitor) and ("BSC" in element.id):
        # NOTE This pattern is very specific to ARES and will need a more complex
        # solution for other accelerators
        if warnings:
            print(
                "WARNING: Diagnostic screen was converted with default screen"
                " properties."
            )
        return lynx.Screen(
            resolution=torch.tensor([2448, 2040]),
            pixel_size=torch.tensor([3.5488e-6, 2.5003e-6]),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Monitor) and "BPM" in element.id:
        return lynx.BPM(name=element.id)
    elif isinstance(element, ocelot.Marker):
        return lynx.Marker(name=element.id)
    elif isinstance(element, ocelot.Monitor):
        return lynx.Marker(name=element.id)
    elif isinstance(element, ocelot.Undulator):
        return lynx.Undulator(
            torch.tensor(element.l, dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Aperture):
        shape_translation = {"rect": "rectangular", "elip": "elliptical"}
        return lynx.Aperture(
            x_max=torch.tensor([element.xmax], dtype=torch.float32),
            y_max=torch.tensor([element.ymax], dtype=torch.float32),
            shape=shape_translation[element.type],
            is_active=True,
            name=element.id,
            device=device,
            dtype=dtype,
        )
    else:
        if warnings:
            print(
                f"WARNING: Unknown element {element.id} of type {type(element)},"
                " replacing with drift section."
            )
        return lynx.Drift(
            length=torch.tensor([element.l], dtype=torch.float32),
            name=element.id,
            device=device,
            dtype=dtype,
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
