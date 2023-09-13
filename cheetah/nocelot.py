import torch

import cheetah


def ocelot2cheetah(element, warnings: bool = True) -> "cheetah.Element":
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
        return cheetah.Drift(
            length=torch.tensor(element.l, dtype=torch.float32), name=element.id
        )
    elif isinstance(element, ocelot.Quadrupole):
        return cheetah.Quadrupole(
            length=torch.tensor(element.l, dtype=torch.float32),
            k1=torch.tensor(element.k1, dtype=torch.float32),
            name=element.id,
        )
    elif isinstance(element, ocelot.Hcor):
        return cheetah.HorizontalCorrector(
            length=torch.tensor(element.l, dtype=torch.float32),
            angle=torch.tensor(element.angle, dtype=torch.float32),
            name=element.id,
        )
    elif isinstance(element, ocelot.Vcor):
        return cheetah.VerticalCorrector(
            length=torch.tensor(element.l, dtype=torch.float32),
            angle=torch.tensor(element.angle, dtype=torch.float32),
            name=element.id,
        )
    elif isinstance(element, ocelot.Cavity):
        return cheetah.Cavity(
            torch.tensor(element.l, dtype=torch.float32), name=element.id
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
            resolution=torch.tensor([2448, 2040]),
            pixel_size=torch.tensor([3.5488e-6, 2.5003e-6]),
            name=element.id,
        )
    elif isinstance(element, ocelot.Monitor) and "BPM" in element.id:
        return cheetah.BPM(name=element.id)
    elif isinstance(element, ocelot.Undulator):
        return cheetah.Undulator(
            torch.tensor(element.l, dtype=torch.float32), name=element.id
        )
    else:
        if warnings:
            print(
                f"WARNING: Unknown element {element.id} of type {type(element)},"
                " replacing with drift section."
            )
        return cheetah.Drift(
            length=torch.tensor(element.l, dtype=torch.float32), name=element.id
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
