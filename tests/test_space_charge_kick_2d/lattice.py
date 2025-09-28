import torch
from cheetah import Segment
from cheetah import Drift
from cheetah import Quadrupole
from cheetah import SpaceChargeKick2D


def slice_segment(segment: Segment, nslice: int) -> Segment:
    slice_length = segment.length / float(nslice)
    elements = segment.split(resolution=slice_length)
    return Segment(elements)    


def build_fodo_segment_one_cell(length: float, kq: float, nslice: int = None) -> Segment:
    kq = torch.as_tensor(kq)
    length = torch.as_tensor(length)

    fill_frac = 0.5
    length_quad = 0.25 * length * fill_frac
    length_drift = 0.25 * length * (1.0 - fill_frac)

    elements = [
        Quadrupole(length=length_quad, k1=kq),
        Drift(length=length_drift),
        Drift(length=length_drift),
        Quadrupole(length=length_quad, k1=-kq),
        Quadrupole(length=length_quad, k1=-kq),
        Drift(length=length_drift),
        Drift(length=length_drift),
        Quadrupole(length=length_quad, k1=kq),
    ]
    segment = Segment(elements)

    if nslice is not None:
        segment = slice_segment(segment, nslice=nslice)

    return segment


def build_fodo_segment(periods: int = 1, **kwargs) -> Segment:
    elements = []
    for _ in range(periods):
        segment = build_fodo_segment_one_cell(**kwargs)
        elements.extend(segment.elements)
    return Segment(elements)


def add_space_charge_elements(segment: Segment, **kwargs) -> Segment:
    new_elements = []
    for element in segment.elements:
        sc_kick = SpaceChargeKick2D(element.length, **kwargs)
        new_elements.append(sc_kick)
        new_elements.append(element)
    return Segment(new_elements)
