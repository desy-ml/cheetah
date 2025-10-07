import math
import torch
from cheetah import Segment
from cheetah import SpaceChargeKick2D


def add_space_charge_elements(segment: Segment, **kwargs) -> Segment:
    new_elements = []
    for element in segment.elements:
        sc_kick = SpaceChargeKick2D(element.length, **kwargs)
        new_elements.append(sc_kick)
        new_elements.append(element)
    return Segment(new_elements)


def split_segment(segment: Segment, n: int) -> Segment:
    slice_length = segment.length / float(n)
    elements = segment.split(resolution=slice_length)
    return Segment(elements)


def build_norm_matrix_from_twiss_2d(
    alpha: float, beta: float, eps: float = None
) -> torch.Tensor:
    V = torch.tensor([[beta, 0.0], [-alpha, 1.0]]) * math.sqrt(1.0 / beta)
    if eps is not None:
        eps = torch.as_tensor(eps)
        A = torch.diag(torch.sqrt([eps, eps]))
        V = torch.matmul(V, A)
    return torch.linalg.inv(V)
