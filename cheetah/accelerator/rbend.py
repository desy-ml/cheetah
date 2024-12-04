from typing import Optional

import torch

from cheetah.accelerator.dipole import Dipole
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class RBend(Dipole):
    """
    Rectangular bending magnet.

    :param length: Length in meters.
    :param angle: Deflection angle in rad.
    :param k1: Focussing strength in 1/m^-2.
    :param e1: The angle of inclination of the entrance face [rad].
    :param e2: The angle of inclination of the exit face [rad].
    :param tilt: Tilt of the magnet in x-y plane [rad].
    :param fringe_integral: Fringe field integral (of the enterance face).
    :param fringe_integral_exit: (only set if different from `fint`) Fringe field
        integral of the exit face.
    :param gap: The magnet gap [m], NOTE in MAD and ELEGANT: HGAP = gap/2
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Optional[torch.Tensor],
        angle: Optional[torch.Tensor] = None,
        k1: Optional[torch.Tensor] = None,
        e1: Optional[torch.Tensor] = None,
        e2: Optional[torch.Tensor] = None,
        tilt: Optional[torch.Tensor] = None,
        fringe_integral: Optional[torch.Tensor] = None,
        fringe_integral_exit: Optional[torch.Tensor] = None,
        gap: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            length=length,
            angle=angle,
            k1=k1,
            e1=e1,
            e2=e2,
            tilt=tilt,
            fringe_integral=fringe_integral,
            fringe_integral_exit=fringe_integral_exit,
            gap=gap,
            name=name,
            device=device,
            dtype=dtype,
        )

        # Rectangular bend
        self.e1 = self.e1 + self.angle / 2
        self.e2 = self.e2 + self.angle / 2
