from typing import Optional, Union

import torch
from scipy import constants
from scipy.constants import physical_constants
from torch import nn

from cheetah.utils import UniqueNameGenerator

from .dipole import Dipole

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = torch.tensor(
    constants.electron_mass
    * constants.speed_of_light**2
    / constants.elementary_charge  # electron mass
)
electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class RBend(Dipole):
    """
    Rectangular bending magnet.

    :param length: Length in meters.
    :param angle: Deflection angle in rad.
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
        length: Optional[Union[torch.Tensor, nn.Parameter]],
        angle: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        e1: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        e2: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        tilt: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        fringe_integral: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        fringe_integral_exit: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        gap: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ):
        angle = angle if angle is not None else torch.tensor(0.0)
        e1 = e1 if e1 is not None else torch.tensor(0.0)
        e2 = e2 if e2 is not None else torch.tensor(0.0)
        tilt = tilt if tilt is not None else torch.tensor(0.0)
        fringe_integral = (
            fringe_integral if fringe_integral is not None else torch.tensor(0.0)
        )
        # fringe_integral_exit is left out on purpose
        gap = gap if gap is not None else torch.tensor(0.0)

        e1 = e1 + angle / 2
        e2 = e2 + angle / 2

        super().__init__(
            length=length,
            angle=angle,
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
