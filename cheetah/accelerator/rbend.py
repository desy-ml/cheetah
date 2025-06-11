from typing import Literal

import torch

from cheetah.accelerator.dipole import Dipole
from cheetah.utils import UniqueNameGenerator, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class RBend(Dipole):
    """
    Rectangular bending magnet.

    :param length: Length in meters.
    :param angle: Deflection angle in rad.
    :param k1: Focussing strength in 1/m^-2.
    :param rbend_e1: The angle of inclination of the entrance face in rad.
    :param rbend_e2: The angle of inclination of the exit face in rad.
    :param gap: The magnet gap in meters. Note that in MAD and ELEGANT: HGAP = gap/2.
    :param gap_exit: The magnet gap at the exit in meters. Note that in MAD and
        ELEGANT: HGAP = gap/2. Only set if different from `gap`. Only used with
        `"bmadx"` tracking method.
    :param fringe_integral: Fringe field integral (of the enterance face).
    :param fringe_integral_exit: Fringe field integral of the exit face. Only set if
        different from `fringe_integral`. Only used with `"bmadx"` tracking method.
    :param fringe_at: Where to apply the fringe fields for `"bmadx"` tracking. The
        available options are:
        - "neither": Do not apply fringe fields.
        - "entrance": Apply fringe fields at the entrance end.
        - "exit": Apply fringe fields at the exit end.
        - "both": Apply fringe fields at both ends.
    :param fringe_type: Type of fringe field for `"bmadx"` tracking. Currently only
        supports `"linear_edge"`.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python
        variable name. This is needed if you want to use the `segment.element_name`
        syntax to access the element in a segment.
    """

    def __init__(
        self,
        length: torch.Tensor,
        angle: torch.Tensor | None = None,
        k1: torch.Tensor | None = None,
        rbend_e1: torch.Tensor | None = None,
        rbend_e2: torch.Tensor | None = None,
        tilt: torch.Tensor | None = None,
        gap: torch.Tensor | None = None,
        gap_exit: torch.Tensor | None = None,
        fringe_integral: torch.Tensor | None = None,
        fringe_integral_exit: torch.Tensor | None = None,
        fringe_at: Literal["neither", "entrance", "exit", "both"] = "both",
        fringe_type: Literal["linear_edge"] = "linear_edge",
        tracking_method: Literal["cheetah", "bmadx"] = "cheetah",
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        # Set default values needed for conversion from RBend to Dipole
        device, dtype = verify_device_and_dtype(
            [
                length,
                angle,
                k1,
                rbend_e1,
                rbend_e2,
                tilt,
                gap,
                gap_exit,
                fringe_integral,
                fringe_integral_exit,
            ],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        angle = (
            torch.as_tensor(angle, **factory_kwargs)
            if angle is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        rbend_e1 = (
            torch.as_tensor(rbend_e1, **factory_kwargs)
            if rbend_e1 is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        rbend_e2 = (
            torch.as_tensor(rbend_e2, **factory_kwargs)
            if rbend_e2 is not None
            else torch.tensor(0.0, **factory_kwargs)
        )

        super().__init__(
            length=length,
            angle=angle,
            k1=k1,
            dipole_e1=rbend_e1 + angle / 2,
            dipole_e2=rbend_e2 + angle / 2,
            tilt=tilt,
            gap=gap,
            gap_exit=gap_exit,
            fringe_integral=fringe_integral,
            fringe_integral_exit=fringe_integral_exit,
            fringe_at=fringe_at,
            fringe_type=fringe_type,
            tracking_method=tracking_method,
            name=name,
            sanitize_name=sanitize_name,
            device=device,
            dtype=dtype,
        )

    @property
    def rbend_e1(self):
        return self.dipole_e1 - self.angle / 2

    @rbend_e1.setter
    def rbend_e1(self, value):
        self.dipole_e1 = value + self.angle / 2

    @property
    def rbend_e2(self):
        return self.dipole_e2 - self.angle / 2

    @rbend_e2.setter
    def rbend_e2(self, value):
        self.dipole_e2 = value + self.angle / 2

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"angle={repr(self.angle)}, "
            + f"k1={repr(self.k1)}, "
            + f"rbend_e1={repr(self.rbend_e1)},"
            + f"rbend_e2={repr(self.rbend_e2)},"
            + f"tilt={repr(self.tilt)},"
            + f"gap={repr(self.gap)},"
            + f"gap_exit={repr(self.gap_exit)},"
            + f"fringe_integral={repr(self.fringe_integral)},"
            + f"fringe_integral_exit={repr(self.fringe_integral_exit)},"
            + f"fringe_at={repr(self.fringe_at)},"
            + f"fringe_type={repr(self.fringe_type)},"
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )

    @property
    def defining_features(self):
        dipole_features = super().defining_features
        dipole_features.remove("dipole_e1")
        dipole_features.remove("dipole_e2")
        return dipole_features + ["rbend_e1", "rbend_e2"]
