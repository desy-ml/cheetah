import numpy as np
import pandas as pd
import pyoptics.tfsdata as tfs
import torch

import cheetah


def load_tfs(madx_tfs_file_path) -> dict:
    """
    Load the twiss file (tfs format) in a dictionary using the pyoptics tools
    """
    return tfs.open(madx_tfs_file_path)


def tfs_to_pandas(madx_tfs_file_path):
    """
    Load the twiss dictionary in a pandas dataframe for easy and quick inspection
    To do: add the header info as metadata to the dataframe, or similar
    """
    tfs_dict = load_tfs(madx_tfs_file_path)
    tfs_header = tfs_dict["header"]
    tfs_col_names = tfs_dict["_col_names"]
    tfs_keywords = list(set(tfs_dict["keyword"]))
    # print(tfs_keywords)

    keys_to_remove = ["header", "_col_names", "_header_names", "_filename"]
    tfs_dict_updated = {
        key: value for key, value in tfs_dict.items() if key not in keys_to_remove
    }
    tfs_df = pd.DataFrame(tfs_dict_updated)
    # madx_rbends_pd = tfs_df[tfs_df['keyword'] == 'rbend']

    return tfs_df, tfs_header, tfs_col_names, tfs_keywords


def convert_madx_element(df_row):
    """
    Input: row from tfs file in a pandas dataframe
    """

    # RBEND -------------------------------------------------------------------
    # Rectangular bending magnet.

    # L --> param length: Length in meters.
    # ANGLE --> param angle: Deflection angle in rad.
    # E1 --> param e1: The angle of inclination of the entrance face [rad].
    # E2 --> param e2: The angle of inclination of the exit face [rad].
    # TILT (horizontal, pi/2 turns it to vertical bend
    #     --> param tilt: Tilt of the magnet in x-y plane [rad].
    # FINT --> param fringe_integral: Fringe field integral (of the enterance face).
    # FINTX --> param fringe_integral_exit: (only set if different from `fint`)
    #     Fringe field integral of the exit face.
    # HGAP --> param gap: The magnet gap [m], NOTE in MAD and ELEGANT: HGAP = gap/2
    # NAME --> param name: Unique identifier of the element.

    # Not available in Cheetah:
    # - K0L: The integrated strength of the dipole component of the field in the magnet.

    if df_row.keyword == "rbend":
        return cheetah.RBend(
            length=torch.tensor([df_row.l]),
            angle=torch.tensor([df_row.angle]),
            e1=torch.tensor([df_row.e1]),
            e2=torch.tensor([df_row.e2]),
            tilt=torch.tensor([df_row.tilt]),
            fringe_integral=torch.tensor([df_row.fint]),
            fringe_integral_exit=torch.tensor([df_row.fintx]),
            gap=torch.tensor([df_row.hgap]),
            name=df_row["name"],
        )

    # QUADRUPOLE -------------------------------------------------------------------
    # Quadrupole magnet in a particle accelerator.

    # L --> param length: Length in meters.
    # K1L (1/m) --> param k1: Strength of the quadrupole in rad/m.
    # ! VERIFICATION NEEDED THAT K1L UNITS ARE CORRECT (SIN MIGHT BE MISSING)
    # not available in MAD-X: param misalignment: Misalignment vector of the quadrupole
    # in x- and y-directions.
    # TILT (rad) --> param tilt: Tilt angle of the quadrupole in x-y plane [rad].
    #   pi/4 for skew-quadrupole.

    # Not available in Cheetah:
    # - K1S (m^-2): skew quadrupole coeff.
    # - KTAP: The relative change of the quadrupoles strengths (both K1 and K1S)
    #   to account for energy change of the reference particle due to RF cavities or
    #   synchrotron radiation. The actual strength of the quadrupole is calculated
    #   as K1act = K1 * (1 + KTAP)

    elif df_row.keyword == "quadrupole":
        return cheetah.Quadrupole(
            length=torch.tensor([df_row.l]),
            k1=torch.tensor([df_row.k1l]),
            tilt=torch.tensor([df_row.tilt]),
            name=df_row["name"],
        )

    # KICKERS -------------------------------------------------------------------

    # Horizontal and vertical magnetic correctors
    # L --> param length: Length in meters.
    # HKICK/VKICK --> ARCSIN(param angle): Particle deflection angle in the
    #   horizontal plane in rad.
    # HKICK/VKICK is the momentum change DPX/DP0 or DPY/DP0. In order to get
    #   the deviation angle:
    #   sin(angle) = HKICK = DPX/DP0
    #   sin(angle) = VKICK = DPX/DP0

    # Not available in Cheetah:
    # - TILT (rad): The roll angle about the longitudinal axis

    elif df_row.keyword == "hkicker":
        return cheetah.HorizontalCorrector(
            length=torch.tensor([df_row.l]),
            angle=torch.tensor([np.arcsin(df_row.hkick)]),
            name=df_row["name"],
        )

    elif df_row.keyword == "vkicker":
        return cheetah.VerticalCorrector(
            length=torch.tensor([df_row.l]),
            angle=torch.tensor([np.arcsin(df_row.vkick)]),
            name=df_row["name"],
        )

    # Corrector for both planes
    # NOT AVAILABLE IN CHEETAH YET
    # elif df_row.keyword == "kicker":
    #     return cheetah.Corrector(
    #         length=torch.tensor([df_row.l]),
    #         angle_v=torch.tensor([np.arcsin(df_row.vkick)]),
    #         angle_h=torch.tensor([np.arcsin(df_row.hkick)]),
    #         name=df_row["name"],
    #     )

    # UNKNOWN ELEMENTS -----------------------------------------------------------------
    # Transformed to drifts and keeping the original name
    else:
        print(df_row["name"])
        return cheetah.Drift(
            length=torch.tensor([df_row.l]),
            name=df_row["name"],
        )


def convert_madx_lattice(madx_tfs_file_path):
    """
    Convert a lattice from a tfs file to a cheetah segment
    """
    tfs_df, tfs_header, tfs_col_names, tfs_keywords = tfs_to_pandas(madx_tfs_file_path)

    elements = []

    for index, row in tfs_df.iterrows():
        cheetah_element = convert_madx_element(row)
        if cheetah_element is not None:
            elements.append(cheetah_element)
    # print(elements)
    return cheetah.Segment(elements)
