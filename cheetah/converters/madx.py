import pyoptics.tfsdata as tfs
import pandas as pd

import cheetah
import torch

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
    tfs_header = tfs_dict['header']
    tfs_col_names = tfs_dict['_col_names']
    tfs_keywords = list(set(tfs_dict['keyword']))
    # print(tfs_keywords)

    keys_to_remove = ['header', '_col_names', '_header_names', '_filename']
    tfs_dict_updated = {key: value for key, value in tfs_dict.items() if key not in keys_to_remove}
    tfs_df = pd.DataFrame(tfs_dict_updated)

    return tfs_df, tfs_header, tfs_col_names, tfs_keywords


def convert_madx_element(df_row):
    """
    Input: row from tfs file in a pandas dataframe
    """

    # RBEND -------------------------------------------------------------------
    """
    Rectangular bending magnet.

    L --> param length: Length in meters.
    ANGLE --> param angle: Deflection angle in rad.
    E1 --> param e1: The angle of inclination of the entrance face [rad].
    E2 --> param e2: The angle of inclination of the exit face [rad].
    TILT (horizontal, pi/2 turns it to vertical bend) --> param tilt: Tilt of the magnet in x-y plane [rad].
    FINT --> param fringe_integral: Fringe field integral (of the enterance face).
    FINTX --> param fringe_integral_exit: (only set if different from `fint`) Fringe field
        integral of the exit face.
    HGAP --> param gap: The magnet gap [m], NOTE in MAD and ELEGANT: HGAP = gap/2
    NAME --> param name: Unique identifier of the element.

    Not available:
    - K0: assignment of relative field errors to a bending magnet
    """
    if df_row.keyword == 'rbend':
        return cheetah.RBend(
            length=torch.tensor([df_row.l]),
            angle=torch.tensor([df_row.angle]),
            e1=torch.tensor([df_row.e1]),
            e2=torch.tensor([df_row.e2]),
            tilt=torch.tensor([df_row.tilt]),
            fringe_integral=torch.tensor([df_row.fint]),
            fringe_integral_exit=torch.tensor([df_row.fintx]),
            gap=torch.tensor([df_row.hgap]),
            name=df_row.name,
        )
    else:
        pass


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

#     # To do: list of MAD-X elements, then compare to loaded keywords

#     madx_rbends_pd = tfs_df[tfs_df['keyword'] == 'rbend']
#     madx_quads_pd = tfs_df[tfs_df['keyword'] == 'quadrupole']
#     madx_kicker_pd = tfs_df[tfs_df['keyword'] == 'kicker']
#     madx_instrument_pd = tfs_df[tfs_df['keyword'] == 'instrument']
#     madx_drifts_pd = tfs_df[tfs_df['keyword'] == 'drift']
#     print(madx_kicker_pd)
# # ---------------------------------------------------------------