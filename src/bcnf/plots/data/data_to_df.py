import os
import pickle

import pandas as pd

from bcnf.utils import get_dir


def data_to_df(name: str,
               pop_entries: list = ['traj', 'cams']
               ) -> pd.DataFrame:
    with open(os.path.join(get_dir('data', 'bcnf-data'), f'{name}.pkl'), 'rb') as f:
        data = pickle.load(f)

    # split up the data which consists of arrays
    cam_radians = data['cam_radian']
    num_cam_radians = len(cam_radians[0])

    for i in range(num_cam_radians):
        data['cam_radian_' + str(i)] = [x[i] for x in cam_radians]

    cam_angles = data['cam_angles']
    num_cam_angles = len(cam_angles[0])

    for i in range(num_cam_angles):
        data['cam_angle_' + str(i)] = [x[i] for x in cam_angles]

    cam_heights = data['cam_heights']
    num_cam_heights = len(cam_heights[0])

    for i in range(num_cam_heights):
        data['cam_height_' + str(i)] = [x[i] for x in cam_heights]

    data.pop('cam_radian')
    data.pop('cam_angles')
    data.pop('cam_heights')

    if pop_entries:
        for p in pop_entries:
            data.pop(p)

    return pd.DataFrame(data)
