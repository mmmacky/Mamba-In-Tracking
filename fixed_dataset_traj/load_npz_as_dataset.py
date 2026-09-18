import glob
import numpy as np

"""
将 fixed_dataset_traj/*.npz 加载成与 generate_trajectory() 相同结构的列表，
从而可直接喂给 datasets/td.py::TrajectoryDataset。

用法（示例）：
    from fixed_dataset_traj.load_npz_as_dataset import load_fixed_npz_list
    data = load_fixed_npz_list('fixed_dataset_traj/*.npz')
"""


def load_fixed_npz_list(pattern: str):
    out = []
    for fp in sorted(glob.glob(pattern)):
        z = np.load(fp, allow_pickle=True)
        out.append({
            'positions': z['positions'],
            'velocities': z['velocities'],
            'accelerations': z['accelerations'],
            'measurements_polar': z['measurements_polar'],
            'meas_xyz': z['meas_xyz'],
            'meas_xy': z['meas_xyz'][:, :2],
            'radar_pos': z['radar_pos'],
            'trajectory_type': str(z['trajectory_type']),
        })
    return out
