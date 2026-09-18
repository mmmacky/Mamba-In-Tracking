from torch.utils.data import Dataset
import numpy as np
import torch


class TrajectoryDataset(Dataset):
    """
    Dataset that provides sliding-window samples for radar filtering.

    输入: (r, az_deg, el_deg, vr) - shape (L, 4) - 极坐标量测 (距离, 方位角(度), 俯仰角(度), 径向速度量测)
    输出: (x, y, z, vr_true) - shape (4,) - 笛卡尔位置 + 真实径向速度

    参数:
        data: 轨迹数据列表，每条轨迹是generate_trajectory()返回的字典
        seq_len: 滑动窗口长度
        radar_pos: 雷达位置 (x, y, z) 三维坐标（必须提供三维）
    """

    def __init__(self, data, seq_len=20, radar_pos=(0,0,0)):
        self.seq_len = seq_len
        # 要求传入三维雷达位置
        rp = np.asarray(radar_pos, dtype=np.float32)
        if rp.size != 3:
            raise ValueError('radar_pos must be a 3-element sequence (x, y, z)')
        self.radar_pos = rp.astype(np.float32)

        self.samples = []  # tuples (input_measurements, target)
        self.trajectory_ids = []  # 记录每个样本来自哪条轨迹

        for traj_idx, traj in enumerate(data):
            # 获取数据
            meas_polar = traj['measurements_polar']  # (T, 4) - (r, az_deg, el_deg, vr_meas)
            pos_cartesian = traj['positions']  # (T, 3) - (x, y, z)
            vel_cartesian = traj['velocities']  # (T, 3) - (vx, vy, vz)

            T = meas_polar.shape[0]

            # 计算真实径向速度，使用三维雷达位置
            p_relative = pos_cartesian - self.radar_pos  # 相对雷达的位置 (T,3)
            r_magnitude = np.linalg.norm(p_relative, axis=1, keepdims=True)  # (T, 1)
            r_magnitude = np.maximum(r_magnitude, 1e-10)
            vr_true = np.sum(vel_cartesian * p_relative, axis=1) / r_magnitude.squeeze()  # (T,)

            for t in range(seq_len - 1, T):
                # 输入: 极坐标量测序列 (r, az_deg, el_deg, vr_meas)
                inp = meas_polar[t - seq_len + 1:t + 1]  # (seq_len, 4)

                # 输出: 笛卡尔位置 (x,y,z) + 真实径向速度
                tgt = np.concatenate([
                    pos_cartesian[t],  # (3,) - (x, y, z)
                    [vr_true[t]]       # (1,) - vr_true
                ])  # (4,)

                self.samples.append((
                    inp.astype(np.float32),
                    tgt.astype(np.float32)
                ))
                self.trajectory_ids.append(traj_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.from_numpy(inp), torch.from_numpy(tgt)

    def get_trajectory_info(self, idx):
        """获取样本的轨迹信息"""
        return self.trajectory_ids[idx]

    def get_data_info(self):
        """获取数据集信息"""
        return {
            'num_samples': len(self.samples),
            'seq_len': self.seq_len,
            'radar_pos': self.radar_pos.tolist()
        }