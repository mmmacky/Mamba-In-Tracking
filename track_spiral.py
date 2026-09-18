import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from special_traj_quasi_spiral import generate_quasi_spiral
from datasets.td import TrajectoryDataset
from models.trajectory_mamba import TrajectoryMamba

OUT_DIR = 'e:/python/做一个有条理的人/mamba（4D-结项原版）/3D_run/radar_filter_runs'

"""
使用已训练模型对类螺旋轨迹进行跟踪，并可视化（按 v_simple 风格）：
- 真值轨迹（蓝色实线）
- 量测点（橙色散点，添加高斯噪声）
- 模型跟踪轨迹（红色虚线 + 散点）
"""


def cartesian_to_polar_with_vr(p: np.ndarray, v: np.ndarray, radar_pos=np.array([0.0, 0.0, 0.0])):
    rel = p - radar_pos
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    u = np.zeros_like(p)
    mask = r > 1e-6
    u[mask] = rel[mask] / r[mask][:, None]
    vr = np.sum(v * u, axis=1)
    az_deg = np.rad2deg(az)
    el_deg = np.rad2deg(el)
    return np.stack([r, az_deg, el_deg, vr], axis=1)


def build_traj_dict_from_spiral(T=400, dt=1.0, noise_std_r=5.0, noise_std_az=1.0,  noise_std_el=1.0, noise_std_vr=0.1):
    """
    使用正确的极坐标噪声模型（与 traj.py 一致）
    noise_std_r: 距离噪声 (米)
    noise_std_az: 方位角噪声 (度)
    noise_std_el: 俯仰角噪声 (度)
    noise_std_vr: 径向速度噪声 (m/s)
    """
    p_true, v_true, info = generate_quasi_spiral(T=T, dt=dt)
    a_true = np.zeros_like(v_true)
    a_true[1:] = (v_true[1:] - v_true[:-1]) / dt
    
    radar_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    
    # 计算真值极坐标
    polar_true = cartesian_to_polar_with_vr(p_true, v_true, radar_pos)
    r_true = polar_true[:, 0]
    az_true_deg = polar_true[:, 1]
    el_true_deg = polar_true[:, 2]
    vr_true = polar_true[:, 3]
    
    # 在极坐标域添加噪声（使用固定种子保证可重复）
    rng = np.random.default_rng(123)
    r_meas = r_true + rng.normal(0.0, noise_std_r, size=T)
    az_meas_deg = az_true_deg + rng.normal(0.0, noise_std_az, size=T)
    el_meas_deg = el_true_deg + rng.normal(0.0, noise_std_el, size=T)
    vr_meas = vr_true + rng.normal(0.0, noise_std_vr, size=T)
    
    # 确保距离为正
    r_meas = np.clip(r_meas, 1e-6, None)
    
    # 将带噪极坐标转换回笛卡尔坐标（用于可视化）
    az_meas_rad = np.deg2rad(az_meas_deg)
    el_meas_rad = np.deg2rad(el_meas_deg)
    meas_x = r_meas * np.cos(el_meas_rad) * np.cos(az_meas_rad) + radar_pos[0]
    meas_y = r_meas * np.cos(el_meas_rad) * np.sin(az_meas_rad) + radar_pos[1]
    meas_z = r_meas * np.sin(el_meas_rad) + radar_pos[2]
    meas_xyz = np.stack([meas_x, meas_y, meas_z], axis=1)
    
    # 构造带噪极坐标量测（这是模型的输入）
    meas_polar = np.stack([r_meas, az_meas_deg, el_meas_deg, vr_meas], axis=1)
    
    traj = {
        'positions': p_true,
        'velocities': v_true,
        'accelerations': a_true,
        'meas_xyz': meas_xyz,
        'measurements_xy': None,
        'measurements_polar': meas_polar,
        'radar_pos': radar_pos,
        'trajectory_type': 'turn',
    }
    return traj


def load_trained_model(out_dir: str):
    meta_path = os.path.join(out_dir, 'meta.json')
    model_path = os.path.join(out_dir, 'model.pt')
    if not (os.path.exists(meta_path) and os.path.exists(model_path)):
        raise FileNotFoundError('meta.json 或 model.pt 未找到，请确认训练结果位于 3D_run/radar_filter_runs')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    model = TrajectoryMamba(d_model=meta['d_model'], n_layers=meta['n_layers'], seq_len=meta['seq_len'])
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        raise RuntimeError('加载的模型不是 state_dict，请检查保存方式')
    model.eval()
    return model, meta


def main():
    # 使用与 traj.py 一致的噪声参数
    traj = build_traj_dict_from_spiral(T=400, dt=1.0, 
                                        noise_std_r=5.0,
                                        noise_std_az=1.0,
                                        noise_std_el=1.0,
                                        noise_std_vr=0.1)
    positions = traj['positions']
    measurements_xyz = traj['meas_xyz']

    model, meta = load_trained_model(OUT_DIR)
    seq_len = meta['seq_len']

    traj_ds = TrajectoryDataset([traj], seq_len=seq_len)
    loader = torch.utils.data.DataLoader(traj_ds, batch_size=64, shuffle=False)

    preds = []
    trues = []
    time_indices = []
    global_start = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)  # (B,4): x,y,z,vr
            pred_np = pred.cpu().numpy()
            true_np = yb.cpu().numpy()
            batch_size = xb.shape[0]
            for i in range(batch_size):
                time_idx = global_start + i + seq_len - 1
                time_indices.append(time_idx)
            global_start += batch_size
            preds.append(pred_np)
            trues.append(true_np)

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    time_indices = np.array(time_indices)

    pred_pos = preds[:, :3]
    true_pos = trues[:, :3]
    pos_rmse = np.sqrt(np.mean(np.sum((pred_pos - true_pos) ** 2, axis=1)))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='True')
    ax.scatter(measurements_xyz[:, 0], measurements_xyz[:, 1], measurements_xyz[:, 2], c='orange', s=10, alpha=0.6, label='Measurements')
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', label='Tracked')
    ax.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], c='red', s=18, alpha=0.7)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Model Tracking: Quasi-Spiral | Pos RMSE={pos_rmse:.3f} m')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
