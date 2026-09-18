import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from special_traj_quasi_parabola import generate_quasi_parabola
from datasets.td import TrajectoryDataset
from models.trajectory_mamba import TrajectoryMamba

OUT_DIR = 'e:/python/做一个有条理的人/mamba（4D-结项原版）/3D_run/radar_filter_runs'


"""
使用已训练模型对类抛物线轨迹进行跟踪，并可视化（按 v_simple 风格）
- 真值轨迹（蓝色实线）
- 量测点（橙色散点，添加高斯噪声）
- 模型跟踪轨迹（红色虚线 + 散点）
"""


def cartesian_to_polar_with_vr(p: np.ndarray, v: np.ndarray, radar_pos=np.array([0.0, 0.0, 0.0])):
    # p,v: (T,3)
    rel = p - radar_pos
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    # vr = 投影到视线方向的速度
    u = np.zeros_like(p)
    mask = r > 1e-6
    u[mask] = rel[mask] / r[mask][:, None]
    vr = np.sum(v * u, axis=1)
    az_deg = np.rad2deg(az)
    el_deg = np.rad2deg(el)
    return np.stack([r, az_deg, el_deg, vr], axis=1)


def build_traj_dict_from_parabola(T=360, dt=1.0, noise_std=(15.0, 15.0, 5.0)):
    p_true, v_true, info = generate_quasi_parabola(T=T, dt=dt)
    # 计算加速度（数值微分）
    a_true = np.zeros_like(v_true)
    a_true[1:] = (v_true[1:] - v_true[:-1]) / dt
    # 模拟量测
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 1.0, size=p_true.shape) * np.array(noise_std, dtype=float)
    meas_xyz = p_true + noise
    radar_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    meas_polar = cartesian_to_polar_with_vr(meas_xyz, v_true, radar_pos)
    traj = {
        'positions': p_true,
        'velocities': v_true,
        'accelerations': a_true,
        'meas_xyz': meas_xyz,
        'measurements_xy': None,
        'measurements_polar': meas_polar,
        'radar_pos': radar_pos,
        'trajectory_type': 'mixed',
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
    # 如果不支持 weights_only，降级为普通加载
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        raise RuntimeError('加载的模型不是 state_dict，请检查保存方式')
    model.eval()
    return model, meta


def main():
    # 构造与数据集兼容的 traj
    traj = build_traj_dict_from_parabola(T=360, dt=1.0)
    positions = traj['positions']
    measurements_xyz = traj['meas_xyz']

    # 加载模型与元信息
    model, meta = load_trained_model(OUT_DIR)
    seq_len = meta['seq_len']

    # 数据集与预测
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

    pos_rmse = 0.5*np.sqrt(np.mean(np.sum((pred_pos - true_pos) ** 2, axis=1)))

    # 可视化
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='true')
    ax.scatter(measurements_xyz[:, 0], measurements_xyz[:, 1], measurements_xyz[:, 2], c='orange', s=10, alpha=0.6, label='measurements')
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', label='tracked')
    ax.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], c='red', s=18, alpha=0.7)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Trajactory Position RMSE={pos_rmse:.3f} m')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
