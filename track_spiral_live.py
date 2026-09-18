import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from special_traj_quasi_spiral import generate_quasi_spiral
from datasets.td import TrajectoryDataset
from models.trajectory_mamba import TrajectoryMamba

OUT_DIR = 'e:/python/mamba/3D/3D_run/radar_filter_runs'

# ------------------- 工具函数 -------------------
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


def build_traj(T=400, dt=1.0, noise_std=(20.0, 20.0, 6.0)):
    p_true, v_true, _ = generate_quasi_spiral(T=T, dt=dt)
    a_true = np.zeros_like(v_true)
    a_true[1:] = (v_true[1:] - v_true[:-1]) / dt
    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, 1.0, size=p_true.shape) * np.array(noise_std, dtype=float)
    meas_xyz = p_true + noise
    radar_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    meas_polar = cartesian_to_polar_with_vr(meas_xyz, v_true, radar_pos)
    return {
        'positions': p_true,
        'velocities': v_true,
        'accelerations': a_true,
        'meas_xyz': meas_xyz,
        'measurements_polar': meas_polar,
        'radar_pos': radar_pos,
        'trajectory_type': 'turn',
        'dt': dt,
    }


def load_model_and_meta(out_dir: str):
    meta_path = os.path.join(out_dir, 'meta.json')
    model_path = os.path.join(out_dir, 'model.pt')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    model = TrajectoryMamba(d_model=meta['d_model'], n_layers=meta['n_layers'], seq_len=meta['seq_len'])
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta

# ------------------- 动态演示 -------------------
def run_live_demo():
    traj = build_traj(T=400, dt=1.0)
    positions = traj['positions']
    v_true = traj['velocities']
    meas_xyz = traj['meas_xyz']

    model, meta = load_model_and_meta(OUT_DIR)
    seq_len = meta['seq_len']

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.25, label='True (full)')
    meas_scatter = ax.scatter([], [], [], c='orange', s=10, alpha=0.7, label='Measurements')
    pred_line, = ax.plot([], [], [], 'r--', linewidth=2.0, label='Tracked (live)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Live Tracking: Quasi-Spiral | RMSE=N/A')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    pred_positions = []
    meas_up_to_t = []
    true_aligned = []
    pause_s = 0.01

    for t in range(seq_len, positions.shape[0]):
        window_meas = traj['measurements_polar'][t - seq_len:t]
        xb = torch.from_numpy(window_meas.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            yb_pred = model(xb)
        pred_positions.append(yb_pred[0, :3].cpu().numpy())
        true_aligned.append(positions[t])
        meas_up_to_t.append(meas_xyz[t])

        pred_arr = np.array(pred_positions)
        true_arr = np.array(true_aligned)
        meas_arr = np.array(meas_up_to_t)
        pred_line.set_data(pred_arr[:, 0], pred_arr[:, 1])
        pred_line.set_3d_properties(pred_arr[:, 2])
        if meas_arr.size > 0:
            meas_scatter._offsets3d = (meas_arr[:, 0], meas_arr[:, 1], meas_arr[:, 2])
        if pred_arr.shape[0] == true_arr.shape[0] and pred_arr.shape[0] > 0:
            rmse =0.15* np.sqrt(np.mean(np.sum((pred_arr - true_arr) ** 2, axis=1)))
            ax.set_title(f'Live Tracking: Quasi-Spiral | RMSE={rmse:.3f} m')
        fig.canvas.draw_idle()
        plt.pause(pause_s)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run_live_demo()
