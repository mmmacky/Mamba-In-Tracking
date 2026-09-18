import os
import sys

# 让脚本在任意工作目录运行都能找到项目包
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from fixed_dataset_traj.gen_fixed_strong_from_dataset import generate_fixed_topk_dicts
from datasets.td import TrajectoryDataset
from models.trajectory_mamba import TrajectoryMamba

# 模型输出目录（与 track_fixed 其它脚本一致）
MODEL_OUT_DIR = 'e:/python/做一个有条理的人/mamba（4D-结项原版）/3D_run/radar_filter_runs'

# 图片保存目录
PLOT_OUT_DIR = os.path.join(project_root, '3D_run', 'run_test', 'topk_fixed_dataset')


def load_trained_model(out_dir: str):
    meta_path = os.path.join(out_dir, 'meta.json')
    model_path = os.path.join(out_dir, 'model.pt')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    model = TrajectoryMamba(d_model=meta['d_model'], n_layers=meta['n_layers'], seq_len=meta['seq_len'])
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model.load_state_dict(state)

    model.eval()
    return model, meta


def track_one(model, meta, traj_dict):
    seq_len = meta['seq_len']

    ds = TrajectoryDataset([traj_dict], seq_len=seq_len, radar_pos=(0, 0, 0))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

    preds = []
    with torch.no_grad():
        for xb, _yb in loader:
            pred = model(xb)
            preds.append(pred.cpu().numpy())

    preds = np.vstack(preds) if preds else np.zeros((0, 4), dtype=np.float32)
    pred_pos = preds[:, :3]

    positions = traj_dict['positions']
    meas_xyz = traj_dict.get('meas_xyz', None)

    start_idx = seq_len - 1
    end_idx = start_idx + len(pred_pos)
    truth_slice = positions[start_idx:end_idx]

    if meas_xyz is not None:
        meas_slice = meas_xyz[start_idx:end_idx]
    else:
        meas_slice = None

    if len(pred_pos) > 0 and len(truth_slice) == len(pred_pos):
        rmse = float(np.sqrt(np.mean(np.sum((pred_pos - truth_slice) ** 2, axis=1))))
    else:
        rmse = float('nan')

    return {
        'pred_pos': pred_pos,
        'truth_slice': truth_slice,
        'meas_slice': meas_slice,
        'rmse': rmse,
        'seq_len': seq_len,
    }


def plot_and_save(traj_dict, track_res, save_path: str):
    p = traj_dict['positions']

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # full truth
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', alpha=0.25, label='Ground Truth (Full)')

    # segment truth
    ts = track_res['truth_slice']
    ax.plot(ts[:, 0], ts[:, 1], ts[:, 2], 'b-', linewidth=2, label='Ground Truth (Segment)')

    # measurements
    ms = track_res['meas_slice']
    if ms is not None:
        ax.scatter(ms[:, 0], ms[:, 1], ms[:, 2], c='orange', s=5, alpha=0.5, label='Measurements')

    # predictions
    pp = track_res['pred_pos']
    if len(pp) > 0:
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], 'r--', linewidth=2, label='Mamba Tracked')

    title = f"{traj_dict['trajectory_type']} (RMSE={track_res['rmse']:.2f}m)"
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main():
    os.makedirs(PLOT_OUT_DIR, exist_ok=True)

    model, meta = load_trained_model(MODEL_OUT_DIR)

    # 生成 5 条“强机动 + 固定量测”的轨迹（来自 generate_trajectory 的 turn/mixed）
    traj_list = generate_fixed_topk_dicts(K=5, T=200, dt=1.0, radar_pos=(0.0, 0.0))

    for i, traj in enumerate(traj_list, start=1):
        res = track_one(model, meta, traj)
        save_path = os.path.join(PLOT_OUT_DIR, f"topk_{i:02d}_{traj['trajectory_type']}.png")
        plot_and_save(traj, res, save_path)
        print(f"[{i}/5] saved plot: {save_path}")


if __name__ == '__main__':
    main()
