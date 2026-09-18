import os
import sys
import numpy as np
# ================= 关键修复开始 =================
# 1. 获取当前脚本所在的目录 (例如: .../track_fixed)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (即 current_dir 的上一级: .../mamba（4D-结项原版）)
# 这样无论你在哪里运行，都能自动找到正确的根目录，不需要手动写死路径
project_root = os.path.dirname(current_dir)

# 3. 将根目录加入 Python 搜索路径
# 只有加了这一句，Python 才能跨文件夹找到 fixed_traj, datasets, models
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"当前脚本路径: {current_dir}")
print(f"已加载项目根目录: {project_root}")

# 复用你原来的生成逻辑（turn/mixed 的机动性来自 data/traj.py）
from data.traj import generate_trajectory

"""
从 data/traj.py 的 generate_trajectory() 中
“筛选出机动性较强的 turn/mixed 轨迹”，并将量测噪声固定下来。

实现要点：
1) 先用固定 seed 反复生成 turn/mixed 候选轨迹。
2) 计算一个“机动性得分”（基于加速度幅值、角速度/曲率近似、速度变化等）。
3) 挑出得分最高的 K 条。
4) 对每条轨迹：用该轨迹的 positions/velocities 重新生成量测，且噪声使用固定 RandomState，
   从而实现“轨迹确定 + 量测确定”。

本文件默认提供“生成并返回 list[dict]”版本，便于直接喂给 datasets/td.py::TrajectoryDataset。
"""


def _compute_maneuver_score(traj: dict, dt: float = 1.0) -> float:
    v = traj['velocities'].astype(np.float64)
    a = traj['accelerations'].astype(np.float64)

    speed = np.linalg.norm(v, axis=1)
    a_mag = np.linalg.norm(a, axis=1)

    # 1) 平均加速度与峰值加速度
    a_mean = float(np.mean(a_mag))
    a_peak = float(np.max(a_mag))

    # 2) 速度变化强度
    dv = np.diff(speed)
    dv_rms = float(np.sqrt(np.mean(dv * dv))) if dv.size else 0.0

    # 3) 水平转弯强度（用航向变化率近似角速度）
    vx, vy = v[:, 0], v[:, 1]
    heading = np.arctan2(vy, vx)
    # unwrap 避免 2pi 跳变
    heading = np.unwrap(heading)
    omega = np.diff(heading) / dt
    omega_rms = float(np.sqrt(np.mean(omega * omega))) if omega.size else 0.0
    omega_peak = float(np.max(np.abs(omega))) if omega.size else 0.0

    # 综合得分（权重可调）
    score = 2.0 * a_mean + 3.0 * a_peak + 1.5 * dv_rms + 3.0 * omega_rms + 2.0 * omega_peak
    return float(score)


def _fixed_measurements_from_truth(p, v, radar_pos, rng,
                                  std_r=5.0, std_az=1.0, std_el=1.0, std_vr=0.1):
    radar_pos = np.asarray(radar_pos, dtype=float)

    rel = p - radar_pos
    r_true = np.linalg.norm(rel, axis=1)
    r_true = np.maximum(r_true, 1e-6)

    az_true = np.arctan2(rel[:, 1], rel[:, 0])
    el_true = np.arcsin(rel[:, 2] / r_true)
    vr_true = np.sum(v * rel, axis=1) / r_true

    r_meas = r_true + rng.normal(scale=std_r, size=r_true.shape)
    az_meas_deg = np.rad2deg(az_true) + rng.normal(scale=std_az, size=r_true.shape)
    el_meas_deg = np.rad2deg(el_true) + rng.normal(scale=std_el, size=r_true.shape)
    vr_meas = vr_true + rng.normal(scale=std_vr, size=r_true.shape)

    r_meas = np.clip(r_meas, 1e-6, None)

    meas_polar = np.column_stack([r_meas, az_meas_deg, el_meas_deg, vr_meas]).astype(np.float32)

    az_rad = np.deg2rad(az_meas_deg)
    el_rad = np.deg2rad(el_meas_deg)
    x = r_meas * np.cos(el_rad) * np.cos(az_rad) + radar_pos[0]
    y = r_meas * np.cos(el_rad) * np.sin(az_rad) + radar_pos[1]
    z = r_meas * np.sin(el_rad) + radar_pos[2]
    meas_xyz = np.column_stack([x, y, z]).astype(np.float32)

    return meas_polar, meas_xyz


def generate_fixed_topk_dicts(
    K: int = 5,
    T: int = 200,
    dt: float = 1.0,
    radar_pos=(0.0, 0.0),
    candidate_seed: int = 1234,
    num_candidates: int = 2000,
    noise_std_r: float = 5.0,
    noise_std_az: float = 1.0,
    noise_std_el: float = 1.0,
    noise_std_vr: float = 0.1,
):
    """生成并返回机动性最强的 K 条轨迹（dict 列表），且量测固定。

    返回的每个 dict 字段与 generate_trajectory() / TrajectoryDataset 兼容：
    - positions (T,3)
    - velocities (T,3)
    - accelerations (T,3)
    - measurements_polar (T,4): [r, az_deg, el_deg, vr]
    - meas_xyz (T,3)
    - meas_xy (T,2)
    - radar_pos (3,)
    - trajectory_type (str)

    额外字段：
    - score（机动性评分，float）
    - dt
    """

    # 生成候选时固定全局随机，确保“筛选结果可复现”
    np.random.seed(candidate_seed)

    candidates = []
    for i in range(num_candidates):
        traj_type = 'turn' if (i % 2 == 0) else 'mixed'
        # 候选生成时仍按原逻辑生成（其 measurements_polar 带随机噪声，但后面会被我们重算并固定）
        traj = generate_trajectory(
            T=T,
            trajectory_type=traj_type,
            dt=dt,
            noise_std_r=noise_std_r,
            noise_std_az=noise_std_az,
            noise_std_el=noise_std_el,
            noise_std_vr=noise_std_vr,
            radar_pos=radar_pos,
        )
        score = _compute_maneuver_score(traj, dt=dt)
        candidates.append((score, traj_type, traj))

    candidates.sort(key=lambda x: x[0], reverse=True)
    topk = candidates[:K]

    out = []
    for idx, (score, traj_type, traj) in enumerate(topk, start=1):
        p = traj['positions'].astype(np.float64)
        v = traj['velocities'].astype(np.float64)
        a = traj['accelerations'].astype(np.float64)

        # 每条轨迹单独固定一个测量噪声 seed，确保每次都一样
        meas_rng = np.random.RandomState(9000 + idx)
        meas_polar, meas_xyz = _fixed_measurements_from_truth(
            p,
            v,
            radar_pos=np.array([radar_pos[0], radar_pos[1], 0.0], dtype=float),
            rng=meas_rng,
            std_r=noise_std_r,
            std_az=noise_std_az,
            std_el=noise_std_el,
            std_vr=noise_std_vr,
        )

        out.append({
            'positions': p.astype(np.float32),
            'velocities': v.astype(np.float32),
            'accelerations': a.astype(np.float32),
            'measurements_polar': meas_polar,
            'meas_xyz': meas_xyz,
            'meas_xy': meas_xyz[:, :2].astype(np.float32),
            'radar_pos': np.array([radar_pos[0], radar_pos[1], 0.0], dtype=np.float32),
            'trajectory_type': f"{traj_type}_fixed_from_dataset_{idx:02d}",
            'score': float(score),
            'dt': float(dt),
        })

    return out


# 兼容旧名字（如果你别处已经 import 了 generate_and_save_topk）
# 现在返回 dict 列表，不再写 npz。
def generate_and_save_topk(*args, **kwargs):  # noqa: D401
    """Deprecated: 改为直接返回轨迹 dict 列表。"""
    return generate_fixed_topk_dicts(*args, **kwargs)


if __name__ == '__main__':
    data_list = generate_fixed_topk_dicts(K=5)
    for i, d in enumerate(data_list, start=1):
        v = d['velocities']
        speed = np.linalg.norm(v, axis=1)
        print(
            f"[{i}] {d['trajectory_type']} | score={d['score']:.3f} | "
            f"speed=[{speed.min():.2f},{speed.max():.2f}] m/s | "
            f"meas_shape={d['measurements_polar'].shape}"
        )
