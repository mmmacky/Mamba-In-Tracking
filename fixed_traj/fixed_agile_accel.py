import numpy as np
import matplotlib.pyplot as plt

"""
强机动：强加速/强减速 + 垂直“跃升/骤降” + 横向抖动 的确定性轨迹（带固定量测）

设计目标：
- 构造速度模长呈“脉冲式”变化：短时强加速 -> 短时强减速 -> 再次强加速。
- 与此同时叠加横向蛇形（侧向加速度明显）以及高度的跃升/俯冲，提高机动性。
- 量测同 fixed_traj 其它文件：固定随机种子、输出 measurements_polar 和 meas_xyz。

注意：此轨迹不依赖 data/traj.py 的随机 generate_trajectory，而是手工确定性设计，便于评估滤波器对强机动的响应。
"""


def _polar_measure_with_fixed_noise(p, v, radar_pos, rng,
                                   std_r=8.0, std_az=0.7, std_el=0.7, std_vr=1.2):
    radar_pos = np.asarray(radar_pos, dtype=float)

    rel_p = p - radar_pos
    r_true = np.linalg.norm(rel_p, axis=1)
    r_true = np.maximum(r_true, 1e-6)

    az_true = np.arctan2(rel_p[:, 1], rel_p[:, 0])
    el_true = np.arcsin(rel_p[:, 2] / r_true)
    vr_true = np.sum(v * rel_p, axis=1) / r_true

    r_meas = r_true + rng.normal(0, std_r, size=r_true.shape)
    az_meas_deg = np.rad2deg(az_true) + rng.normal(0, std_az, size=r_true.shape)
    el_meas_deg = np.rad2deg(el_true) + rng.normal(0, std_el, size=r_true.shape)
    vr_meas = vr_true + rng.normal(0, std_vr, size=r_true.shape)

    measurements_polar = np.column_stack([r_meas, az_meas_deg, el_meas_deg, vr_meas]).astype(np.float32)

    az_m = np.deg2rad(az_meas_deg)
    el_m = np.deg2rad(el_meas_deg)
    meas_x = r_meas * np.cos(el_m) * np.cos(az_m) + radar_pos[0]
    meas_y = r_meas * np.cos(el_m) * np.sin(az_m) + radar_pos[1]
    meas_z = r_meas * np.sin(el_m) + radar_pos[2]
    meas_xyz = np.column_stack([meas_x, meas_y, meas_z]).astype(np.float32)

    return measurements_polar, meas_xyz


def generate_agile_accel_fixed(T=320, dt=1.0, radar_pos=(0, 0, 0)):
    # 分段：速度模长强变化
    b1, b2, b3 = 80, 160, 240

    # 基础航向
    heading0 = np.deg2rad(-15.0)

    # 速度模长 (m/s)
    v_base = 6.0
    v_peak1 = 18.0
    v_low = 4.0
    v_peak2 = 16.0

    # 高度机动：跃升/俯冲/再跃升
    vz1, vz2, vz3, vz4 = 0.3, 5.0, -4.0, 3.5

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)

    p[0] = np.array([3500.0, 1200.0, 400.0], dtype=float)

    for t in range(T):
        if t < b1:
            speed = v_base + (v_peak1 - v_base) * (t / max(1, b1))
            vz = vz1
        elif t < b2:
            # 快速减速
            frac = (t - b1) / max(1, (b2 - b1))
            speed = v_peak1 + (v_low - v_peak1) * frac
            vz = vz2
        elif t < b3:
            # 再次加速
            frac = (t - b2) / max(1, (b3 - b2))
            speed = v_low + (v_peak2 - v_low) * frac
            vz = vz3
        else:
            speed = v_peak2
            vz = vz4

        # 横向蛇形：用偏航摆动模拟横向机动（导致明显横向加速度）
        heading = heading0 + np.deg2rad(25.0) * np.sin(2 * np.pi * t / 22.0)

        # 叠加一点高度摆动
        vz = vz + 1.5 * np.sin(2 * np.pi * t / 35.0)

        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        v[t] = np.array([vx, vy, vz], dtype=float)

        if t > 0:
            p[t] = p[t - 1] + v[t] * dt

        p[t, 2] = np.clip(p[t, 2], 80.0, 3000.0)

    a = np.gradient(v, dt, axis=0)

    rng = np.random.RandomState(102)
    measurements_polar, meas_xyz = _polar_measure_with_fixed_noise(
        p, v, radar_pos=np.asarray(radar_pos, dtype=float), rng=rng,
        std_r=9.0, std_az=0.7, std_el=0.7, std_vr=1.4
    )

    info = {
        'positions': p.astype(np.float32),
        'velocities': v.astype(np.float32),
        'accelerations': a.astype(np.float32),
        'measurements_polar': measurements_polar,
        'meas_xyz': meas_xyz,
        'radar_pos': np.asarray(radar_pos, dtype=np.float32),
        'trajectory_type': 'agile_accel_fixed',
        'dt': dt
    }
    return info


if __name__ == '__main__':
    data = generate_agile_accel_fixed()
    p = data['positions']
    m = data['meas_xyz']

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'g-', linewidth=2, label='True')
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], c='orange', s=4, alpha=0.35, label='Meas')
    ax.set_title('Agile Accel Fixed Trajectory')
    ax.legend()
    plt.show()
