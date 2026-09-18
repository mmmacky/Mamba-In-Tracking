import numpy as np
import matplotlib.pyplot as plt

"""
固定 mixed 轨迹 #1：
- 段1：中高速 CV 进入
- 段2：强 CA 加速
- 段3：强 CT 急转（omega 较大）
- 段4：强减速 + 俯冲

特点：轨迹确定；量测噪声使用固定 RandomState(seed=201)，每次运行量测一致。
输出字段与数据集兼容：positions/velocities/accelerations/measurements_polar/meas_xyz/radar_pos/trajectory_type/dt
measurements_polar: [r, az_deg, el_deg, vr]
"""


def _polar_measure_fixed(p, v, radar_pos, rng, std_r=10.0, std_az=0.8, std_el=0.8, std_vr=1.6):
    radar_pos = np.asarray(radar_pos, dtype=float)

    rel = p - radar_pos
    r_true = np.linalg.norm(rel, axis=1)
    r_true = np.maximum(r_true, 1e-6)

    az_true = np.arctan2(rel[:, 1], rel[:, 0])
    el_true = np.arcsin(rel[:, 2] / r_true)
    vr_true = np.sum(v * rel, axis=1) / r_true

    r_meas = r_true + rng.normal(0, std_r, size=r_true.shape)
    az_meas_deg = np.rad2deg(az_true) + rng.normal(0, std_az, size=r_true.shape)
    el_meas_deg = np.rad2deg(el_true) + rng.normal(0, std_el, size=r_true.shape)
    vr_meas = vr_true + rng.normal(0, std_vr, size=r_true.shape)

    meas_polar = np.column_stack([r_meas, az_meas_deg, el_meas_deg, vr_meas]).astype(np.float32)

    az_m = np.deg2rad(az_meas_deg)
    el_m = np.deg2rad(el_meas_deg)
    x = r_meas * np.cos(el_m) * np.cos(az_m) + radar_pos[0]
    y = r_meas * np.cos(el_m) * np.sin(az_m) + radar_pos[1]
    z = r_meas * np.sin(el_m) + radar_pos[2]
    meas_xyz = np.column_stack([x, y, z]).astype(np.float32)

    return meas_polar, meas_xyz


def generate_mixed_fixed_1(T=360, dt=1.0, radar_pos=(0, 0, 0)):
    # 段边界
    b1, b2, b3 = 90, 170, 280

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)

    # 初始状态（较远，避免近距离数值问题）
    p[0] = np.array([4200.0, -1400.0, 700.0], dtype=float)

    heading = np.deg2rad(35.0)
    speed = 10.0

    # 机动参数
    acc2 = 0.35            # m/s^2 强加速
    omega3 = np.deg2rad(9) # rad/s 强急转
    dec4 = -0.45           # m/s^2 强减速

    for t in range(T):
        if t < b1:
            # CV
            vz = 0.6 + 0.8 * np.sin(2 * np.pi * t / 40.0)
        elif t < b2:
            # CA 加速（速度模长快速增大）
            speed = speed + acc2 * dt
            speed = min(speed, 19.0)
            vz = 2.8 + 1.2 * np.sin(2 * np.pi * t / 22.0)
            heading = heading + np.deg2rad(0.6) * np.sin(2 * np.pi * t / 18.0)
        elif t < b3:
            # CT 急转：heading 连续快速变化
            heading = heading + omega3 * dt
            # 同时轻微减速/加速扰动
            speed = np.clip(speed + 0.15 * np.sin(2 * np.pi * t / 16.0), 10.0, 19.5)
            vz = 1.2 + 1.6 * np.sin(2 * np.pi * t / 28.0)
        else:
            # 强减速 + 俯冲
            speed = max(5.0, speed + dec4 * dt)
            heading = heading + np.deg2rad(0.9) * np.sin(2 * np.pi * t / 14.0)
            vz = -3.2 + 1.4 * np.sin(2 * np.pi * t / 18.0)

        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        v[t] = np.array([vx, vy, vz], dtype=float)

        if t > 0:
            p[t] = p[t - 1] + v[t] * dt

        # 高度约束
        p[t, 2] = np.clip(p[t, 2], 80.0, 3000.0)

    a = np.gradient(v, dt, axis=0)

    rng = np.random.RandomState(201)
    meas_polar, meas_xyz = _polar_measure_fixed(p, v, radar_pos, rng)

    return {
        'positions': p.astype(np.float32),
        'velocities': v.astype(np.float32),
        'accelerations': a.astype(np.float32),
        'measurements_polar': meas_polar,
        'meas_xyz': meas_xyz,
        'radar_pos': np.asarray(radar_pos, dtype=np.float32),
        'trajectory_type': 'mixed_fixed_1',
        'dt': dt,
    }


if __name__ == '__main__':
    d = generate_mixed_fixed_1()
    p = d['positions']
    m = d['meas_xyz']

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'k-', linewidth=2, label='True')
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], s=4, alpha=0.35, c='red', label='Meas')
    ax.set_title('Mixed Fixed #1 (CV->CA->CT->Decel/Dive)')
    ax.legend()
    plt.show()
