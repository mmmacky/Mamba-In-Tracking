import numpy as np
import matplotlib.pyplot as plt

"""
固定 mixed 轨迹 #3：双向急转（S-turn）+ 强横向蛇形 + 高度“台阶式”变化

段设计：
- 段1：CV + 轻蛇形
- 段2：强 CT 正向急转
- 段3：强 CT 反向急转（S形）
- 段4：强横向蛇形 + 高度阶跃爬升/俯冲耦合

量测固定：RandomState(seed=203)
"""


def _polar_measure_fixed(p, v, radar_pos, rng, std_r=9.0, std_az=0.8, std_el=0.8, std_vr=1.6):
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


def generate_mixed_fixed_3(T=380, dt=1.0, radar_pos=(0, 0, 0)):
    b1, b2, b3 = 110, 210, 290

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)

    p[0] = np.array([3000.0, 3200.0, 600.0], dtype=float)

    speed = 14.0
    heading = np.deg2rad(-110.0)

    omega2 = np.deg2rad(8.0)
    omega3 = -np.deg2rad(11.0)

    for t in range(T):
        if t < b1:
            # CV + 轻蛇形
            heading = heading + np.deg2rad(1.5) * np.sin(2 * np.pi * t / 26.0)
            vz = 0.6 + 0.9 * np.sin(2 * np.pi * t / 33.0)
        elif t < b2:
            # 强急转（正向）
            heading = heading + omega2 * dt
            speed = np.clip(speed + 0.10 * np.sin(2 * np.pi * t / 14.0), 10.0, 18.0)
            vz = 2.0 + 1.8 * np.sin(2 * np.pi * t / 20.0)
        elif t < b3:
            # 强急转（反向）形成 S-turn
            heading = heading + omega3 * dt
            speed = np.clip(speed + 0.12 * np.sin(2 * np.pi * t / 12.0), 9.0, 18.5)
            vz = -1.8 + 1.6 * np.sin(2 * np.pi * t / 18.0)
        else:
            # 强横向蛇形 + 高度台阶变化
            heading = heading + np.deg2rad(22.0) * np.sin(2 * np.pi * t / 18.0)

            # 高度“台阶”：每 30 帧换一次基线
            step = ((t - b3) // 30) % 3
            if step == 0:
                base_vz = 3.8
            elif step == 1:
                base_vz = -3.0
            else:
                base_vz = 2.6
            vz = base_vz + 1.2 * np.sin(2 * np.pi * t / 16.0)

            # 速度也做轻脉冲
            speed = np.clip(speed + 0.28 * np.sin(2 * np.pi * t / 10.0), 8.0, 19.0)

        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        v[t] = np.array([vx, vy, vz], dtype=float)

        if t > 0:
            p[t] = p[t - 1] + v[t] * dt

        p[t, 2] = np.clip(p[t, 2], 80.0, 3000.0)

    a = np.gradient(v, dt, axis=0)

    rng = np.random.RandomState(203)
    meas_polar, meas_xyz = _polar_measure_fixed(p, v, radar_pos, rng)

    return {
        'positions': p.astype(np.float32),
        'velocities': v.astype(np.float32),
        'accelerations': a.astype(np.float32),
        'measurements_polar': meas_polar,
        'meas_xyz': meas_xyz,
        'radar_pos': np.asarray(radar_pos, dtype=np.float32),
        'trajectory_type': 'mixed_fixed_3',
        'dt': dt,
    }


if __name__ == '__main__':
    d = generate_mixed_fixed_3()
    p = d['positions']
    m = d['meas_xyz']

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'm-', linewidth=2, label='True')
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], s=4, alpha=0.35, c='orange', label='Meas')
    ax.set_title('Mixed Fixed #3 (S-turn + Strong Weave + Step Vz)')
    ax.legend()
    plt.show()
