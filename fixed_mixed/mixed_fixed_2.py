import numpy as np
import matplotlib.pyplot as plt

"""
固定 mixed 轨迹 #2：Stop-Go + 反向急转 + 强爬升/俯冲

段设计（强机动）：
- 段1：高速接近（使 vr 为显著负值）
- 段2：强制急停到近零并短暂停留（stop）
- 段3：低速状态下快速转向（大 omega）
- 段4：强加速撤离（go）并爬升

量测固定：RandomState(seed=202)
"""


def _polar_measure_fixed(p, v, radar_pos, rng, std_r=10.0, std_az=0.9, std_el=0.9, std_vr=1.8):
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


def generate_mixed_fixed_2(T=340, dt=1.0, radar_pos=(0, 0, 0)):
    b1, b2, b3 = 110, 170, 240

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)

    p[0] = np.array([-3800.0, 2600.0, 750.0], dtype=float)

    # 段1：高速接近
    heading = np.deg2rad(225.0)
    speed = 18.0

    # 段2：急停
    speed_stop = 0.9

    # 段3：快速转向（低速下转向更“刁钻”）
    omega = np.deg2rad(13.0)

    # 段4：强加速撤离
    heading4 = np.deg2rad(55.0)

    for t in range(T):
        if t < b1:
            vz = 0.8 + 0.5 * np.sin(2 * np.pi * t / 30.0)
        elif t < b2:
            frac = (t - b1) / max(1, (b2 - b1))
            speed = speed + (speed_stop - speed) * frac  # 快速降到接近0
            vz = -1.2 + 0.8 * np.sin(2 * np.pi * t / 18.0)
            # stop 段加一点抖动，避免完全静止
            drift = 0.6 * np.sin(2 * np.pi * t / 10.0)
            speed = max(0.3, speed + drift * 0.05)
        elif t < b3:
            speed = speed_stop
            heading = heading + omega * dt
            vz = 0.2 + 0.7 * np.sin(2 * np.pi * t / 16.0)
        else:
            frac = (t - b3) / max(1, (T - b3))
            speed = np.clip(speed_stop + (17.0 - speed_stop) * frac, 0.5, 19.0)
            # 逐渐收敛到新航向，但仍保持强变化
            heading = heading + 0.14 * (heading4 - heading)
            vz = 3.8 + 1.2 * np.sin(2 * np.pi * t / 22.0)

        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        v[t] = np.array([vx, vy, vz], dtype=float)

        if t > 0:
            p[t] = p[t - 1] + v[t] * dt

        p[t, 2] = np.clip(p[t, 2], 60.0, 2800.0)

    a = np.gradient(v, dt, axis=0)

    rng = np.random.RandomState(202)
    meas_polar, meas_xyz = _polar_measure_fixed(p, v, radar_pos, rng)

    return {
        'positions': p.astype(np.float32),
        'velocities': v.astype(np.float32),
        'accelerations': a.astype(np.float32),
        'measurements_polar': meas_polar,
        'meas_xyz': meas_xyz,
        'radar_pos': np.asarray(radar_pos, dtype=np.float32),
        'trajectory_type': 'mixed_fixed_2',
        'dt': dt,
    }


if __name__ == '__main__':
    d = generate_mixed_fixed_2()
    p = d['positions']
    m = d['meas_xyz']

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', linewidth=2, label='True')
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], s=4, alpha=0.35, c='purple', label='Meas')
    ax.set_title('Mixed Fixed #2 (Stop-Go + Fast Turn)')
    ax.legend()
    plt.show()
