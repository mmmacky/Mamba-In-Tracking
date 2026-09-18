import numpy as np
import matplotlib.pyplot as plt

"""
强机动：Stop-Go（近似悬停/急停/再启动）+ 急转撤离 的确定性轨迹（带固定量测）

设计目标：
- 对滤波器最“难”的情况之一是：速度/加速度突变、短时接近静止、然后迅速改变方向重新加速。
- 这里做 4 段：
  1) 高速接近（较大负径向速度）
  2) 急减速到近零（stop）并伴随轻微漂移（减少完全静止的数值问题）
  3) 原地附近快速转向（heading 快速变化）
  4) 以新的方向强加速撤离（go + turn）

注意：输出同 fixed_traj 其它文件。
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


def generate_agile_stopgo_fixed(T=340, dt=1.0, radar_pos=(0, 0, 0)):
    b1, b2, b3 = 110, 180, 240

    # 段1：高速接近
    heading1 = np.deg2rad(210.0)  # 往雷达方向的某个分量
    speed1 = 17.0
    vz1 = 0.8

    # 段2：急刹车到近零
    speed2_start = speed1
    speed2_end = 0.8
    vz2 = -0.5

    # 段3：近似“原地”快速转向
    speed3 = 1.2
    omega3 = np.deg2rad(14.0)  # 非常快的转向
    vz3 = 0.2

    # 段4：新方向强加速撤离
    heading4 = np.deg2rad(60.0)
    speed4_start = 4.0
    speed4_end = 16.0
    vz4 = 3.0

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)

    p[0] = np.array([-3200.0, 2800.0, 700.0], dtype=float)

    heading = heading1

    for t in range(T):
        if t < b1:
            speed = speed1
            vz = vz1
        elif t < b2:
            frac = (t - b1) / max(1, (b2 - b1))
            speed = speed2_start + (speed2_end - speed2_start) * frac
            vz = vz2
        elif t < b3:
            speed = speed3
            vz = vz3
            heading = heading + omega3 * dt
        else:
            frac = (t - b3) / max(1, (T - b3))
            speed = speed4_start + (speed4_end - speed4_start) * frac
            vz = vz4
            # 逐渐过渡到 heading4（避免瞬跳太不真实，但仍然是强机动）
            heading = heading + 0.12 * (heading4 - heading)

        # stop 段加入小漂移
        drift = 0.0
        if b1 <= t < b3:
            drift = 0.4 * np.sin(2 * np.pi * t / 18.0)

        vx = (speed + drift) * np.cos(heading)
        vy = (speed + drift) * np.sin(heading)
        v[t] = np.array([vx, vy, vz], dtype=float)

        if t > 0:
            p[t] = p[t - 1] + v[t] * dt

        p[t, 2] = np.clip(p[t, 2], 60.0, 2600.0)

    a = np.gradient(v, dt, axis=0)

    rng = np.random.RandomState(103)
    measurements_polar, meas_xyz = _polar_measure_with_fixed_noise(
        p, v, radar_pos=np.asarray(radar_pos, dtype=float), rng=rng,
        std_r=10.0, std_az=0.9, std_el=0.9, std_vr=1.6
    )

    info = {
        'positions': p.astype(np.float32),
        'velocities': v.astype(np.float32),
        'accelerations': a.astype(np.float32),
        'measurements_polar': measurements_polar,
        'meas_xyz': meas_xyz,
        'radar_pos': np.asarray(radar_pos, dtype=np.float32),
        'trajectory_type': 'agile_stopgo_fixed',
        'dt': dt
    }
    return info


if __name__ == '__main__':
    data = generate_agile_stopgo_fixed()
    p = data['positions']
    m = data['meas_xyz']

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', linewidth=2, label='True')
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], c='purple', s=4, alpha=0.35, label='Meas')
    ax.set_title('Agile Stop-Go Fixed Trajectory')
    ax.legend()
    plt.show()
