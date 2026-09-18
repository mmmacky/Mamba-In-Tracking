import numpy as np
import matplotlib.pyplot as plt

""" 
强机动：高过载“蛇形”急转 + 高度机动的确定性轨迹（带固定量测）

设计目标：
- 相比 data/traj.py 的 'turn'/'mixed'，这里让角速度 omega 分段突变（急转、反向急转）、并叠加垂直方向的强机动（vz 与 az/elev 都有显著变化）。
- 输出字段与 fixed_traj 其它文件一致：positions/velocities/accelerations/measurements_polar/meas_xyz/radar_pos/trajectory_type/dt
- 量测噪声使用固定 RandomState，确保每次运行测量一致（便于复现实验/对比）。

量测格式：measurements_polar = [r, az_deg, el_deg, vr_meas]
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


def generate_agile_turn_fixed(T=360, dt=1.0, radar_pos=(0, 0, 0)):
    # 轨迹参数（人为设计的强机动分段）
    # 段1：高速直线进入
    # 段2：强正向急转（大 omega）
    # 段3：强反向急转（omega 反号更大）
    # 段4：再次正向急转 + 强爬升/俯冲耦合
    b1, b2, b3 = 90, 180, 270

    v0 = 16.0  # m/s，较高
    heading = np.deg2rad(20.0)

    # 角速度(rad/s)：这里取比 data/traj.py 的 turn 更大，造成更强机动
    omega1 = 0.0
    omega2 = np.deg2rad(6.0)     # 急转
    omega3 = -np.deg2rad(10.0)   # 反向更急
    omega4 = np.deg2rad(8.0)

    # 垂直强机动：分段给 vz（并叠加一定正弦扰动）
    vz1, vz2, vz3, vz4 = 0.5, 3.0, -2.5, 4.0

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)

    # 初始位置选在较远处，避免距离太小
    p[0] = np.array([3000.0, -1500.0, 600.0], dtype=float)

    for t in range(T):
        if t < b1:
            omega = omega1
            vz = vz1
        elif t < b2:
            omega = omega2
            vz = vz2
        elif t < b3:
            omega = omega3
            vz = vz3
        else:
            omega = omega4
            vz = vz4

        # 叠加强机动的“波动”
        vz = vz + 1.8 * np.sin(2 * np.pi * t / 30.0)

        # 当前速度（水平恒定模长 + heading 变化）
        vx = v0 * np.cos(heading)
        vy = v0 * np.sin(heading)
        v[t] = np.array([vx, vy, vz], dtype=float)

        # 欧拉积分更新位置（用 v[t]）
        if t > 0:
            p[t] = p[t - 1] + v[t] * dt

        # 更新航向（强机动的核心：omega 分段突变）
        heading = heading + omega * dt

        # 高度简单约束
        p[t, 2] = np.clip(p[t, 2], 50.0, 2800.0)

    a = np.gradient(v, dt, axis=0)

    rng = np.random.RandomState(101)
    measurements_polar, meas_xyz = _polar_measure_with_fixed_noise(
        p, v, radar_pos=np.asarray(radar_pos, dtype=float), rng=rng,
        std_r=10.0, std_az=0.8, std_el=0.8, std_vr=1.5
    )

    info = {
        'positions': p.astype(np.float32),
        'velocities': v.astype(np.float32),
        'accelerations': a.astype(np.float32),
        'measurements_polar': measurements_polar,
        'meas_xyz': meas_xyz,
        'radar_pos': np.asarray(radar_pos, dtype=np.float32),
        'trajectory_type': 'agile_turn_fixed',
        'dt': dt
    }
    return info


if __name__ == '__main__':
    data = generate_agile_turn_fixed()
    p = data['positions']
    m = data['meas_xyz']

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'k-', linewidth=2, label='True')
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], c='red', s=4, alpha=0.35, label='Meas')
    ax.set_title('Agile Turn Fixed Trajectory')
    ax.legend()
    plt.show()
