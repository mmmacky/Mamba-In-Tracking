import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
确定性类螺旋轨迹（带固定量测）
- 轨迹：三段式螺旋（内收上升 -> 恒定盘旋 -> 外扩下降）
- 量测：距离、方位、俯仰、多普勒（带固定种子的高斯噪声）
"""

def generate_spiral_fixed(T=400, dt=1.0, radar_pos=(0, 0, 0)):
    # --- 1. 生成真实轨迹 ---
    center = np.array([0.0, 0.0, 0.0])
    R0 = 4000.0
    
    # 状态参数
    omega1, omega2, omega3 = np.deg2rad(2.0), np.deg2rad(2.0), np.deg2rad(2.0)
    vz1, vz2, vz3 = 2.0, 1.0, -0.5
    T1, T2 = 150, 300

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)
    a = np.zeros((T, 3), dtype=float) # 需要数值微分计算

    theta = np.deg2rad(30.0)
    z = 500.0

    for t in range(T):
        if t < T1:
            frac = t / T1
            R = R0 * (1.0 - 0.4 * frac)
            omega = omega1
            vz = vz1
            dR = -R0 * 0.4 / (T1 * dt) # 半径变化率
        elif t < T2:
            R = R0 * 0.6
            omega = omega2
            vz = vz2
            dR = 0.0
        else:
            frac = (t - T2) / (T - T2)
            R = R0 * (0.6 + 0.3 * frac)
            omega = omega3
            vz = vz3
            dR = R0 * 0.3 / ((T - T2) * dt)

        # 位置
        x = center[0] + R * np.cos(theta)
        y = center[1] + R * np.sin(theta)
        p[t] = np.array([x, y, z])

        # 速度 (推导: x' = R'cos - R*sin*omega)
        vx = dR * np.cos(theta) - R * np.sin(theta) * omega
        vy = dR * np.sin(theta) + R * np.cos(theta) * omega
        v[t] = np.array([vx, vy, vz])

        # 更新状态
        theta += omega * dt
        z += vz * dt

    # 计算加速度 (数值微分)
    a = np.gradient(v, dt, axis=0)

    # --- 2. 生成固定量测 ---
    rng = np.random.RandomState(44) # 独立固定种子
    
    radar_pos = np.array(radar_pos)
    std_r = 5.0
    std_az = 0.5
    std_el = 0.5
    std_vr = 1.0

    rel_p = p - radar_pos
    r_true = np.linalg.norm(rel_p, axis=1)
    r_true = np.maximum(r_true, 1e-6)
    
    az_true = np.arctan2(rel_p[:, 1], rel_p[:, 0])
    el_true = np.arcsin(rel_p[:, 2] / r_true)
    vr_true = np.sum(v * rel_p, axis=1) / r_true

    r_meas = r_true + rng.normal(0, std_r, T)
    az_meas_deg = np.rad2deg(az_true) + rng.normal(0, std_az, T)
    el_meas_deg = np.rad2deg(el_true) + rng.normal(0, std_el, T)
    vr_meas = vr_true + rng.normal(0, std_vr, T)

    measurements_polar = np.column_stack([r_meas, az_meas_deg, el_meas_deg, vr_meas]).astype(np.float32)

    # 还原
    az_rad_m = np.deg2rad(az_meas_deg)
    el_rad_m = np.deg2rad(el_meas_deg)
    meas_x = r_meas * np.cos(el_rad_m) * np.cos(az_rad_m) + radar_pos[0]
    meas_y = r_meas * np.cos(el_rad_m) * np.sin(az_rad_m) + radar_pos[1]
    meas_z = r_meas * np.sin(el_rad_m) + radar_pos[2]
    meas_xyz = np.column_stack([meas_x, meas_y, meas_z]).astype(np.float32)

    # --- 3. 打包 ---
    info = {
        'positions': p.astype(np.float32),
        'velocities': v.astype(np.float32),
        'accelerations': a.astype(np.float32),
        'measurements_polar': measurements_polar,
        'meas_xyz': meas_xyz,
        'radar_pos': radar_pos.astype(np.float32),
        'trajectory_type': 'spiral_fixed',
        'dt': dt
    }
    return info

if __name__ == '__main__':
    data = generate_spiral_fixed()
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = data['positions']
    m = data['meas_xyz']
    ax.plot(p[:,0], p[:,1], p[:,2], 'm-', linewidth=2, label='True')
    ax.scatter(m[:,0], m[:,1], m[:,2], c='blue', s=5, alpha=0.5, label='Meas')
    ax.legend()
    ax.set_title('Spiral Fixed Trajectory')
    plt.show()