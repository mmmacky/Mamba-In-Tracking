import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
确定性类抛物线轨迹（带固定量测）
- 轨迹：空中起止的平滑抛物线
- 量测：距离、方位、俯仰、多普勒（带固定种子的高斯噪声）
"""

def generate_parabola_fixed(T=360, dt=1.0, radar_pos=(0, 0, 0)):
    # --- 1. 生成真实轨迹 ---
    p0 = np.array([0.0, -2000.0, 300.0])
    vx = 5.0
    vy = 1.5
    vz0 = 4.0
    g = -0.04

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)
    a = np.zeros((T, 3), dtype=float) # 加速度恒定

    # 初始状态
    p[0] = p0
    v[0] = np.array([vx, vy, vz0])
    acc_vec = np.array([0.0, 0.0, g])
    a[:] = acc_vec

    for t in range(1, T):
        t_s = t * dt
        # 运动学方程
        p[t, 0] = p0[0] + vx * t_s
        p[t, 1] = p0[1] + vy * t_s
        p[t, 2] = p0[2] + vz0 * t_s + 0.5 * g * (t_s ** 2)
        
        v[t, 0] = vx
        v[t, 1] = vy
        v[t, 2] = vz0 + g * t_s

    # --- 2. 生成固定量测 ---
    rng = np.random.RandomState(43) # 使用不同种子以示区别，但固定
    
    radar_pos = np.array(radar_pos)
    std_r = 10.0
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

    # 还原坐标用于验证
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
        'trajectory_type': 'parabola_fixed',
        'dt': dt
    }
    return info

if __name__ == '__main__':
    data = generate_parabola_fixed()
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = data['positions']
    m = data['meas_xyz']
    ax.plot(p[:,0], p[:,1], p[:,2], 'g-', linewidth=2, label='True')
    ax.scatter(m[:,0], m[:,1], m[:,2], c='red', s=5, alpha=0.5, label='Meas')
    ax.legend()
    ax.set_title('Parabola Fixed Trajectory')
    plt.show()