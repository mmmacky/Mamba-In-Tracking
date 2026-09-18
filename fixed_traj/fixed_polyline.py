import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
确定性折线轨迹（带固定量测）
- 轨迹：三段折线（匀速 -> 加速 -> 减速爬升）
- 量测：距离、方位、俯仰、多普勒（带固定种子的高斯噪声）
"""

def generate_polyline_fixed(T=300, dt=1.0, radar_pos=(0, 0, 0)):
    # --- 1. 生成真实轨迹 (Ground Truth) ---
    # 关键拐点
    p0 = np.array([1000.0, -1500.0, 200.0])
    p1 = np.array([3000.0, -500.0, 250.0])
    p2 = np.array([4500.0, 1000.0, 300.0])
    p3 = np.array([5200.0, 2500.0, 450.0])

    T1, T2 = 120, 220
    
    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)
    a = np.zeros((T, 3), dtype=float) # 加速度

    # 计算各段速度
    v1 = (p1 - p0) / (T1 * dt)
    v2 = (p2 - p1) / ((T2 - T1) * dt)
    v3 = (p3 - p2) / ((T - T2) * dt)

    p[0] = p0
    v[0] = v1

    for t in range(1, T):
        if t < T1:
            v[t] = v1
            # 在拐点处会有瞬时加速度，这里简化为0
            a[t] = 0.0 
        elif t < T2:
            v[t] = v2
            # 简单处理：拐点时刻给一个脉冲加速度（可选），此处保持为0方便处理
            a[t] = 0.0
        else:
            v[t] = v3
            a[t] = 0.0
        p[t] = p[t-1] + v[t] * dt

    # --- 2. 生成固定量测 (Fixed Measurements) ---
    # 使用固定种子确保噪声一致
    rng = np.random.RandomState(42)
    
    # 雷达参数
    radar_pos = np.array(radar_pos)
    std_r = 5.0      # 距离噪声 (m)
    std_az = 0.5      # 角度噪声 (deg)
    std_el = 0.5
    std_vr = 1.0      # 速度噪声 (m/s)

    # 计算真值相对量
    rel_p = p - radar_pos
    r_true = np.linalg.norm(rel_p, axis=1)
    r_true = np.maximum(r_true, 1e-6) # 避免除零
    
    # 角度 (rad)
    az_true = np.arctan2(rel_p[:, 1], rel_p[:, 0])
    el_true = np.arcsin(rel_p[:, 2] / r_true)
    
    # 径向速度
    vr_true = np.sum(v * rel_p, axis=1) / r_true

    # 添加噪声
    noise_r = rng.normal(0, std_r, T)
    noise_az = rng.normal(0, std_az, T)
    noise_el = rng.normal(0, std_el, T)
    noise_vr = rng.normal(0, std_vr, T)

    r_meas = r_true + noise_r
    az_meas_deg = np.rad2deg(az_true) + noise_az
    el_meas_deg = np.rad2deg(el_true) + noise_el
    vr_meas = vr_true + noise_vr

    # 组合量测矩阵 (T, 4): [r, az, el, vr]
    measurements_polar = np.column_stack([r_meas, az_meas_deg, el_meas_deg, vr_meas]).astype(np.float32)

    # 量测转笛卡尔 (仅用于可视化验证)
    az_rad_m = np.deg2rad(az_meas_deg)
    el_rad_m = np.deg2rad(el_meas_deg)
    meas_x = r_meas * np.cos(el_rad_m) * np.cos(az_rad_m) + radar_pos[0]
    meas_y = r_meas * np.cos(el_rad_m) * np.sin(az_rad_m) + radar_pos[1]
    meas_z = r_meas * np.sin(el_rad_m) + radar_pos[2]
    meas_xyz = np.column_stack([meas_x, meas_y, meas_z]).astype(np.float32)

    # --- 3. 打包返回 ---
    info = {
        'positions': p.astype(np.float32),
        'velocities': v.astype(np.float32),
        'accelerations': a.astype(np.float32),
        'measurements_polar': measurements_polar,
        'meas_xyz': meas_xyz,
        'radar_pos': radar_pos.astype(np.float32),
        'trajectory_type': 'polyline_fixed',
        'dt': dt
    }
    return info

if __name__ == '__main__':
    data = generate_polyline_fixed()
    
    # 简易可视化
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = data['positions']
    m = data['meas_xyz']
    
    ax.plot(p[:,0], p[:,1], p[:,2], 'b-', linewidth=2, label='True')
    ax.scatter(m[:,0], m[:,1], m[:,2], c='orange', s=5, alpha=0.5, label='Meas (Sampled)')
    ax.legend()
    ax.set_title('Polyline Fixed Trajectory')
    plt.show()