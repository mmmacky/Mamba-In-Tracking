import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_hard_maneuver_trajectory(T=200, dt=1, radar_pos=(0.0, 0.0, 0.0)):
    """
    生成一个具有高机动性、非线性特征的确定性轨迹，用于测试跟踪算法的极限性能。

    特点：
    1. 确定性：使用固定种子和公式，保证每次生成数据一致，便于横向对比算法。
    2. 高机动：包含急剧的加减速、变加速盘旋和高频战术规避动作。
    3. 难跟踪：状态转移高度非线性，加速度导数（加加速度 Jerk）不为零且变化剧烈。
    
    参数:
        T: 时间步数
        dt: 采样间隔 (秒)
        radar_pos: 雷达位置 (x, y, z)
    
    返回:
        字典 (格式兼容 traj.py)
    """
    
    # 1. 初始化确定性随机数生成器 (用于噪声)
    rng = np.random.RandomState(42) 
    
    # 2. 定义时间轴
    t = np.linspace(0, T*dt, T)
    
    # 3. 构造高机动轨迹 (Parametric Equations)
    # 这里的轨迹方程组合了：
    # - 基础运动：大范围的 figure-8 (8字形) 盘旋
    # - 战术规避：叠加在 XYZ 轴上的不同频率的高频正弦波 (模拟蛇形机动/颤振)
    # - 垂直机动：非线性的俯冲与拉起
    
    # 基础频率
    f_main = 0.03  # 宏观机动频率
    f_evade = 0.2  # 规避机动频率
    
    # X轴: 主要是正弦运动，叠加快速扰动
    # 3000m 初始距离 + 2000m 幅度的摆动 + 150m 幅度的快速抖动
    x_true = 3000 + 2500 * np.sin(f_main * t) + 150 * np.sin(f_evade * t)
    
    # Y轴: 余弦运动 (形成圆/椭圆)，但频率随时间微变 (导致轨迹不闭合，且难以预测)
    # 引入 t**1.1 让周期逐渐发生变化
    y_true = 2000 * np.cos(f_main * t**1.05) + 200 * np.cos(f_evade * 1.5 * t)
    
    # Z轴: 模拟“俯冲-拉起-再俯冲”的剧烈高度变化
    # 1500m 基准 + 大幅度低频波动 + 线性爬升趋势
    z_true = 1500 + 800 * np.sin(0.05 * t) + 5 * t + 100 * np.cos(f_evade * t)
    
    # 组合位置
    p = np.stack([x_true, y_true, z_true], axis=1) # (T, 3)
    
    # 4. 计算动力学参数 (速度 v 和 加速度 a)
    # 使用 np.gradient 计算数值微分，保证运动学一致性
    v = np.zeros_like(p)
    a = np.zeros_like(p)
    
    for i in range(3):
        v[:, i] = np.gradient(p[:, i], dt)
        a[:, i] = np.gradient(v[:, i], dt)
        
    # 限制Z轴不钻地 (虽然上述方程基本保证 z>0，但做个保险)
    p[:, 2] = np.maximum(p[:, 2], 10.0)
    
    # 5. 生成雷达量测数据 (模拟真实雷达探测)
    # ----------------------------------------
    # 噪声标准差设置 (固定值，对应中等精度雷达)
    std_r = 10.0       # 距离噪声 (m)
    std_az = 0.5       # 方位角噪声 (deg)
    std_el = 0.5       # 俯仰角噪声 (deg)
    std_vr = 1.0       # 多普勒速度噪声 (m/s)
    
    # 计算相对于雷达的几何关系
    radar_pos = np.array(radar_pos)
    p_rel = p - radar_pos
    
    # 距离 r
    r_true = np.linalg.norm(p_rel, axis=1)
    
    # 方位角 azimuth (rad) -> atan2(y, x)
    az_true = np.arctan2(p_rel[:, 1], p_rel[:, 0])
    
    # 俯仰角 elevation (rad) -> arcsin(z / r)
    el_true = np.arcsin(p_rel[:, 2] / (r_true + 1e-6))
    
    # 径向速度 (多普勒) vr = (v . p_rel) / r
    # 注意：这是径向速度的真值
    vr_true = np.sum(v * p_rel, axis=1) / (r_true + 1e-6)
    
    # 添加高斯白噪声
    noise_r = rng.normal(0, std_r, T)
    noise_az = rng.normal(0, std_az, T) # 度
    noise_el = rng.normal(0, std_el, T) # 度
    noise_vr = rng.normal(0, std_vr, T)
    
    r_meas = r_true + noise_r
    # 注意：内部计算用弧度，输出量测通常用度 (依照 traj.py 的习惯)
    az_meas_deg = np.rad2deg(az_true) + noise_az
    el_meas_deg = np.rad2deg(el_true) + noise_el
    vr_meas = vr_true + noise_vr
    
    # 裁剪无效值
    r_meas = np.maximum(r_meas, 1.0)
    
    # 组合量测矩阵: [r, az_deg, el_deg, vr]
    measurements_polar = np.column_stack([r_meas, az_meas_deg, el_meas_deg, vr_meas]).astype(np.float32)
    
    # 6. 计算量测对应的笛卡尔坐标 (仅用于可视化或转换算法)
    az_rad_m = np.deg2rad(az_meas_deg)
    el_rad_m = np.deg2rad(el_meas_deg)
    
    x_m = r_meas * np.cos(el_rad_m) * np.cos(az_rad_m) + radar_pos[0]
    y_m = r_meas * np.cos(el_rad_m) * np.sin(az_rad_m) + radar_pos[1]
    z_m = r_meas * np.sin(el_rad_m) + radar_pos[2]
    
    meas_xyz = np.column_stack([x_m, y_m, z_m]).astype(np.float32)

    # 7. 打包返回信息
    info = {
        'positions': p.astype(np.float32),          # 真值位置 (T,3)
        'velocities': v.astype(np.float32),         # 真值速度 (T,3)
        'accelerations': a.astype(np.float32),      # 真值加速度 (T,3)
        'measurements_polar': measurements_polar,   # 量测 [r, az, el, vr]
        'meas_xyz': meas_xyz,                       # 量测转笛卡尔 (T,3)
        'dt': dt,
        'description': 'Deterministic High-Maneuver Trajectory (Snake/Weave + Dive)',
        'noise_params': {'std_r': std_r, 'std_az': std_az, 'std_el': std_el, 'std_vr': std_vr}
    }
    
    return info

def visualize_trajectory(data):
    """
    可视化生成的轨迹、速度变化及量测点
    """
    p = data['positions']
    m = data['meas_xyz']
    v = data['velocities']
    
    fig = plt.figure(figsize=(12, 10))
    
    # 1. 3D 轨迹图
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', linewidth=2, label='True Trajectory')
    # 稀疏采样画量测点，避免图太乱
    ax.scatter(m[::5, 0], m[::5, 1], m[::5, 2], c='r', s=2, alpha=0.5, label='Measurements (sampled)')
    
    # 画起点和终点
    ax.scatter(p[0,0], p[0,1], p[0,2], c='g', s=50, marker='^', label='Start')
    ax.scatter(p[-1,0], p[-1,1], p[-1,2], c='k', s=50, marker='x', label='End')
    
    ax.set_title('3D High-Maneuver Trajectory')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    
    # 2. 速度大小变化
    ax2 = fig.add_subplot(222)
    speed = np.linalg.norm(v, axis=1)
    ax2.plot(speed, 'k-')
    ax2.set_title('Target Speed Profile')
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_xlabel('Time Step')
    ax2.grid(True)
    
    # 3. XY 平面投影 (展示转弯和蛇形机动)
    ax3 = fig.add_subplot(223)
    ax3.plot(p[:, 0], p[:, 1], 'b-')
    ax3.set_title('Top View (X-Y) - Evasive Maneuvers')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.grid(True)
    ax3.axis('equal')
    
    # 4. 径向速度 (多普勒) 对比
    ax4 = fig.add_subplot(224)
    vr_meas = data['measurements_polar'][:, 3]
    # 计算真值 vr 用于对比
    p_rel = p  # 假设雷达在原点
    r = np.linalg.norm(p_rel, axis=1)
    vr_true = np.sum(v * p_rel, axis=1) / r
    
    ax4.plot(vr_meas, 'r.', markersize=2, alpha=0.3, label='Doppler Meas')
    ax4.plot(vr_true, 'b-', linewidth=1, label='True Doppler')
    ax4.set_title('Doppler Velocity Profile')
    ax4.set_ylabel('Radial Velocity (m/s)')
    ax4.set_xlabel('Time Step')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 生成 200秒的数据，采样率 2Hz (dt=0.5s) -> 400个点
    traj_data = generate_hard_maneuver_trajectory(T=400, dt=0.5)
    
    print(f"Generated Trajectory Description: {traj_data['description']}")
    print(f"Position Shape: {traj_data['positions'].shape}")
    print(f"Measurement Shape: {traj_data['measurements_polar'].shape} (r, az, el, vr)")
    
    # 验证最大加速度 (g)
    acc_mag = np.linalg.norm(traj_data['accelerations'], axis=1)
    max_g = np.max(acc_mag) / 9.8
    print(f"Max G-Force: {max_g:.2f} g")
    
    visualize_trajectory(traj_data)