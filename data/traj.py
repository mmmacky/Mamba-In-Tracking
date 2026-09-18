import numpy as np


def generate_trajectory(T=200, trajectory_type='none', dt=1.0,
                        noise_std_r=5.0, noise_std_az=1.0, noise_std_el=1.0,
                        noise_std_vr=0.1, radar_pos=(0.0, 0.0)):
    """
    生成3D无人机轨迹，输出雷达极坐标量测（含多普勒速度）

    参数:
        T: 时间步数 (默认200)
        trajectory_type: 轨迹类型
            - 'uniform': 匀速直线运动 (CV)
            - 'accelerate': 匀加速直线运动 (CA)
            - 'decelerate': 匀减速直线运动 (CA 负加速度)
            - 'turn': 转弯 (CT: coordinated turn)（水平面转弯，允许小幅垂直速度）
            - 'hover': 悬停（小幅随机游走）
            - 'mixed': 混合运动（分段，转弯段用 CT）
        dt: 时间间隔 (秒)
        noise_std_r: 距离量测噪声标准差 (米) - 默认5m
        noise_std_az: 方位角量测噪声标准差 (度) - 默认1°
        noise_std_el: 俯仰角量测噪声标准差 (度) - 默认1°
        noise_std_vr: 多普勒速度量测噪声标准差 (m/s) - 默认0.1m/s
        radar_pos: 雷达位置 (x,y)

    返回:
        字典包含真实轨迹、极坐标量测、多普勒速度量测等
        measurements_polar 列顺序: (r, az_deg, el_deg, vr_meas)
        注意：外部所有角度均以度为单位
    """
    # 运动约束
    R_MIN, R_MAX = 3000.0, 5000.0  # 总距离范围 3-5km
    Z_MAX = 3000.0  # 最高高度
    V_NORMAL_MAX = 10.0  # 正常速度上限
    V_ABSOLUTE_MAX = 20.0  # 绝对速度上限

    radar_pos = np.array([radar_pos[0], radar_pos[1], 0.0], dtype=float)

    # 初始化数组 (T,3)
    a = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)
    p = np.zeros((T, 3), dtype=float)

    # 随机采样初始位置，保证 3-5km 的总体距离且高度在 [0,Z_MAX]
    def sample_initial_pos():
        for _ in range(1000):
            z0 = np.random.uniform(0.0, Z_MAX)
            # 先采样水平距离 rho，然后检查 r = sqrt(rho^2 + z0^2)
            rho = np.random.uniform(0.0, R_MAX)
            theta0 = np.random.uniform(0.0, 2 * np.pi)
            x0 = rho * np.cos(theta0)
            y0 = rho * np.sin(theta0)
            r_tot = np.sqrt(x0**2 + y0**2 + z0**2)
            if R_MIN <= r_tot <= R_MAX:
                return np.array([x0 + radar_pos[0], y0 + radar_pos[1], z0])
        # 兜底：将水平距离设为 R_MIN 并回退
        theta0 = np.random.uniform(0.0, 2 * np.pi)
        z0 = np.random.uniform(0.0, min(Z_MAX, R_MIN - 1.0))
        x0 = R_MIN * np.cos(theta0)
        y0 = R_MIN * np.sin(theta0)
        return np.array([x0 + radar_pos[0], y0 + radar_pos[1], z0])

    p[0] = sample_initial_pos()

    # 根据轨迹类型设置初始速度和加速度策略
    if trajectory_type == 'uniform':
        speed = np.random.uniform(3.0, V_NORMAL_MAX)
        # 随机三维方向，但限制竖直分量较小，确保主要在水平面移动
        elev = np.deg2rad(np.random.uniform(-10, 10))
        az = np.random.uniform(0, 2 * np.pi)
        v[0] = speed * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
        a[:] = 0.0

    elif trajectory_type == 'accelerate':
        v0 = np.random.uniform(1.0, 4.0)
        az = np.random.uniform(0, 2 * np.pi)
        elev = np.deg2rad(np.random.uniform(-10, 10))
        v[0] = v0 * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
        acc_mag = np.random.uniform(0.05, 0.3)
        a[:] = acc_mag * (v[0] / (np.linalg.norm(v[0]) + 1e-10))

    elif trajectory_type == 'decelerate':
        v0 = np.random.uniform(6.0, V_NORMAL_MAX)
        az = np.random.uniform(0, 2 * np.pi)
        elev = np.deg2rad(np.random.uniform(-10, 10))
        v[0] = v0 * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
        dec_mag = np.random.uniform(0.05, 0.3)
        a[:] = -dec_mag * (v[0] / (np.linalg.norm(v[0]) + 1e-10))

    elif trajectory_type == 'turn':
        # 水平面协调转弯 (CT)，保持速度大小近似恒定
        v0 = np.random.uniform(3.0, V_NORMAL_MAX)
        heading = np.random.uniform(0, 2 * np.pi)
        elev = np.deg2rad(np.random.uniform(-10, 10))
        v[0] = v0 * np.array([np.cos(elev) * np.cos(heading), np.cos(elev) * np.sin(heading), np.sin(elev)])
        # 初始角速度 (rad/s)
        omega = np.random.uniform(np.deg2rad(0.5), np.deg2rad(3.0)) * np.random.choice([-1, 1])
        # 将水平CT对应的加速度写入 a[t]
        for t in range(T):
            v_dir = np.array([np.cos(heading), np.sin(heading)])
            perp = np.array([-v_dir[1], v_dir[0]])
            # centripetal acc in horizontal plane
            acc_h = v0 * omega * perp
            a[t, 0:2] = acc_h
            a[t, 2] = 0.0
            heading = heading + omega * dt

    elif trajectory_type == 'hover':
        # 初始速度接近 0，之后小幅随机漫步
        v[0] = np.array([0.0, 0.0, 0.0])
        for t in range(T):
            a[t] = np.random.normal(scale=0.02, size=3)  # 小幅随机加速度

    elif trajectory_type == 'mixed':
        # 分段组合前5种模式
        segments = np.random.randint(2, 5)
        seg_len = max(5, T // segments)
        for i in range(segments):
            start = i * seg_len
            end = min(T, (i + 1) * seg_len)
            mode = np.random.choice(['uniform', 'accelerate', 'decelerate', 'turn', 'hover'])
            if mode == 'uniform':
                speed = np.random.uniform(3.0, V_NORMAL_MAX)
                az = np.random.uniform(0, 2 * np.pi)
                elev = np.deg2rad(np.random.uniform(-15, 15))
                if start == 0:
                    v[start] = speed * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
                for t in range(start, end):
                    a[t] = 0.0
            elif mode == 'accelerate':
                if start == 0:
                    az = np.random.uniform(0, 2 * np.pi)
                    elev = np.deg2rad(np.random.uniform(-15, 15))
                    v[start] = np.random.uniform(1.0, 4.0) * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
                acc_mag = np.random.uniform(0.05, 0.3)
                for t in range(start, end):
                    a[t] = acc_mag * (v[start] / (np.linalg.norm(v[start]) + 1e-10))
            elif mode == 'decelerate':
                if start == 0:
                    az = np.random.uniform(0, 2 * np.pi)
                    elev = np.deg2rad(np.random.uniform(-15, 15))
                    v[start] = np.random.uniform(6.0, V_NORMAL_MAX) * np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
                dec_mag = np.random.uniform(0.05, 0.3)
                for t in range(start, end):
                    a[t] = -dec_mag * (v[start] / (np.linalg.norm(v[start]) + 1e-10))
            elif mode == 'turn':
                # horizontal CT for the segment
                v0 = np.linalg.norm(v[start-1]) if start > 0 and np.linalg.norm(v[start-1]) > 0 else np.random.uniform(3.0, V_NORMAL_MAX)
                heading = np.arctan2(v[start-1,1], v[start-1,0]) if start > 0 and np.linalg.norm(v[start-1])>0 else np.random.uniform(0, 2*np.pi)
                omega = np.random.uniform(np.deg2rad(0.5), np.deg2rad(3.0)) * np.random.choice([-1, 1])
                for t in range(start, end):
                    v_dir = np.array([np.cos(heading), np.sin(heading)])
                    perp = np.array([-v_dir[1], v_dir[0]])
                    a[t, 0:2] = v0 * omega * perp
                    a[t, 2] = 0.0
                    heading = heading + omega * dt
            elif mode == 'hover':
                for t in range(start, end):
                    a[t] = np.random.normal(scale=0.02, size=3)

    else:
        # 默认 fallback：保持静止
        v[0] = np.array([0.0, 0.0, 0.0])
        a[:] = 0.0

    # 运动学积分：位置与速度
    for t in range(1, T):
        # 更新速度
        v[t] = v[t-1] + a[t-1] * dt
        # 限制速度大小
        speed = np.linalg.norm(v[t])
        if speed > V_ABSOLUTE_MAX:
            v[t] = v[t] * (V_ABSOLUTE_MAX / speed)
        # 更新位置
        p_new = p[t-1] + v[t-1] * dt + 0.5 * a[t-1] * dt * dt
        # 检查高度限制
        if p_new[2] < 0:
            p_new[2] = 0.0
            v[t,2] = 0.0
        if p_new[2] > Z_MAX:
            p_new[2] = Z_MAX
            v[t,2] = 0.0
        p[t] = p_new

    # 真实位置转换为相对于雷达的坐标（3D）
    p_relative = p - radar_pos  # (T,3)

    # 计算真值
    r_true = np.linalg.norm(p_relative, axis=1)
    theta_true = np.arctan2(p_relative[:, 1], p_relative[:, 0])  # bearing
    # 俯仰角 true: elevation = arcsin(z / r)
    el_true = np.arcsin(p_relative[:, 2] / (r_true + 1e-10))
    # 多普勒径向速度真值 (v_r = v · r/|r|) 3D
    vr_true = np.sum(v * p_relative, axis=1) / (r_true + 1e-10)

    # 添加量测噪声（角度以度为单位）
    r_meas = r_true + np.random.normal(scale=noise_std_r, size=r_true.shape)
    az_true_deg = np.rad2deg(theta_true)
    el_true_deg = np.rad2deg(el_true)
    az_meas_deg = az_true_deg + np.random.normal(scale=noise_std_az, size=az_true_deg.shape)
    el_meas_deg = el_true_deg + np.random.normal(scale=noise_std_el, size=el_true_deg.shape)
    vr_meas = vr_true + np.random.normal(scale=noise_std_vr, size=vr_true.shape)

    r_meas = np.clip(r_meas, 1e-6, None)

    measurements_polar = np.column_stack([r_meas, az_meas_deg, el_meas_deg, vr_meas]).astype(np.float32)

    # 反算为测量坐标 (x,y,z) 用于可视化 —— 使用测量的角度（以度为单位），在计算时转换为弧度
    az_rad = np.deg2rad(az_meas_deg)
    el_rad = np.deg2rad(el_meas_deg)
    # x = r * cos(el) * cos(az)
    x_meas = r_meas * np.cos(el_rad) * np.cos(az_rad) + radar_pos[0]
    y_meas = r_meas * np.cos(el_rad) * np.sin(az_rad) + radar_pos[1]
    z_meas = r_meas * np.sin(el_rad)
    meas_xyz = np.column_stack([x_meas, y_meas, z_meas]).astype(np.float32)

    return {
        'positions': p.astype(np.float32),  # (T,3)
        'measurements_polar': measurements_polar,  # (T,4) (r, az_deg, el_deg, vr)
        'velocities': v.astype(np.float32),  # (T,3)
        'accelerations': a.astype(np.float32),  # (T,3)
        'meas_xy': meas_xyz[:, :2].astype(np.float32),
        'meas_xyz': meas_xyz,
        'radar_pos': radar_pos.astype(np.float32),
        'trajectory_type': trajectory_type
    }


def generate_dataset(N=1000, T=200, trajectory_distribution=None, **kwargs):
    types = ['uniform', 'accelerate', 'decelerate', 'turn', 'hover', 'mixed']
    if trajectory_distribution is None:
        trajectory_distribution = {k: 1.0 / len(types) for k in types}

    # build type list
    type_list = []
    for traj_type, prob in trajectory_distribution.items():
        count = max(1, int(N * prob))
        type_list.extend([traj_type] * count)
    # trim or pad
    type_list = type_list[:N]
    while len(type_list) < N:
        type_list.append(np.random.choice(types))
    np.random.shuffle(type_list)

    dataset = [generate_trajectory(T=T, trajectory_type=traj_type, **kwargs) for traj_type in type_list]
    return dataset


if __name__ == "__main__":
    traj = generate_trajectory(T=200, trajectory_type='mixed', dt=1.0)
    print(f"positions shape: {traj['positions'].shape}")
    print(f"measurements shape: {traj['measurements_polar'].shape}")
    print(f"velocity range: {np.min(np.linalg.norm(traj['velocities'], axis=1)):.2f} - {np.max(np.linalg.norm(traj['velocities'], axis=1)):.2f} m/s")
    print(f"range: {np.min(traj['measurements_polar'][:, 0]):.2f} - {np.max(traj['measurements_polar'][:, 0]):.2f} m")
