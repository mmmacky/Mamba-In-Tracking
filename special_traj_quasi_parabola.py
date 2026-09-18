import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

"""
确定性类抛物线轨迹（平滑，空中起止）
- 全程统一抛物线：X/Y 匀速，Z = z0 + vz0*t + 0.5*g*t^2
- 不进行地面高度夹紧，轨迹保持在空中开始与结束（参数可调确保不为负）
"""


def generate_quasi_parabola(T=360, dt=1.0):
    # 初始位置与平滑抛物线参数
    p0 = np.array([0.0, -2000.0, 300.0])
    vx = 10.0   # X 方向匀速 (m/s)
    vy = 1.5    # Y 方向匀速 (m/s)
    vz0 = 4.0   # 初始垂直速度 (m/s)
    g = -0.04   # 垂直加速度（重力方向向下）(m/s^2)

    # 预分配
    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)

    p[0] = p0
    v[0] = np.array([vx, vy, vz0])

    # 平滑抛物线运动学
    for t in range(1, T):
        t_s = t * dt
        # 位置
        x = p0[0] + vx * t_s
        y = p0[1] + vy * t_s
        z = p0[2] + vz0 * t_s + 0.5 * g * (t_s ** 2)
        p[t] = np.array([x, y, z])
        # 速度
        vz = vz0 + g * t_s
        v[t] = np.array([vx, vy, vz])

    info = {
        'model': 'smooth_parabola_air',
        'params': {
            'dt': dt,
            'T': T,
            'p0': p0.tolist(),
            'vx': vx,
            'vy': vy,
            'vz0': vz0,
            'g': g,
        },
        'description': '空中起止的平滑抛物线：不进行地面夹紧，保持连续光滑'
    }

    return p, v, info


def visualize_and_describe(p, v, info):
    print('运动模型:', info.get('model'))
    for k, val in info.get('params', {}).items():
        print(f"  {k}: {val}")
    print('说明:', info.get('description'))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], label='smooth quasi-parabola (air)', color='C2')
    ax.scatter([p[0, 0], p[-1, 0]], [p[0, 1], p[-1, 1]], [p[0, 2], p[-1, 2]], c=['g', 'r'], s=40, label='start/end')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('parabola')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    p, v, info = generate_quasi_parabola(T=360, dt=1.0)
    visualize_and_describe(p, v, info)
