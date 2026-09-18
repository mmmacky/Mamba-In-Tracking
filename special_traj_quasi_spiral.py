import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

"""
确定性类螺旋轨迹（平滑，空中起止）
- 段1：半径线性缩小 + 匀速上升
- 段2：恒定半径绕转 + 小幅上升
- 段3：半径线性增大 + 小幅下降（均保持在空中范围内）
- 全程连续平滑，无地面夹紧
"""


def generate_quasi_spiral(T=400, dt=1.0):
    center = np.array([0.0, 0.0, 0.0])
    R0 = 4000.0
    omega1 = np.deg2rad(2.0)
    omega2 = np.deg2rad(2.0)
    omega3 = np.deg2rad(2.0)
    vz1 = 2.0
    vz2 = 1.0
    vz3 = -0.5

    T1 = 150
    T2 = 300
    assert T1 < T2 < T

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)

    theta = np.deg2rad(30.0)
    z = 500.0

    for t in range(T):
        if t < T1:
            frac = t / T1
            R = R0 * (1.0 - 0.4 * frac)
            omega = omega1
            vz = vz1
        elif t < T2:
            R = R0 * 0.6
            omega = omega2
            vz = vz2
        else:
            frac = (t - T2) / (T - T2)
            R = R0 * (0.6 + 0.3 * frac)
            omega = omega3
            vz = vz3

        x = center[0] + R * np.cos(theta)
        y = center[1] + R * np.sin(theta)
        p[t] = np.array([x, y, z])
        vx = -R * np.sin(theta) * omega
        vy = R * np.cos(theta) * omega
        v[t] = np.array([vx, vy, vz])

        theta += omega * dt
        z += vz * dt

    info = {
        'segments': [
            {
                'name': '段1: 内收螺旋 + 匀速上升',
                't_range': (0, T1 - 1), 'dt': dt,
                'omega_deg_s': float(np.rad2deg(omega1)),
                'radius_start_end': (float(R0), float(R0 * 0.6)), 'vz': float(vz1),
            },
            {
                'name': '段2: 恒定半径 + 小幅上升',
                't_range': (T1, T2 - 1), 'dt': dt,
                'omega_deg_s': float(np.rad2deg(omega2)),
                'radius_const': float(R0 * 0.6), 'vz': float(vz2),
            },
            {
                'name': '段3: 外扩螺旋 + 小幅下降（空中）',
                't_range': (T2, T - 1), 'dt': dt,
                'omega_deg_s': float(np.rad2deg(omega3)),
                'radius_start_end': (float(R0 * 0.6), float(R0 * 0.9)), 'vz': float(vz3),
            },
        ],
        'description': '空中起止的平滑类螺旋：半径与高度分段线性/常量控制，角速度恒定分段，轨迹连续光滑'
    }

    return p, v, info


def visualize_and_describe(p, v, info):
    for seg in info['segments']:
        print(seg['name'])
        print(f"  时间步范围: {seg['t_range']}, dt={seg['dt']}s")
        for k, val in seg.items():
            if k in ['name', 't_range', 'dt']:
                continue
            print(f"  {k}: {val}")
    print('说明:', info.get('description'))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], label='smooth quasi-spiral (air)', color='C1')
    ax.scatter([p[0, 0], p[-1, 0]], [p[0, 1], p[-1, 1]], [p[0, 2], p[-1, 2]], c=['g', 'r'], s=40, label='start/end')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('spiral')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    p, v, info = generate_quasi_spiral(T=400, dt=1.0)
    visualize_and_describe(p, v, info)
