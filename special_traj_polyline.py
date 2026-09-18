import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

"""
确定性折线轨迹：分三段
- 段1 [t=0..T1): 从起点以匀速直线运动到第一个拐点
- 段2 [t=T1..T2): 换方向，以更高速度直线到第二个拐点
- 段3 [t=T2..T): 再次换方向，以较低速度直线到终点，同时有轻微爬升
每段打印详细参数：起点、终点、速度向量、速度大小、持续时间、dt
并输出3D轨迹图
"""


def generate_polyline_trajectory(T=300, dt=1.0):
    # 关键拐点（世界坐标，米）
    p0 = np.array([1000.0, -1500.0, 200.0])
    p1 = np.array([3000.0, -500.0, 250.0])
    p2 = np.array([4500.0, 1000.0, 300.0])
    p3 = np.array([5200.0, 2500.0, 450.0])

    # 时间分段长度
    T1 = 120
    T2 = 220
    assert T1 < T2 < T, "时间分段不合法"

    p = np.zeros((T, 3), dtype=float)
    v = np.zeros((T, 3), dtype=float)

    # 段1：从 p0 到 p1，匀速
    dir1 = p1 - p0
    dur1 = T1 * dt
    v1 = dir1 / dur1

    # 段2：从 p1 到 p2，速度更快（缩短等效时间）
    dir2 = p2 - p1
    dur2 = (T2 - T1) * dt
    v2 = dir2 / dur2

    # 段3：从 p2 到 p3，速度较慢（轻微爬升已体现在 p3.z）
    dir3 = p3 - p2
    dur3 = (T - T2) * dt
    v3 = dir3 / dur3

    # 初始化起点
    p[0] = p0
    v[0] = v1

    # 按段生成轨迹
    for t in range(1, T):
        if t < T1:
            v[t] = v1
            p[t] = p[t-1] + v[t] * dt
        elif t < T2:
            v[t] = v2
            p[t] = p[t-1] + v[t] * dt
        else:
            v[t] = v3
            p[t] = p[t-1] + v[t] * dt

    return p, v, {
        'segments': [
            {
                'name': '段1: p0->p1 匀速',
                't_range': (0, T1-1),
                'dt': dt,
                'start': p0.tolist(),
                'end': p1.tolist(),
                'v': v1.tolist(),
                'speed': float(np.linalg.norm(v1)),
                'duration_s': float(dur1)
            },
            {
                'name': '段2: p1->p2 速度更快',
                't_range': (T1, T2-1),
                'dt': dt,
                'start': p1.tolist(),
                'end': p2.tolist(),
                'v': v2.tolist(),
                'speed': float(np.linalg.norm(v2)),
                'duration_s': float(dur2)
            },
            {
                'name': '段3: p2->p3 速度较慢且轻微爬升',
                't_range': (T2, T-1),
                'dt': dt,
                'start': p2.tolist(),
                'end': p3.tolist(),
                'v': v3.tolist(),
                'speed': float(np.linalg.norm(v3)),
                'duration_s': float(dur3)
            }
        ]
    }


def visualize_and_describe(p, v, info):
    # 打印段信息
    for seg in info['segments']:
        print(f"{seg['name']}")
        print(f"  时间步范围: {seg['t_range']}, dt={seg['dt']}s, 持续={seg['duration_s']:.1f}s")
        print(f"  起点: {seg['start']}")
        print(f"  终点: {seg['end']}")
        print(f"  速度向量: {seg['v']}")
        print(f"  速度大小: {seg['speed']:.3f} m/s")

    # 作图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], label='polyline trajectory', color='C0')
    ax.scatter([p[0, 0], p[-1, 0]], [p[0, 1], p[-1, 1]], [p[0, 2], p[-1, 2]], c=['g', 'r'], s=40, label='start/end')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('polyline')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    p, v, info = generate_polyline_trajectory(T=300, dt=1.0)
    visualize_and_describe(p, v, info)
