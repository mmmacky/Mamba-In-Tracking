#二维协调转弯运动模型
import numpy as np

def gen_newstate_plane(model, Xd, V, k):
    if not isinstance(V, np.ndarray):
        if V == 'noise':
            V = model['B_plane'] @ np.random.randn(model['B_plane'].shape[1], Xd.shape[1])
        elif V == 'noiseless':
            V = np.zeros((model['B_plane'].shape[0], Xd.shape[1]))

    if Xd.size == 0:
        return np.array([])

    # 确保输入为浮点，避免后续加法类型错误
    Xd = Xd.astype(float)

    X = np.zeros_like(Xd, dtype=float)
    L = Xd.shape[1]
    T = model['T']

    # Turn rate change logic
    if 250 <= k <= 400:
        Xd[4, :] = np.pi / 180 * 10
    else:
        Xd[4, :] = 3 * (np.pi / 180)

    omega = Xd[4, :]
    tol = 1e-10

    sin_omega_T = np.sin(omega * T)
    cos_omega_T = np.cos(omega * T)
    a = T * np.ones(L)
    b = np.zeros(L)

    idx = np.abs(omega) > tol
    a[idx] = sin_omega_T[idx] / omega[idx]
    b[idx] = (1 - cos_omega_T[idx]) / omega[idx]

    # State update
    X[0, :] = Xd[0, :] + a * Xd[1, :] - b * Xd[3, :]
    X[1, :] = cos_omega_T * Xd[1, :] - sin_omega_T * Xd[3, :]
    X[2, :] = b * Xd[1, :] + Xd[2, :] + a * Xd[3, :]
    X[3, :] = sin_omega_T * Xd[1, :] + cos_omega_T * Xd[3, :]
    X[4, :] = Xd[4, :]

    # Add noise
    X += model['B2_plane'] @ V

    return X

'''初始：从 [21000, -168, 25000, -168] 开始

前249步：3°/s 缓转弯

250-400步：10°/s 急转弯

401步后：恢复 3°/s 缓转弯'''