import numpy as np

def gen_model_gen():
    model = {}
    model['numSensor'] = 1

    # State dimensions平面 CT model
    model['x_dim_plane'] = 5# x, vx, y, vy, omega转弯率 
    model['z_dim_A_plane'] = 3# range, bearing, altitude雷达观测维度
    model['z_dim_P_plane'] = 2#红外观测维度
    model['v_dim_plane'] = 3# velocity components: vx, vy, omega
    model['wA_dim_plane'] = 3# process noise for x, y, omega
    model['wP_dim_plane'] = 2# process noise for position measurements
    #三维 CA model
    model['x_dim'] = 9# x, vx, y, vy, z, vz
    model['z_dim_A'] = 4# range, bearing, elevation, altitude，距离、方位、俯仰、高度雷达测量维度
    model['z_dim_P'] = 2# infrared measurements
    model['wA_dim'] = 4# process noise for radar measurements
    model['wP_dim'] = 2# process noise for infrared measurements

    # Dynamical model (CA model)
    model['T'] = 24e-3
    model['sigma_vel_plane'] = 0.1#速度噪声标准差
    model['sigma_turn_plane'] = 0.1 * (np.pi / 180)#转弯率噪声标准差
    # 将 bt 定义为列向量 (2x1)，以便与其它块矩阵拼接
    model['bt'] = model['sigma_vel_plane'] * np.array([(model['T']**2)/2, model['T']]).reshape(-1, 1)
    model['B2_plane'] = np.block([
        [model['bt'], np.zeros((2, 2))],
        [np.zeros((2, 1)), model['bt'], np.zeros((2, 1))],
        [np.zeros((1, 2)), np.array([[model['T'] * model['sigma_turn_plane']]])]
    ])
    # 构建过程噪声转换矩阵：
    # [ bt   0    0  ]  影响x方向
    # [  0  bt    0  ]  影响y方向  
    # [  0   0   T*σω]  影响转弯率
    model['B_plane'] = np.eye(model['v_dim_plane'])# 3x3单位矩阵
    model['Q_plane'] = model['B_plane'] @ model['B_plane'].T# 过程噪声协方差矩阵

    # 3D model
    A0 = np.array([
        [1, model['T'], (model['T']**2)/2],
        [0, 1, model['T']],
        [0, 0, 1]
    ])
    model['A0'] = A0
    model['F'] = np.block([
        [A0, np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), A0, np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), A0]
    ])
    model['B0'] = np.array([(model['T']**2)/2, model['T'], 1]).reshape(-1, 1)

    # Process noise for x, y, z
    model['B1'] = np.hstack([model['B0'], np.zeros((3, 1)), np.zeros((3, 1))])
    model['sigma_ax'] = 0.43# acceleration noise standard deviation in x direction
    model['Q1'] = (model['sigma_ax']**2) * (model['B1'] @ model['B1'].T)

    model['B2'] = np.hstack([np.zeros((3, 1)), model['B0'], np.zeros((3, 1))])
    model['sigma_ay'] = 0.3
    model['Q2'] = (model['sigma_ay']**2) * (model['B2'] @ model['B2'].T)

    model['B3'] = np.hstack([np.zeros((3, 1)), np.zeros((3, 1)), model['B0']])
    model['sigma_az'] = 0.01
    model['Q3'] = (model['sigma_az']**2) * (model['B3'] @ model['B3'].T)

    model['Q'] = np.block([
        [model['Q1'], np.zeros((3, 6))],
        [np.zeros((3, 3)), model['Q2'], np.zeros((3, 3))],
        [np.zeros((3, 6)), model['Q3']]
    ])# 9x9过程噪声协方差矩阵

    model['B'] = np.vstack([
        np.hstack([model['B0'], np.zeros((3, 1)), np.zeros((3, 1))]),
        np.hstack([np.zeros((3, 1)), model['B0'], np.zeros((3, 1))]),
        np.hstack([np.zeros((3, 1)), np.zeros((3, 1)), model['B0']])
    ])
    model['sigma_v'] = np.array([model['sigma_ax'], model['sigma_ay'], model['sigma_az']])

    # Observation noise
    model['D_A'] = np.diag([5, 0.3, 0.3, 1])# 雷达观测噪声标准差对角矩阵
    model['R_A'] = model['D_A'] @ model['D_A'].T

    model['D_P'] = np.diag([0.2, 0.2])# 红外观测噪声标准差对角矩阵
    model['R_P'] = model['D_P'] @ model['D_P'].T

    # Detection parameters检测概率参数
    model['P_D'] = np.array([0.98, 0.98])
    model['Q_D'] = 1 - model['P_D']

    # Clutter parameters杂波参数
    model['lambda_c_radar'] = 2
    model['range_c_radar'] = np.array([[15000, 30000], [0, 90], [500, 1000]])
    model['lambda_c_infrared'] = 2
    model['range_c_infrared'] = np.array([[15000, 30000], [15000, 30000]])
    model['pdf_c_radar'] = 1e-7

    return model