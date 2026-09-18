import numpy as np

def gen_model_gen():
    model = {}
    model['numSensor'] = 2
    
    # basic parameters
    model['x_dim'] = 9   # dimension of state vector
    model['z_dim_A'] = 4   # dimension of observation vector
    model['z_dim_P'] = 2   # dimension of observation vector
    model['wA_dim'] = 4   # radar dimension of observation noise
    model['wP_dim'] = 2   # infrared dimension of observation noise

    # dynamical model parameters (CA model)
    model['T'] = 24e-3
    model['A0'] = np.array([
        [1, model['T'], (model['T']**2)/2],
        [0, 1, model['T']],
        [0, 0, 1]
    ])
    
    # State transition matrix
    model['F'] = np.block([
        [model['A0'], np.zeros((3, 3)), np.zeros((3, 3))],
        [np.zeros((3, 3)), model['A0'], np.zeros((3, 3))],
        [np.zeros((3, 3)), np.zeros((3, 3)), model['A0']]
    ])
    
    model['B0'] = np.array([(model['T']**2)/2, model['T'], 1]).reshape(-1, 1)

    # x direction
    model['B1'] = np.hstack([model['B0'], np.zeros((3, 1)), np.zeros((3, 1))])
    model['sigma_ax'] = 0.5
    model['Q1'] = (model['sigma_ax']**2) * (model['B1'] @ model['B1'].T)

    # y direction
    model['B2'] = np.hstack([np.zeros((3, 1)), model['B0'], np.zeros((3, 1))])
    model['sigma_ay'] = 0.5
    model['Q2'] = (model['sigma_ay']**2) * (model['B2'] @ model['B2'].T)

    # z direction
    model['B3'] = np.hstack([np.zeros((3, 1)), np.zeros((3, 1)), model['B0']])
    model['sigma_az'] = 0.01
    model['Q3'] = (model['sigma_az']**2) * (model['B3'] @ model['B3'].T)

    # Process noise covariance
    model['Q'] = np.block([
        [model['Q1'], np.zeros((3, 6))],
        [np.zeros((3, 3)), model['Q2'], np.zeros((3, 3))],
        [np.zeros((3, 6)), model['Q3']]
    ])

    model['B'] = np.vstack([
        np.hstack([model['B0'], np.zeros((3, 1)), np.zeros((3, 1))]),
        np.hstack([np.zeros((3, 1)), model['B0'], np.zeros((3, 1))]),
        np.hstack([np.zeros((3, 1)), np.zeros((3, 1)), model['B0']])
    ])
    model['sigma_v'] = np.array([model['sigma_ax'], model['sigma_ay'], model['sigma_az']])

    # Observation noise
    model['D_A'] = np.diag([5, 0.3, 0.3, 1])  # for Gen
    model['R_A'] = model['D_A'] @ model['D_A'].T  # radar observation noise covariance

    model['D_P'] = np.diag([0.3, 0.3])  # for Gen
    model['R_P'] = model['D_P'] @ model['D_P'].T  # infrared observation noise covariance

    # detection parameters
    model['P_D'] = np.array([0.98, 0.98])   # probability of detection in measurements
    model['Q_D'] = 1 - model['P_D']  # probability of missed detection in measurements

    # clutter parameters radar
    model['lambda_c_radar'] = 20  # poisson average rate of uniform clutter (per scan)
    model['range_c_radar'] = np.array([[15000, 30000], [0, 90], [500, 1000]])  # uniform clutter on r/theta

    model['lambda_c_infrared'] = 20  # poisson average rate of uniform clutter (per scan)
    model['range_c_infrared'] = np.array([[15000, 30000], [15000, 30000]])  # uniform clutter on r/theta

    # uniform clutter density
    range_diff_radar = model['range_c_radar'][:, 1] - model['range_c_radar'][:, 0]
    model['pdf_c_radar'] = 1 / np.prod(range_diff_radar) / 100

    return model