import numpy as np
from gen_model_gen_CV import gen_model_gen

def gen_newstate_target(model, Xd, V, k, inter_frame):
    # Ensure Xd is a 2D column vector (state_dim x num_cols)
    Xd = np.asarray(Xd)
    if Xd.ndim == 1:
        Xd = Xd.reshape(-1, 1)
    # nonlinear state space equation (CT model)
    if not isinstance(V, np.ndarray):
        if V == 'noise':
            # construct process noise for x, y, z separately using corresponding B matrices
            V1 = model['sigma_ax'] * model['B1'] @ np.random.randn(model['B1'].shape[1], Xd.shape[1])
            V2 = model['sigma_ay'] * model['B2'] @ np.random.randn(model['B2'].shape[1], Xd.shape[1])
            V3 = model['sigma_az'] * model['B3'] @ np.random.randn(model['B3'].shape[1], Xd.shape[1])
            V = np.vstack([V1, V2, V3])
        elif V == 'noiseless':
            V = np.zeros((model['B'].shape[0], Xd.shape[1]))

    if Xd.size == 0:
        X = np.array([])
    else:
        # ensure float
        Xd = Xd.astype(float)
        V = V.astype(float)
        # modify below here for user specified transition model
        X = model['F'] @ Xd + V

    return X