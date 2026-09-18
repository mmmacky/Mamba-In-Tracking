import numpy as np
from gen_newstate_plane_Turn import gen_newstate_plane
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gen_plane(model):
    truth = {}
    truth['K'] = 800# total time steps
    truth['X'] = [None] * truth['K']
    truth['N'] = np.zeros(truth['K'])
    truth['track_list'] = [None] * truth['K']
    truth['total_tracks'] = 0

    nbirths = 1
    xstart = np.array([[21000, -168, 25000, -168, 0]]).T
    tbirth = [0]
    tdeath = [truth['K']]

    # Generate tracks
    for targetnum in range(nbirths):
        targetstate = xstart[:, targetnum]
        for k in range(tbirth[targetnum], min(tdeath[targetnum], truth['K'])):
            # 把平面状态转换为列向量，并仅追加一个高度值（6000），得到 (6,1) 列向量
            col_state = np.vstack([targetstate.reshape(-1, 1), np.array([[6000]])])
            if truth['X'][k] is None:
                truth['X'][k] = col_state.copy()
            else:
                truth['X'][k] = np.hstack([truth['X'][k], col_state])
            targetstate = gen_newstate_plane(model, targetstate.reshape(-1, 1), 'noise', k).flatten()
            if truth['track_list'][k] is None:
                truth['track_list'][k] = [targetnum + 1]
            else:
                truth['track_list'][k].append(targetnum + 1)
            truth['N'][k] += 1

    # Compute accelerations
    ax = np.zeros(truth['K'] - 1)
    ay = np.zeros(truth['K'] - 1)
    for i in range(1, truth['K']):
        ax[i-1] = truth['X'][i][1, 0] - truth['X'][i-1][1, 0]
        ay[i-1] = truth['X'][i][3, 0] - truth['X'][i-1][3, 0]

    # Extract trajectory data for plotting
    x_pos = np.zeros(truth['K'])
    y_pos = np.zeros(truth['K']) 
    z_pos = np.zeros(truth['K'])
    vx = np.zeros(truth['K'])
    vy = np.zeros(truth['K'])
    turn_rate = np.zeros(truth['K'])

    for i in range(truth['K']):
        if truth['X'][i] is None:
            x_pos[i] = np.nan
            y_pos[i] = np.nan
            z_pos[i] = np.nan
            vx[i] = np.nan
            vy[i] = np.nan
            turn_rate[i] = np.nan
            continue
        x_pos[i] = truth['X'][i][0, 0]  # x position
        y_pos[i] = truth['X'][i][2, 0]  # y position  
        z_pos[i] = truth['X'][i][5, 0]  # z position (fixed at 6000)
        vx[i] = truth['X'][i][1, 0]     # x velocity
        vy[i] = truth['X'][i][3, 0]     # y velocity
        turn_rate[i] = truth['X'][i][4, 0] * 180/np.pi  # turn rate in deg/s

    # Plot
    plt.figure()    
    # 第一个图 - 加速度
    plt.figure()
    plt.plot(range(1, truth['K']), ax, 'b', label='ax')
    plt.plot(range(1, truth['K']), ay, 'r', label='ay')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Acceleration')
    plt.title('Acceleration in x and y directions')
    plt.grid(True)
    plt.show()
    
    # 第二个图 - 3D轨迹
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_pos, y_pos, z_pos, 'b-', linewidth=2)
    ax.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', s=50, label='Start')
    ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', s=50, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    plt.show()

    truth['total_tracks'] = nbirths
    return truth

# Example usage:
if __name__ == "__main__":
    from gen_model_gen_Turn import gen_model_gen
    model = gen_model_gen()
    truth = gen_plane(model)