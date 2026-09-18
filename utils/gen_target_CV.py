import numpy as np
from gen_newstatetarget_CV import gen_newstate_target
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gen_model_gen_CV import gen_model_gen

def gen_target(model):
    truth = {}
    truth['K'] = 800                   # length of data/number of scans
    truth['X'] = [None] * truth['K']             # ground truth for states of targets
    truth['N'] = np.zeros(truth['K'])            # ground truth for number of targets
    truth['L'] = [None] * truth['K']             # ground truth for labels of targets (k,i)
    truth['track_list'] = [None] * truth['K']    # absolute index target identities (plotting)
    truth['total_tracks'] = 0          # total number of appearing tracks

    # target initial states and birth/death times
    nbirths = 1

    # xstart(:,1)  = [ 20000; -168; 0;20000; -168; 0];        tbirth(1)  = 1;     tdeath(1)  = truth.K;
    xstart = np.array([[21000, -168, -5, 25000, -168, 0, 6000, 0, 0]]).T  # radar_1p5velocity
    tbirth = [1]
    tdeath = [truth['K']]

    # generate the tracks
    for targetnum in range(nbirths):
        targetstate = xstart[:, targetnum]
        for k in range(tbirth[targetnum], min(tdeath[targetnum], truth['K'])):
            if truth['X'][k] is None:
                truth['X'][k] = targetstate.reshape(-1, 1)
            else:
                truth['X'][k] = np.hstack([truth['X'][k], targetstate.reshape(-1, 1)])
            
            targetstate = gen_newstate_target(model, targetstate, 'noiseless', k, 200)
            
            if truth['track_list'][k] is None:
                truth['track_list'][k] = [targetnum + 1]
            else:
                truth['track_list'][k].append(targetnum + 1)
                
            truth['N'][k] = truth['N'][k] + 1
    
    # 提取轨迹数据
    x_pos = []
    y_pos = []
    z_pos = []
    
    for k in range(truth['K']):
        if truth['X'][k] is not None:
            x_pos.append(truth['X'][k][0, 0])  # x position
            y_pos.append(truth['X'][k][3, 0])  # y position (索引3，因为状态是[x,vx,ax,y,vy,ay,z,vz,az])
            z_pos.append(truth['X'][k][6, 0])  # z position (索引6)

     # 绘制2D轨迹
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_pos, y_pos, 'b-', linewidth=2)
    plt.plot(x_pos[0], y_pos[0], 'go', markersize=8, label='Start')
    plt.plot(x_pos[-1], y_pos[-1], 'ro', markersize=8, label='End')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Trajectory (XY Plane)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    # 绘制3D轨迹
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.plot(x_pos, y_pos, z_pos, 'b-', linewidth=2)
    ax.scatter(x_pos[0], y_pos[0], z_pos[0], c='g', s=50, label='Start')
    ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], c='r', s=50, label='End')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    plt.tight_layout()
    plt.show()
    truth['total_tracks'] = nbirths
    
    return truth

# 如果直接运行此脚本，构造模型并生成轨迹图
if __name__ == "__main__":
    model = gen_model_gen()
    truth = gen_target(model)
    # 打印点数供调试
    # 提取点数
    x_pos_count = 0
    for k in range(truth['K']):
        if truth['X'][k] is not None:
            x_pos_count += 1
    print('Generated timesteps with targets:', x_pos_count)
    # 保存最后一张图的备份（如果运行环境不弹窗，可在文件中查看）
    try:
        import matplotlib.pyplot as plt
        plt.savefig(r'e:\jupyter\mamba\3D\gen_target_CV_traj.png', dpi=150)
        print('Saved figure to e:\\jupyter\\mamba\\3D\\gen_target_CV_traj.png')
    except Exception as e:
        print('Could not save figure:', e)