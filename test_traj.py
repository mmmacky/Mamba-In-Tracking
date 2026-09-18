import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.traj import generate_trajectory


OUT_DIR = 'e:/python/做一个有条理的人/mamba（4D-结项原版）/3D_run/run_test'
os.makedirs(OUT_DIR, exist_ok=True)

TYPES = ['uniform', 'accelerate', 'decelerate', 'turn', 'hover', 'mixed']
NUM_PER_TYPE = 3


def plot_and_save(traj, traj_type, idx):
    p = traj['positions']  # (T,3)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'b-', linewidth=1)
    ax.scatter(p[0, 0], p[0, 1], p[0, 2], c='g', s=50, label='Start')
    ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], c='r', s=50, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"{traj_type} #{idx+1}")
    ax.legend()
    plt.tight_layout()

    fname = os.path.join(OUT_DIR, f"{traj_type}_{idx+1}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def main():
    print('Generating and plotting trajectories...')
    saved = []
    for traj_type in TYPES:
        for i in range(NUM_PER_TYPE):
            traj = generate_trajectory(T=200, trajectory_type=traj_type, dt=1.0,
                                       noise_std_r=5.0, noise_std_az=1.0, noise_std_el=1.0,
                                       noise_std_vr=0.1, radar_pos=(0.0, 0.0))
            path = plot_and_save(traj, traj_type, i)
            saved.append(path)
            print('Saved:', path)
    print('Done. Total images saved:', len(saved))


if __name__ == '__main__':
    main()
