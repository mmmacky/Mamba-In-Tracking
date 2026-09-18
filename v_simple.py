import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 可视化：雷达极坐标滤波效果（含多普勒速度）
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.traj import generate_trajectory
from datasets.td import TrajectoryDataset
from models.trajectory_mamba import TrajectoryMamba

OUT_DIR = '3D_run/radar_filter_runs'
RUN_PLOTS = os.path.join('3D_run', 'run_test')
os.makedirs(RUN_PLOTS, exist_ok=True)

# 用户可修改的参数：要生成的轨迹类型
TRAJ_TYPE = 'turn'  # 可选: 'uniform','accelerate','decelerate','turn','hover','mixed'
T = 200

# 生成一条指定类型的轨迹（每次为随机实例）
traj = generate_trajectory(
    T=T,
    trajectory_type=TRAJ_TYPE,
    dt=1.0,
    noise_std_r=5.0,
    noise_std_az=1.0,   # 单位：度
    noise_std_el=1.0,   # 单位：度
    noise_std_vr=0.1,
    radar_pos=(0.0, 0.0, 0.0)
)

positions = traj['positions']  # (T,3)
velocities = traj['velocities']  # (T,3)
measurements_xyz = traj.get('meas_xyz')  # (T,3)
measurements_xy = traj.get('meas_xy')
measurements_polar = traj.get('measurements_polar')
radar_pos = traj.get('radar_pos', np.array([0.0, 0.0, 0.0]))

# 打印轨迹的运动参数概览
speeds = np.linalg.norm(velocities, axis=1)
accels = np.linalg.norm(traj['accelerations'], axis=1)
print('Generated trajectory:')
print(f'  type: {TRAJ_TYPE}')
print(f'  length T: {T}')
print(f'  initial position: {positions[0]}')
print(f'  initial velocity: {velocities[0]} (speed {speeds[0]:.2f} m/s)')
print(f'  speed: min={speeds.min():.2f}, mean={speeds.mean():.2f}, max={speeds.max():.2f} m/s')
print(f'  accel magnitude: min={accels.min():.4f}, mean={accels.mean():.4f}, max={accels.max():.4f} m/s^2')

# 加载训练好的模型
meta_path = os.path.join(OUT_DIR, 'meta.json')
model_path = os.path.join(OUT_DIR, 'model.pt')
if not os.path.exists(meta_path) or not os.path.exists(model_path):
    raise FileNotFoundError('meta.json 或 model.pt 未找到，请确认训练结果位于 3D_run/radar_filter_runs')

with open(meta_path, 'r') as f:
    meta = json.load(f)

model = TrajectoryMamba(d_model=meta['d_model'], n_layers=meta['n_layers'], seq_len=meta['seq_len'])
state = torch.load(model_path, map_location='cpu')
model.load_state_dict(state)
model.eval()
print('Loaded model with seq_len=', meta['seq_len'])

# 构造数据集并用模型逐窗口预测
seq_len = meta['seq_len']
traj_ds = TrajectoryDataset([traj], seq_len=seq_len)
loader = torch.utils.data.DataLoader(traj_ds, batch_size=64, shuffle=False)

preds = []
trues = []

time_indices = []
global_start = 0
with torch.no_grad():
    for xb, yb in loader:
        pred = model(xb)  # (B,4): x,y,z,vr
        pred_np = pred.cpu().numpy()
        true_np = yb.cpu().numpy()

        batch_size = xb.shape[0]
        for i in range(batch_size):
            time_idx = global_start + i + seq_len - 1
            time_indices.append(time_idx)
        global_start += batch_size

        preds.append(pred_np)
        trues.append(true_np)

if len(preds) == 0:
    raise RuntimeError('No predictions produced. Check seq_len and trajectory length.')

preds = np.vstack(preds)  # (N,4)
trues = np.vstack(trues)  # (N,4)
time_indices = np.array(time_indices)

pred_pos = preds[:, :3]
pred_vr = preds[:, 3]
true_pos = trues[:, :3]
true_vr = trues[:, 3]

# 对齐：pred_pos 对应时间点为 time_indices

# 计算RMSE（位置和径向速度）
pos_rmse = np.sqrt(np.mean(np.sum((pred_pos - true_pos) ** 2, axis=1)))
vel_rmse = np.sqrt(np.mean((pred_vr - true_vr) ** 2))
print('\nTracking results:')
print(f'  position RMSE: {pos_rmse:.4f} m')
print(f'  radial velocity RMSE: {vel_rmse:.4f} m/s')

# 可视化 3D：真实轨迹（线）、量测点（散点）、预测轨迹（虚线）
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# measurements
if measurements_xyz is not None:
    ax.scatter(measurements_xyz[:, 0], measurements_xyz[:, 1], measurements_xyz[:, 2],
               c='orange', s=10, alpha=0.6, label='Measurements')
elif measurements_xy is not None:
    ax.scatter(measurements_xy[:, 0], measurements_xy[:, 1], np.zeros(len(measurements_xy)),
               c='orange', s=10, alpha=0.6, label='Measurements')

# true trajectory (full)
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2.2, label='True Trajectory')

# predicted trajectory (only predicted time steps)
ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', linewidth=2.0, label='Predicted Trajectory')
ax.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], c='red', s=20, alpha=0.7)

# start marker
ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], s=160, c='green', marker='*', label='Start')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'Trajectory type={TRAJ_TYPE} | Pos RMSE={pos_rmse:.3f} m | Vel RMSE={vel_rmse:.3f} m/s')
ax.legend()
ax.view_init(elev=25, azim=-60)

out_path = os.path.join(RUN_PLOTS, f'traj_{TRAJ_TYPE}_single_3d.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ Figure saved to: {out_path}')


# 分析轨迹运动分段（基于速度、加速度和横向加速度）
def analyze_motion_segments(positions, velocities, accelerations, dt=1.0):
    """根据速度与加速度信号将轨迹分成运动段并输出参数。
    返回列表，每项为 dict: {'start':int,'end':int,'mode':str,'params':{...}}
    模式判定规则（经验阈值）：
      - hover: 速度很小
      - uniform: 加速度很小且横向加速度也小
      - accelerate / decelerate: 纵向加速度明显正/负
      - turn: 横向加速度或航向变化明显
    """
    T = positions.shape[0]
    speed = np.linalg.norm(velocities, axis=1)
    accel_mag = np.linalg.norm(accelerations, axis=1)

    # 水平速度与航向（用于判断转弯）
    horiz_v = np.linalg.norm(velocities[:, :2], axis=1)
    heading = np.arctan2(velocities[:, 1], velocities[:, 0])
    # 处理速度接近0时的heading跳变
    heading = np.nan_to_num(heading)
    heading_unwrap = np.unwrap(heading)
    yaw_rate = np.concatenate([[0.0], np.diff(heading_unwrap)]) / dt

    # 横向与纵向加速度分量（基于当前航向）
    # 单位切向向量
    heading_unit = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    perp_unit = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
    # 对速度近零点避免除0：当 horiz_v<1e-3 时使用 heading_unit from previous
    for i in range(1, T):
        if horiz_v[i] < 1e-3 and horiz_v[i-1] >= 1e-3:
            heading_unit[i] = heading_unit[i-1]
            perp_unit[i] = perp_unit[i-1]

    acc_xy = accelerations[:, :2]
    long_acc = np.sum(acc_xy * heading_unit, axis=1)
    lat_acc = np.sum(acc_xy * perp_unit, axis=1)

    # 阈值（可调整）
    TH_SPEED_HOVER = 0.5  # m/s
    TH_ACCEL_UNIFORM = 0.05  # m/s^2
    TH_LONG_ACCEL = 0.05
    TH_LAT_ACCEL = 0.02
    TH_YAW_RATE = np.deg2rad(0.2)  # rad per step

    labels = []
    for i in range(T):
        if speed[i] < TH_SPEED_HOVER:
            labels.append('hover')
        elif np.abs(long_acc[i]) > TH_LONG_ACCEL:
            labels.append('accelerate' if long_acc[i] > 0 else 'decelerate')
        elif np.abs(lat_acc[i]) > TH_LAT_ACCEL or np.abs(yaw_rate[i]) > TH_YAW_RATE:
            labels.append('turn')
        elif accel_mag[i] < TH_ACCEL_UNIFORM:
            labels.append('uniform')
        else:
            # fallback：看纵向符号
            labels.append('accelerate' if long_acc[i] > 0 else 'decelerate')

    # 合并连续相同标签为段
    segments = []
    if T == 0:
        return segments
    cur_label = labels[0]
    cur_start = 0
    for i in range(1, T):
        if labels[i] != cur_label:
            cur_end = i - 1
            seg_slice = slice(cur_start, cur_end + 1)
            seg = {
                'start': cur_start,
                'end': cur_end,
                'mode': cur_label,
                'params': {
                    'mean_speed': float(speed[seg_slice].mean()),
                    'median_speed': float(np.median(speed[seg_slice])),
                    'mean_long_acc': float(long_acc[seg_slice].mean()),
                    'mean_lat_acc': float(lat_acc[seg_slice].mean()),
                    'mean_acc_mag': float(accel_mag[seg_slice].mean()),
                    'mean_yaw_rate_deg': float(np.rad2deg(yaw_rate[seg_slice].mean()))
                }
            }
            # 若为 turn，估计角速度与半径
            if cur_label == 'turn':
                omega = np.nanmean(yaw_rate[seg_slice])
                mean_v = speed[seg_slice].mean()
                if np.abs(omega) > 1e-6:
                    radius = mean_v / omega
                else:
                    radius = float('inf')
                seg['params'].update({'omega_mean_rad': float(omega), 'est_turn_radius_m': float(radius)})

            segments.append(seg)
            cur_label = labels[i]
            cur_start = i
    # append last
    cur_end = T - 1
    seg_slice = slice(cur_start, cur_end + 1)
    seg = {
        'start': cur_start,
        'end': cur_end,
        'mode': cur_label,
        'params': {
            'mean_speed': float(speed[seg_slice].mean()),
            'median_speed': float(np.median(speed[seg_slice])),
            'mean_long_acc': float(long_acc[seg_slice].mean()),
            'mean_lat_acc': float(lat_acc[seg_slice].mean()),
            'mean_acc_mag': float(accel_mag[seg_slice].mean()),
            'mean_yaw_rate_deg': float(np.rad2deg(yaw_rate[seg_slice].mean()))
        }
    }
    if cur_label == 'turn':
        omega = np.nanmean(yaw_rate[seg_slice])
        mean_v = speed[seg_slice].mean()
        seg['params'].update({'omega_mean_rad': float(omega), 'est_turn_radius_m': float(mean_v / omega) if np.abs(omega) > 1e-6 else float('inf')})
    segments.append(seg)

    return segments

# 在生成轨迹后调用，打印更详细的分段信息
segments = analyze_motion_segments(positions, velocities, traj['accelerations'], dt=1.0)
print('\nMotion segments:')
for s in segments:
    start, end = s['start'], s['end']
    mode = s['mode']
    p = s['params']
    print(f'  - {start} -> {end}: {mode}')
    if mode in ['accelerate', 'decelerate']:
        print(f"      mean_speed={p['mean_speed']:.2f} m/s, mean_long_acc={p['mean_long_acc']:.3f} m/s^2")
    elif mode == 'turn':
        print(f"      mean_speed={p['mean_speed']:.2f} m/s, mean_yaw_rate={p['mean_yaw_rate_deg']:.3f} deg/step, est_radius={p.get('est_turn_radius_m', float('nan')):.2f} m")
    elif mode == 'hover':
        print(f"      mean_speed={p['mean_speed']:.2f} m/s (hover), mean_acc={p['mean_acc_mag']:.4f} m/s^2")
    else:
        print(f"      mean_speed={p['mean_speed']:.2f} m/s, mean_acc={p['mean_acc_mag']:.4f} m/s^2")