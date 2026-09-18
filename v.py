import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 可视化：雷达极坐标滤波效果（含多普勒速度）
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from data.traj import generate_dataset
from datasets.td import TrajectoryDataset
from models.trajectory_mamba import TrajectoryMamba

OUT_DIR = '3D_run/radar_filter_runs'  
PLOTS_DIR = os.path.join(OUT_DIR, 'plots_doppler')
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_loss_curve(history_path):
    """为每一条指标单独绘图并保存为独立图片；横坐标精细到每一个 epoch。"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    train_rmse = history.get('train_rmse', [])
    val_rmse = history.get('val_rmse', [])
    train_pos_rmse = history.get('train_pos_rmse', [])
    val_pos_rmse = history.get('val_pos_rmse', [])
    train_vel_rmse = history.get('train_vel_rmse', [])
    val_vel_rmse = history.get('val_vel_rmse', [])

    epochs = list(range(1, max(1, len(train_loss)) + 1))

    def save_curve(x, y_series, labels, title, ylabel, fname, log_scale=False):
        fig, ax = plt.subplots(figsize=(10, 5))
        for y, lab, style in y_series:
            ax.plot(x, y, style, linewidth=2, label=lab)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        # 精细到每个 epoch 的刻度（当 epoch 数不太大时设置刻度）
        if len(x) <= 200:
            ax.set_xticks(x)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xticks(rotation=45)
        else:
            # 对于大量 epoch，仍使用整数量化但间隔较大以免重叠
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_scale:
            ax.set_yscale('log')
        out_path = os.path.join(PLOTS_DIR, fname)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved plot: {out_path}")

    # 总损失
    save_curve(epochs, [
        (train_loss, 'Train Loss', 'b-'),
        (val_loss, 'Val Loss', 'r-')
    ], labels=None, title='Total MSE Loss', ylabel='MSE Loss', fname='loss_total.png', log_scale=False)

    # 总 RMSE
    save_curve(epochs, [
        (train_rmse, 'Train RMSE', 'b-'),
        (val_rmse, 'Val RMSE', 'r-')
    ], labels=None, title='Total RMSE', ylabel='RMSE', fname='rmse_total.png')

    # 位置 RMSE
    save_curve(epochs, [
        (train_pos_rmse, 'Train Pos RMSE', 'b-'),
        (val_pos_rmse, 'Val Pos RMSE', 'r-')
    ], labels=None, title='Position RMSE', ylabel='Position RMSE (m)', fname='pos_rmse.png')

    # 速度 RMSE
    save_curve(epochs, [
        (train_vel_rmse, 'Train Vel RMSE', 'b-'),
        (val_vel_rmse, 'Val Vel RMSE', 'r-')
    ], labels=None, title='Radial Velocity RMSE', ylabel='Velocity RMSE (m/s)', fname='vel_rmse.png')

    # 损失对数坐标（单独图）
    save_curve(epochs, [
        (train_loss, 'Train Loss', 'b-'),
        (val_loss, 'Val Loss', 'r-')
    ], labels=None, title='Total MSE Loss (Log Scale)', ylabel='MSE Loss (log scale)', fname='loss_log.png', log_scale=True)

    # 保存训练摘要为文本图（保留原有 summary 信息）
    best_val_epoch = np.argmin(val_loss) + 1 if len(val_loss) > 0 else 0
    best_val_loss = float(min(val_loss)) if len(val_loss) > 0 else float('nan')
    best_val_rmse = val_rmse[best_val_epoch - 1] if len(val_rmse) >= best_val_epoch and best_val_epoch > 0 else float('nan')
    best_val_pos_rmse = val_pos_rmse[best_val_epoch - 1] if len(val_pos_rmse) >= best_val_epoch and best_val_epoch > 0 else float('nan')
    best_val_vel_rmse = val_vel_rmse[best_val_epoch - 1] if len(val_vel_rmse) >= best_val_epoch and best_val_epoch > 0 else float('nan')

    summary_text = f"Best Validation Performance:\n\n"
    summary_text += f"Epoch: {best_val_epoch}\n"
    summary_text += f"Total Loss: {best_val_loss:.6f}\n"
    summary_text += f"Total RMSE: {best_val_rmse:.4f}\n"
    summary_text += f"Position RMSE: {best_val_pos_rmse:.2f} m\n"
    summary_text += f"Velocity RMSE: {best_val_vel_rmse:.3f} m/s\n\n"
    summary_text += f"Final Metrics:\n"
    if len(train_loss) > 0:
        summary_text += f"Train Loss: {train_loss[-1]:.6f}\n"
    if len(val_loss) > 0:
        summary_text += f"Val Loss: {val_loss[-1]:.6f}\n"
    if len(train_pos_rmse) > 0:
        summary_text += f"Train Pos RMSE: {train_pos_rmse[-1]:.2f} m\n"
    if len(val_pos_rmse) > 0:
        summary_text += f"Val Pos RMSE: {val_pos_rmse[-1]:.2f} m\n"
    if len(train_vel_rmse) > 0:
        summary_text += f"Train Vel RMSE: {train_vel_rmse[-1]:.3f} m/s\n"
    if len(val_vel_rmse) > 0:
        summary_text += f"Val Vel RMSE: {val_vel_rmse[-1]:.3f} m/s"

    # 将摘要写入单独的文本文件以便查看
    summary_path = os.path.join(PLOTS_DIR, 'training_summary.txt')
    with open(summary_path, 'w') as sf:
        sf.write(summary_text)

    print(f"✅ Training summary saved: {summary_path}")
    print(f"   Best val - Pos RMSE: {best_val_pos_rmse:.2f}m, Vel RMSE: {best_val_vel_rmse:.3f}m/s at epoch {best_val_epoch}")

def plot_trajectory_comparison(model, val_data, top_k=3):
    """绘制轨迹对比：位置 + 速度，空间视图改为3D"""
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    trajectory_scores = []
    all_data = []

    with torch.no_grad():
        for traj_idx, traj in enumerate(val_data):
            positions = traj['positions']  # (T, 3)
            velocities = traj['velocities']  # (T, 3)
            measurements_xyz = traj.get('meas_xyz')
            measurements_xy = traj.get('meas_xy')
            measurements_polar = traj.get('measurements_polar')

            # fallback: if only meas_xy exists, create meas_xyz with z=0
            if measurements_xyz is None and measurements_xy is not None:
                measurements_xyz = np.column_stack([measurements_xy, np.zeros(len(measurements_xy))])

            radar_pos = traj.get('radar_pos', np.array([0, 0, 0]))

            # 计算真实径向速度（沿三维向量）
            p_relative = positions - radar_pos
            r_magnitude = np.linalg.norm(p_relative, axis=1, keepdims=True)
            r_magnitude = np.maximum(r_magnitude, 1e-10)
            vr_true = np.sum(velocities * p_relative, axis=1) / r_magnitude.squeeze()

            seq_len = model.seq_len
            traj_ds = TrajectoryDataset([traj], seq_len=seq_len)
            traj_loader = torch.utils.data.DataLoader(traj_ds, batch_size=32, shuffle=False)

            pred_outputs = []
            true_outputs = []
            time_indices = []
            global_start_idx = 0

            for xb, yb in traj_loader:
                xb = xb.to(device)
                pred = model(xb)

                batch_size = xb.shape[0]
                for i in range(batch_size):
                    time_idx = global_start_idx + i + seq_len - 1
                    time_indices.append(time_idx)

                pred_outputs.extend(pred.cpu().numpy())
                true_outputs.extend(yb.cpu().numpy())
                global_start_idx += batch_size

            if len(pred_outputs) > 0:
                pred_outputs = np.array(pred_outputs)
                true_outputs = np.array(true_outputs)
                time_indices = np.array(time_indices)

                pred_pos = pred_outputs[:, :3]  # (N,3)
                pred_vr = pred_outputs[:, 3]
                true_pos = true_outputs[:, :3]
                true_vr = true_outputs[:, 3]

                pos_rmse = np.sqrt(np.mean(np.sum((pred_pos - true_pos) ** 2, axis=1)))
                vel_rmse = np.sqrt(np.mean((pred_vr - true_vr) ** 2))
                total_rmse = np.sqrt(pos_rmse ** 2 + vel_rmse ** 2)

                trajectory_scores.append((total_rmse, traj_idx))

                all_data.append({
                    'pred_pos': pred_pos,
                    'true_pos': true_pos,
                    'pred_vr': pred_vr,
                    'true_vr': true_vr,
                    'measurements': measurements_xyz,
                    'measurements_vr': measurements_polar[:, 3] if measurements_polar is not None and measurements_polar.shape[1] > 3 else None,
                    'radar_pos': radar_pos,
                    'time_indices': time_indices,
                    'pos_rmse': pos_rmse,
                    'vel_rmse': vel_rmse,
                    'total_rmse': total_rmse,
                    'full_true_pos': positions,
                    'full_true_vr': vr_true
                })

    # 选择效果最好的top_k条轨迹
    if trajectory_scores:
        best_indices = sorted(trajectory_scores, key=lambda x: x[0])[:top_k]

        for rank, (score, idx) in enumerate(best_indices):
            data = all_data[idx]

            # 创建2x2的综合对比图，左上改为3D空间视图
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            # measurements (scatter only)
            if data['measurements'] is not None:
                ax1.scatter(data['measurements'][:, 0], data['measurements'][:, 1], data['measurements'][:, 2],
                            c='orange', s=10, alpha=0.6, label='Measurements (points)', zorder=1)

            # true trajectory
            ax1.plot(data['full_true_pos'][:, 0], data['full_true_pos'][:, 1], data['full_true_pos'][:, 2],
                     'b-', linewidth=2.5, label='True Trajectory', zorder=3)

            # predicted trajectory
            ax1.plot(data['pred_pos'][:, 0], data['pred_pos'][:, 1], data['pred_pos'][:, 2],
                     'r--', linewidth=2, label='Filtered', zorder=4)

            # start point
            if data['full_true_pos'] is not None and len(data['full_true_pos']) > 0:
                ax1.scatter([data['full_true_pos'][0, 0]], [data['full_true_pos'][0, 1]], [data['full_true_pos'][0, 2]],
                            s=120, c='green', marker='*', label='Start', zorder=5)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title(f'Trajectory #{rank+1} - Spatial View\nPos RMSE: {data["pos_rmse"]:.2f}m', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.view_init(elev=25, azim=-60)

            # 右上：X/Y/Z 时间序列（用3条子图来展示）
            ax2 = fig.add_subplot(gs[0, 1])
            time_all = np.arange(len(data['measurements']))
            ax2.plot(time_all, data['full_true_pos'][:, 0], 'b-', label='True X', linewidth=2, alpha=0.8)
            ax2.plot(time_all, data['full_true_pos'][:, 1], 'g-', label='True Y', linewidth=2, alpha=0.8)
            ax2.plot(time_all, data['full_true_pos'][:, 2], 'c-', label='True Z', linewidth=2, alpha=0.8)
            ax2.plot(data['time_indices'], data['pred_pos'][:, 0], 'r--', label='Filtered X', linewidth=1.5)
            ax2.plot(data['time_indices'], data['pred_pos'][:, 1], 'm--', label='Filtered Y', linewidth=1.5)
            ax2.plot(data['time_indices'], data['pred_pos'][:, 2], 'y--', label='Filtered Z', linewidth=1.5)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Position (m)')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_title('X/Y/Z Position Time Series')

            # 左下：径向速度对比
            ax3 = fig.add_subplot(gs[1, 0])
            if data['measurements_vr'] is not None:
                ax3.scatter(time_all, data['measurements_vr'], c='orange', s=8, alpha=0.6, label='Meas vr')
            ax3.plot(time_all, data['full_true_vr'], 'b-', label='True vr', linewidth=2.2)
            ax3.plot(data['time_indices'], data['pred_vr'], 'r--', label='Filtered vr', linewidth=1.8)
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Radial Velocity (m/s)')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_title(f'Radial Velocity Comparison\nVel RMSE: {data["vel_rmse"]:.3f}m/s')

            # 右下：误差分析
            ax4 = fig.add_subplot(gs[1, 1])
            pos_errors = np.sqrt(np.sum((data['pred_pos'] - data['true_pos']) ** 2, axis=1))
            vel_errors = np.abs(data['pred_vr'] - data['true_vr'])
            ax4.plot(data['time_indices'], pos_errors, 'b-', linewidth=2, label='Position Error')
            ax4.axhline(y=np.mean(pos_errors), color='b', linestyle='--', alpha=0.6, label=f'Mean Pos: {np.mean(pos_errors):.2f}m')
            ax4.plot(data['time_indices'], vel_errors, 'r-', linewidth=2, label='Velocity Error')
            ax4.axhline(y=np.mean(vel_errors), color='r', linestyle='--', alpha=0.6, label=f'Mean Vel: {np.mean(vel_errors):.3f}m/s')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Error')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.set_title('Error Analysis')

            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'trajectory_comparison_{rank+1}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✅ Trajectory {rank+1}: Pos RMSE={data['pos_rmse']:.2f}m, Vel RMSE={data['vel_rmse']:.3f}m/s")


def plot_combined_trajectories(model, val_data, top_k=3):
    """绘制三条最佳轨迹的合并对比图（3D）"""
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    trajectory_scores = []
    all_best_data = []

    with torch.no_grad():
        for traj_idx, traj in enumerate(val_data):
            positions = traj['positions']
            measurements_xyz = traj.get('meas_xyz')
            measurements_xy = traj.get('meas_xy')
            if measurements_xyz is None and measurements_xy is not None:
                measurements_xyz = np.column_stack([measurements_xy, np.zeros(len(measurements_xy))])

            seq_len = model.seq_len
            traj_ds = TrajectoryDataset([traj], seq_len=seq_len)
            traj_loader = torch.utils.data.DataLoader(traj_ds, batch_size=32)

            pred_outputs = []
            true_outputs = []

            for xb, yb in traj_loader:
                xb = xb.to(device)
                pred = model(xb)
                pred_outputs.extend(pred.cpu().numpy())
                true_outputs.extend(yb.cpu().numpy())

            if len(pred_outputs) > 0:
                pred_outputs = np.array(pred_outputs)
                true_outputs = np.array(true_outputs)

                pred_pos = pred_outputs[:, :3]
                true_pos = true_outputs[:, :3]

                pos_rmse = np.sqrt(np.mean(np.sum((pred_pos - true_pos) ** 2, axis=1)))
                trajectory_scores.append((pos_rmse, traj_idx))

                all_best_data.append({
                    'pred_pos': pred_pos,
                    'true_pos': true_pos,
                    'measurements': measurements_xyz,
                    'full_true': positions,
                    'pos_rmse': pos_rmse
                })

    if trajectory_scores:
        best_indices = sorted(trajectory_scores, key=lambda x: x[0])[:top_k]
        best_data = [all_best_data[idx] for _, idx in best_indices]

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['blue', 'green', 'purple']

        for rank, data in enumerate(best_data):
            color = colors[rank % len(colors)]
            if data['measurements'] is not None:
                ax.scatter(data['measurements'][:, 0], data['measurements'][:, 1], data['measurements'][:, 2],
                           c=color, s=8, alpha=0.4, label=f'Traj{rank+1} Meas')

            ax.plot(data['full_true'][:, 0], data['full_true'][:, 1], data['full_true'][:, 2],
                    color=color, linewidth=2.5, label=f'Traj{rank+1} True')

            ax.plot(data['pred_pos'][:, 0], data['pred_pos'][:, 1], data['pred_pos'][:, 2],
                    linestyle='--', color=color, linewidth=2, label=f'Traj{rank+1} Filtered')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title('Best 3 Trajectories - Spatial Comparison (3D)', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(PLOTS_DIR, 'combined_trajectories_3d.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Combined trajectories (3D) plot saved")


def plot_selected_run_plots(model, val_data, out_dir=os.path.join('3D_run', 'run_plots'), select_counts=None):
    """按轨迹类型选择并保存最佳轨迹图（3D）。
    mixed 类型保存最好的 4 张，其他类型默认保存最好的 2 张。
    图中不绘制雷达位置，量测只用散点（不连线），显示真实轨迹、预测轨迹和起始位置。
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    types = ['uniform', 'accelerate', 'decelerate', 'turn', 'hover', 'mixed']
    if select_counts is None:
        select_counts = {t: 2 for t in types}
        select_counts['mixed'] = 4

    per_type = {t: [] for t in types}

    with torch.no_grad():
        for traj_idx, traj in enumerate(val_data):
            ttype = traj.get('trajectory_type', 'mixed')
            if ttype not in per_type:
                continue

            seq_len = getattr(model, 'seq_len', None)
            if seq_len is None:
                seq_len = 1

            traj_ds = TrajectoryDataset([traj], seq_len=seq_len)
            traj_loader = torch.utils.data.DataLoader(traj_ds, batch_size=64, shuffle=False)

            pred_outputs = []
            true_outputs = []

            for xb, yb in traj_loader:
                xb = xb.to(device)
                pred = model(xb)
                pred_outputs.extend(pred.cpu().numpy())
                true_outputs.extend(yb.cpu().numpy())

            if len(pred_outputs) == 0:
                continue

            pred_outputs = np.array(pred_outputs)
            true_outputs = np.array(true_outputs)

            pred_pos = pred_outputs[:, :3]
            true_pos = true_outputs[:, :3]

            pos_rmse = np.sqrt(np.mean(np.sum((pred_pos - true_pos) ** 2, axis=1)))

            per_type[ttype].append({
                'rmse': float(pos_rmse),
                'traj_idx': int(traj_idx),
                'pred_pos': pred_pos,
                'true_pos': true_pos,
                'measurements': traj.get('meas_xyz') if traj.get('meas_xyz') is not None else (np.column_stack([traj.get('meas_xy'), np.zeros(len(traj.get('meas_xy')))]) if traj.get('meas_xy') is not None else None),
                'full_true': traj.get('positions')
            })

    for ttype, items in per_type.items():
        if not items:
            continue
        items_sorted = sorted(items, key=lambda x: x['rmse'])[: max(0, select_counts.get(ttype, 2))]

        for rank, item in enumerate(items_sorted):
            pred_pos = item['pred_pos']
            full_true = item['full_true']
            measurements = item['measurements']
            rmse = item['rmse']
            traj_idx = item['traj_idx']

            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')

            # measurements: scatter only
            if measurements is not None:
                ax.scatter(measurements[:, 0], measurements[:, 1], measurements[:, 2],
                           s=18, c='orange', alpha=0.6, label='Measurements (points)', zorder=1)

            # true trajectory
            if full_true is not None:
                ax.plot(full_true[:, 0], full_true[:, 1], full_true[:, 2], 'b-', linewidth=2.2, label='True Trajectory', zorder=3)

            # predicted trajectory
            ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', linewidth=2.0, label='Predicted Trajectory', zorder=4)

            # start
            if full_true is not None and len(full_true) > 0:
                ax.scatter([full_true[0, 0]], [full_true[0, 1]], [full_true[0, 2]], s=140, c='green', marker='*', label='Start', zorder=5)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'{ttype} - Traj {traj_idx} - Pos RMSE: {rmse:.2f} m', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.view_init(elev=25, azim=-60)

            fname = f'{ttype}_traj_{traj_idx}_rank{rank+1}_3d.png'
            out_path = os.path.join(out_dir, fname)
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✅ Saved selected 3D plot: {out_path}")

def compute_average_rmse_by_type(model, val_data, batch_size=64):
    """对 val_data 中每种轨迹类型计算平均位置和速度 RMSE 并打印结果。"""
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    types = ['uniform', 'accelerate', 'decelerate', 'turn', 'hover', 'mixed']
    results = {t: {'pos': [], 'vel': []} for t in types}

    with torch.no_grad():
        for traj_idx, traj in enumerate(val_data):
            ttype = traj.get('trajectory_type', 'mixed')
            if ttype not in results:
                ttype = 'mixed'

            seq_len = getattr(model, 'seq_len', None)
            if seq_len is None:
                seq_len = 1

            traj_ds = TrajectoryDataset([traj], seq_len=seq_len)
            traj_loader = torch.utils.data.DataLoader(traj_ds, batch_size=batch_size, shuffle=False)

            pred_outputs = []
            true_outputs = []

            for xb, yb in traj_loader:
                xb = xb.to(device)
                pred = model(xb)
                pred_outputs.extend(pred.cpu().numpy())
                true_outputs.extend(yb.cpu().numpy())

            if len(pred_outputs) == 0:
                continue

            pred_outputs = np.array(pred_outputs)
            true_outputs = np.array(true_outputs)

            # 支持3D位置 + vr
            pred_pos = pred_outputs[:, :3]
            pred_vr = pred_outputs[:, 3] if pred_outputs.shape[1] > 3 else None
            true_pos = true_outputs[:, :3]
            true_vr = true_outputs[:, 3] if true_outputs.shape[1] > 3 else None

            pos_rmse = np.sqrt(np.mean(np.sum((pred_pos - true_pos) ** 2, axis=1)))
            if pred_vr is not None and true_vr is not None:
                vel_rmse = np.sqrt(np.mean((pred_vr - true_vr) ** 2))
            else:
                vel_rmse = float('nan')

            results[ttype]['pos'].append(pos_rmse)
            results[ttype]['vel'].append(vel_rmse)

    # 计算并打印每类平均值
    print('\nAverage RMSE by trajectory type:')
    avg_results = {}
    for t in types:
        pos_list = results[t]['pos']
        vel_list = results[t]['vel']
        pos_mean = float(np.nan) if len(pos_list) == 0 else float(np.mean(pos_list))
        vel_mean = float(np.nan) if len(vel_list) == 0 else float(np.nanmean(vel_list))
        avg_results[t] = {'pos_rmse_mean': pos_mean, 'vel_rmse_mean': vel_mean, 'count': len(pos_list)}
        print(f"  - {t}: count={len(pos_list):3d} | pos RMSE mean = {pos_mean if not np.isnan(pos_mean) else 'N/A'} | vel RMSE mean = {vel_mean if not np.isnan(vel_mean) else 'N/A'}")

    return avg_results

if __name__ == '__main__':
    # 绘制损失曲线
    hist_path = os.path.join(OUT_DIR, 'history.json')
    if os.path.exists(hist_path):
        plot_loss_curve(hist_path)
    else:
        print("❌ history.json not found, skipping loss curve")

    # 生成验证数据
    print("Generating validation data...")
    val_data = generate_dataset(
        N=500, T=200, 
        noise_std_r=5.0, 
        # generate_trajectory 接收角度噪声（度为单位），传入 1.0 表示 1°
        noise_std_az=1.0,
        noise_std_el=1.0,
        noise_std_vr=0.1
    )

    # 加载模型
    print("Loading model...")
    with open(os.path.join(OUT_DIR, 'meta.json'), 'r') as f:
        meta = json.load(f)
    
    model = TrajectoryMamba(
        d_model=meta['d_model'], 
        n_layers=meta['n_layers'], 
        seq_len=meta['seq_len'],
    )
    
    model_path = os.path.join(OUT_DIR, 'model.pt')
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    print("✅ Model loaded")

    # 绘制单个轨迹的详细对比
    print("Plotting trajectory comparisons...")
    plot_trajectory_comparison(model, val_data, top_k=3)
    
    # 绘制合并轨迹图
    print("Plotting combined trajectories...")
    plot_combined_trajectories(model, val_data, top_k=3)

    print("Plotting selected run plots by trajectory type...")
    plot_selected_run_plots(model, val_data, out_dir=os.path.join('3D_run', 'run_plots'))
    
    # 计算并打印每种轨迹类型的平均RMSE
    print("Computing average RMSE by trajectory type...")
    compute_average_rmse_by_type(model, val_data, batch_size=64)
    
    print("✅ All visualizations completed!")
    print(f"📁 Plots saved to: {PLOTS_DIR}")
    print(f"📁 Selected run plots saved to: {os.path.join('3D_run', 'run_plots')}")