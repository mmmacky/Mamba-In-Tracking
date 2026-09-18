import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# 引入必要的库和模型
# 假设上一轮的代码保存为了 special_traj_hard_maneuver.py
try:
    from traj_common_1 import generate_hard_maneuver_trajectory
except ImportError:
    raise ImportError("请确保 'special_traj_hard_maneuver.py' 在当前目录下，或者将该函数粘贴到本文件中。")

from datasets.td import TrajectoryDataset
from models.trajectory_mamba import TrajectoryMamba

# 模型路径配置
OUT_DIR = 'e:/python/做一个有条理的人/mamba（4D-结项原版）/3D_run/radar_filter_runs'

"""
使用已训练的 Mamba 模型对 [高机动性/蛇形机动] 轨迹进行跟踪测试
"""

def prepare_maneuver_data(T=200, dt=1.0):
    """
    生成高机动轨迹并适配 TrajectoryDataset 的数据格式
    """
    # 1. 调用生成器 (已经包含了真值生成、噪声添加、极坐标转换)
    # 注意：这里我们使用 dt=0.5s，采样率较高以捕捉机动细节
    raw_data = generate_hard_maneuver_trajectory(T=T, dt=dt)
    
    # 2. 提取数据
    p_true = raw_data['positions']      # (T, 3)
    v_true = raw_data['velocities']     # (T, 3)
    a_true = raw_data['accelerations']  # (T, 3)
    meas_polar = raw_data['measurements_polar'] # (T, 4): r, az, el, vr
    meas_xyz = raw_data['meas_xyz']     # (T, 3): 量测转换回笛卡尔坐标用于画图
    
    # 3. 封装成 Dataset 需要的字典格式
    # 这里的键名必须与 datasets/td.py 中要求的一致
    traj = {
        'positions': p_true,
        'velocities': v_true,
        'accelerations': a_true,
        'meas_xyz': meas_xyz,
        'measurements_xy': meas_xyz[:, :2], # 备用，如果是2D模型需要
        'measurements_polar': meas_polar,
        'radar_pos': np.array([0.0, 0.0, 0.0]), # 生成器默认原点
        'trajectory_type': 'hard_maneuver_snake',
    }
    
    return traj

def load_trained_model(out_dir: str):
    """
    加载训练好的模型权重和元数据
    """
    meta_path = os.path.join(out_dir, 'meta.json')
    model_path = os.path.join(out_dir, 'model.pt')
    
    if not (os.path.exists(meta_path) and os.path.exists(model_path)):
        raise FileNotFoundError(f'文件缺失: 请检查 {out_dir} 下是否有 meta.json 和 model.pt')
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    print(f"Loading model with config: d_model={meta['d_model']}, layers={meta['n_layers']}, seq_len={meta['seq_len']}")
    
    model = TrajectoryMamba(d_model=meta['d_model'], n_layers=meta['n_layers'], seq_len=meta['seq_len'])
    
    # 尝试加载权重
    try:
        state = torch.load(model_path, map_location='cpu', weights_only=True)
    except TypeError:
        # 兼容旧版 torch
        state = torch.load(model_path, map_location='cpu')
        
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        raise RuntimeError('模型文件格式错误，应为 state_dict')
        
    model.eval()
    return model, meta

def main():
    # ----------------------------------------------------------------
    # 1. 准备数据
    # ----------------------------------------------------------------
    # 生成 400 个时间步的数据
    traj = prepare_maneuver_data(T=400, dt=0.5)
    positions = traj['positions']
    measurements_xyz = traj['meas_xyz']
    
    print(f"轨迹生成完毕: {len(positions)} 步，类型: {traj['trajectory_type']}")

    # ----------------------------------------------------------------
    # 2. 加载模型
    # ----------------------------------------------------------------
    try:
        model, meta = load_trained_model(OUT_DIR)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    seq_len = meta['seq_len']

    # ----------------------------------------------------------------
    # 3. 构造 DataLoader 进行推理
    # ----------------------------------------------------------------
    traj_ds = TrajectoryDataset([traj], seq_len=seq_len)
    loader = torch.utils.data.DataLoader(traj_ds, batch_size=64, shuffle=False)

    preds = []
    trues = []
    
    print("开始推理跟踪...")
    with torch.no_grad():
        for xb, yb in loader:
            # xb shape: (Batch, Seq_Len, Features) -> 通常是量测数据
            # yb shape: (Batch, 4) -> (x, y, z, vr) 真值
            
            # 模型前向传播
            pred = model(xb)  # output: (B, 4)
            
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())

    # 拼接批次结果
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    
    # 提取位置 (x, y, z)
    pred_pos = preds[:, :3]
    true_pos = trues[:, :3]
    
    # 由于 Dataset 使用滑动窗口，预测结果会比原始轨迹短 (seq_len - 1)
    # 我们需要对齐时间轴进行绘图
    start_idx = seq_len - 1
    end_idx = start_idx + len(pred_pos)
    
    # 截取对应的原始真值和量测用于绘图对齐
    plot_true = positions[start_idx:end_idx]
    plot_meas = measurements_xyz[start_idx:end_idx]

    # ----------------------------------------------------------------
    # 4. 评估与可视化
    # ----------------------------------------------------------------
    # 计算 RMSE (Root Mean Square Error)
    pos_error = np.linalg.norm(pred_pos - plot_true, axis=1)
    rmse = np.sqrt(np.mean(pos_error ** 2))
    print(f"跟踪结束. Position RMSE: {rmse:.4f} m")

    # 绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 画真值 (完整轨迹作为背景)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            color='blue', alpha=0.3, linewidth=1, label='Ground Truth (Full)')
    
    # 2. 画当前推理段的真值 (加粗)
    ax.plot(plot_true[:, 0], plot_true[:, 1], plot_true[:, 2], 
            'b-', linewidth=2, label='Ground Truth (Window)')
    
    # 3. 画量测值 (含噪声)
    ax.scatter(plot_meas[:, 0], plot_meas[:, 1], plot_meas[:, 2], 
               c='orange', s=10, alpha=0.5, label='Measurements (Noisy)')
    
    # 4. 画模型预测轨迹
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
            'r--', linewidth=2, label='Mamba Tracked')
    
    # 起点和终点标记
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], c='g', s=50, marker='^', label='Track Start')
    ax.scatter(pred_pos[-1, 0], pred_pos[-1, 1], pred_pos[-1, 2], c='k', s=50, marker='x', label='Track End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Hard Maneuver Tracking (Mamba)\nPosition RMSE={rmse:.3f} m')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # 设置视角以更好地观察蛇形机动
    ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()