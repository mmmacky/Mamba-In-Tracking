import sys
import os

# ================= 关键修复开始 =================
# 1. 获取当前脚本所在的目录 (例如: .../track_fixed)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (即 current_dir 的上一级: .../mamba（4D-结项原版）)
# 这样无论你在哪里运行，都能自动找到正确的根目录，不需要手动写死路径
project_root = os.path.dirname(current_dir)

# 3. 将根目录加入 Python 搜索路径
# 只有加了这一句，Python 才能跨文件夹找到 fixed_traj, datasets, models
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"当前脚本路径: {current_dir}")
print(f"已加载项目根目录: {project_root}")
# ================= 关键修复结束 =================

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# 引入生成的轨迹函数
try:
    # 注意：请确保你的文件夹结构是 fixed_traj/fixed_polyline.py
    from fixed_traj.fixed_spiral import generate_spiral_fixed
except ImportError as e:
    print(f"导入错误详情: {e}")
    raise ImportError("导入失败。请检查：\n1. 根目录下是否有 fixed_traj 文件夹\n2. 该文件夹内是否有 fixed_polyline.py\n3. 该文件夹内是否有 __init__.py")

from datasets.td import TrajectoryDataset
from models.trajectory_mamba import TrajectoryMamba

# 模型路径 (请根据实际情况修改)
OUT_DIR = 'e:/python/做一个有条理的人/mamba（4D-结项原版）/3D_run/radar_filter_runs'

def load_trained_model(out_dir: str):
    """加载模型通用函数"""
    meta_path = os.path.join(out_dir, 'meta.json')
    model_path = os.path.join(out_dir, 'model.pt')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    model = TrajectoryMamba(d_model=meta['d_model'], n_layers=meta['n_layers'], seq_len=meta['seq_len'])
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        # 兼容旧版本 torch 保存格式
        model.load_state_dict(state)
        
    model.eval()
    return model, meta

def main():
    # 1. 生成固定轨迹数据
    # 调用刚才写的生成器，它已经包含了噪声固定的量测
    try:
        raw_data = generate_spiral_fixed(T=300, dt=1.0)
    except Exception as e:
        print(f"生成轨迹时出错: {e}")
        return
    
    # 补充 Dataset 可能需要的字段 (如 xy)
    traj_data = raw_data.copy()
    if 'measurements_xy' not in traj_data:
        # 如果生成器没有返回 xy，则手动计算补充
        traj_data['measurements_xy'] = raw_data['meas_xyz'][:, :2] 

    positions = traj_data['positions']
    meas_xyz = traj_data['meas_xyz']
    
    # 2. 加载模型
    try:
        model, meta = load_trained_model(OUT_DIR)
        seq_len = meta['seq_len']
    except Exception as e:
        print(f"加载模型出错: {e}")
        return
    
    # 3. 构造数据集
    # TrajectoryDataset 会自动处理归一化和滑动窗口
    ds = TrajectoryDataset([traj_data], seq_len=seq_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    
    # 4. 推理
    preds = []
    trues = [] # 用于验证对齐
    
    print("正在跟踪折线轨迹...")
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb) # (B, 4) -> x, y, z, vr
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
            
    if len(preds) == 0:
        print("未生成任何预测结果，请检查数据长度是否大于 seq_len")
        return

    preds = np.vstack(preds)
    pred_pos = preds[:, :3]
    
    # 5. 对齐时间轴
    # 模型输出是从第 seq_len 个点开始的
    start_idx = seq_len - 1
    end_idx = start_idx + len(pred_pos)
    
    truth_slice = positions[start_idx:end_idx]
    meas_slice = meas_xyz[start_idx:end_idx]
    
    # 计算误差
    rmse = np.sqrt(np.mean(np.sum((pred_pos - truth_slice)**2, axis=1)))
    print(f"Tracking RMSE: {rmse:.4f} m")
    
    # 6. 可视化
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画完整真值 (蓝色背景)
    ax.plot(positions[:,0], positions[:,1], positions[:,2], 'b-', alpha=0.3, label='Ground Truth (Full)')
    # 画当前跟踪段真值 (深蓝实线)
    ax.plot(truth_slice[:,0], truth_slice[:,1], truth_slice[:,2], 'b-', linewidth=2, label='Ground Truth (Segment)')
    
    # 画量测 (橙色散点)
    ax.scatter(meas_slice[:,0], meas_slice[:,1], meas_slice[:,2], c='orange', s=5, alpha=0.5, label='Measurements')
    
    # 画跟踪结果 (红色虚线)
    ax.plot(pred_pos[:,0], pred_pos[:,1], pred_pos[:,2], 'r--', linewidth=2, label='Mamba Tracked')
    
    ax.set_title(f'Polyline Tracking (RMSE={rmse:.2f}m)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()