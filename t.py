# GPU优化的雷达滤波训练（含多普勒速度）
import torch
from torch.utils.data import DataLoader
from datasets.td import TrajectoryDataset
from data.traj import generate_dataset
from models.trajectory_mamba import TrajectoryMamba
import torch.optim as optim
import os
import json
import time
import numpy as np
from tqdm import tqdm  # 进度条显示
import multiprocessing as mp  # 多进程

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_data_parallel(params):
    """并行数据生成的worker函数"""
    return generate_dataset(**params)

def main():
    # ==================== GPU配置 ====================
    # 设置CUDA优化
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速
    torch.backends.cudnn.allow_tf32 = True
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU.")
    
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    out_dir = '3D_run/radar_filter_runs'
    os.makedirs(out_dir, exist_ok=True)

    # 初始化训练历史记录
    history = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': [],
        'train_pos_rmse': [], 'val_pos_rmse': [],
        'train_vel_rmse': [], 'val_vel_rmse': []
    }

    print("=" * 70)
    print("Generating radar dataset with Doppler velocity...")
    print("=" * 70)

    # 轨迹类型分布配置
    trajectory_distribution = {
        'uniform': 0.1,
        'accelerate': 0.1,
        'decelerate': 0.1,
        'turn': 0.1,
        'hover': 0.1,
        'mixed': 0.5
    }
    
    # ==================== 并行数据生成 ====================
    # 数据生成参数
    base_params = {
        'T': 200,
        'dt': 1.0,
        'noise_std_r': 5.0,
        # 方位角(az)和俯仰角(el)测量噪声，单位为度（generate_trajectory 内部会处理）
        'noise_std_az': 1.0,
        'noise_std_el': 1.0,
        'noise_std_vr': 0.1,
        'radar_pos': (0, 0, 0),
        'trajectory_distribution': trajectory_distribution
    }
    
    # 训练集：分批并行生成
    N_train = 10000
    N_val = 200
    num_workers = min(8, mp.cpu_count())  # 使用8个进程或CPU核心数
    batch_per_worker = N_train // num_workers
    
    print(f"\nGenerating training data using {num_workers} parallel workers...")
    train_params_list = []
    for i in range(num_workers):
        params = base_params.copy()
        params['N'] = batch_per_worker if i < num_workers - 1 else (N_train - batch_per_worker * (num_workers - 1))
        train_params_list.append(params)
    
    # 并行生成训练数据
    start_time = time.time()
    with mp.Pool(processes=num_workers) as pool:  # 多进程池
        train_data_batches = pool.map(generate_data_parallel, train_params_list)
    
    # 合并所有批次
    train_data = []
    for batch in train_data_batches:
        train_data.extend(batch)
    
    data_gen_time = time.time() - start_time
    print(f"Training data generated: {len(train_data)} trajectories in {data_gen_time:.1f}s")
    
    # 验证集
    print(f"Generating validation data...")
    val_params = base_params.copy()
    val_params['N'] = N_val
    val_data = generate_dataset(**val_params)
    print(f"Validation data generated: {len(val_data)} trajectories")

    # ==================== 数据集和DataLoader ====================
    seq_len = 20
    train_ds = TrajectoryDataset(train_data, seq_len=seq_len)
    val_ds = TrajectoryDataset(val_data, seq_len=seq_len)

    print(f"\n{'='*70}")
    print("Dataset statistics:")
    print(f"  - Training samples: {len(train_ds):,}")
    print(f"  - Validation samples: {len(val_ds):,}")
    print(f"{'='*70}")

    # DataLoader优化配置
    batch_size = 256  # 3090可以处理更大的batch size
    num_workers_loader = 4  # DataLoader的worker数量
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers_loader,  # 多进程数据加载
        pin_memory=True,  # 固定内存，加速GPU传输
        persistent_workers=True  # 保持worker进程活跃
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size * 2,  # 验证时可以用更大batch
        shuffle=False,
        num_workers=num_workers_loader,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"\nDataLoader configuration:")
    print(f"  - Batch size (train): {batch_size}")
    print(f"  - Batch size (val): {batch_size * 2}")
    print(f"  - Num workers: {num_workers_loader}")
    print(f"  - Batches per epoch: {len(train_loader)}")

    # ==================== 模型初始化 ====================
    model = TrajectoryMamba(
        d_model=64,
        n_layers=4, 
        seq_len=seq_len,
    ).to(device)
    
    # 跳过模型编译以避免Triton依赖
    print("🔧 使用标准训练模式（跳过torch.compile以兼容性优先）")
    
    total_params = count_parameters(model)
    print(f"\nModel configuration:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - d_model: 64")
    print(f"  - n_layers: 4")
    print(f"  - seq_len: {seq_len}")

    # ==================== 优化器配置 ====================
    # 检查是否支持fused优化器
    try:
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-5,
            fused=True  # 使用融合优化器（更快）
        )
        print("✅ Using fused AdamW optimizer")
    except:
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-5
        )
        print("⚠️  Using standard AdamW optimizer (fused not available)")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=20,  # epochs
        eta_min=1e-5
    )
    
    # 混合精度训练配置
    use_amp = False
    # 使用正确的autocast API
    try:
        from torch.amp import autocast
        autocast_context = autocast('cuda', enabled=use_amp)
        print("✅ Using torch.amp.autocast (new API)")
    except ImportError:
        from torch.cuda.amp import autocast
        autocast_context = autocast(enabled=use_amp)
        print("✅ Using torch.cuda.amp.autocast (legacy API)")
    
    # 梯度缩放器
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    print(f"\nTraining configuration:")
    print(f"  - Mixed precision (AMP): {use_amp}")
    print(f"  - Optimizer: AdamW")
    print(f"  - Learning rate scheduler: CosineAnnealing")
    
    # 损失权重
    pos_weight = 1.0
    vel_weight = 0.1

    epochs = 20
    best_val = float('inf')
    
    print(f"\n{'='*70}")
    print(f"Starting training for {epochs} epochs...")
    print(f"Loss weights: position={pos_weight}, velocity={vel_weight}")
    print(f"{'='*70}\n")
    
    # ==================== 训练循环 ====================
    for ep in range(epochs):
        epoch_start = time.time()
        model.train()
        
        # 训练统计
        train_loss_total = 0.0
        train_pos_mse = 0.0
        train_vel_mse = 0.0
        
        # 使用tqdm显示进度
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", 
                   leave=False, dynamic_ncols=True)
        
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)  # 异步传输
            yb = yb.to(device, non_blocking=True)
            
            # 混合精度前向传播
            with autocast_context:
                pred = model(xb)
                
                # 分离位置和径向速度（现在输出为 x,y,z,vr）
                pred_pos = pred[:, :3]
                pred_vr = pred[:, 3:4]
                yb_pos = yb[:, :3]
                yb_vr = yb[:, 3:4]
                
                # 计算分离损失
                loss_pos = torch.nn.functional.mse_loss(pred_pos, yb_pos)
                loss_vr = torch.nn.functional.mse_loss(pred_vr, yb_vr)
                loss = pos_weight * loss_pos + vel_weight * loss_vr
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 累积损失
            batch_size_curr = xb.size(0)
            train_loss_total += loss.item() * batch_size_curr
            train_pos_mse += loss_pos.item() * batch_size_curr
            train_vel_mse += loss_vr.item() * batch_size_curr
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pos': f'{np.sqrt(loss_pos.item()):.2f}m',
                'vr': f'{np.sqrt(loss_vr.item()):.3f}m/s'
            })
        
        pbar.close()
        
        # 学习率调度
        scheduler.step()
        
        # 计算训练集平均损失和RMSE
        n_train = len(train_loader.dataset)
        train_loss = train_loss_total / n_train
        train_pos_rmse = np.sqrt(train_pos_mse / n_train)
        train_vel_rmse = np.sqrt(train_vel_mse / n_train)
        train_rmse = np.sqrt((train_pos_mse + train_vel_mse) / n_train)

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss_total = 0.0
        val_pos_mse = 0.0
        val_vel_mse = 0.0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                
                with autocast_context:
                    pred = model(xb)
                    
                    pred_pos = pred[:, :3]
                    pred_vr = pred[:, 3:4]
                    yb_pos = yb[:, :3]
                    yb_vr = yb[:, 3:4]
                    
                    loss_pos = torch.nn.functional.mse_loss(pred_pos, yb_pos)
                    loss_vr = torch.nn.functional.mse_loss(pred_vr, yb_vr)
                    
                    loss = pos_weight * loss_pos + vel_weight * loss_vr
                
                batch_size_curr = xb.size(0)
                val_loss_total += loss.item() * batch_size_curr
                val_pos_mse += loss_pos.item() * batch_size_curr
                val_vel_mse += loss_vr.item() * batch_size_curr
        
        # 计算验证集平均损失和RMSE
        n_val = len(val_loader.dataset)
        val_loss = val_loss_total / n_val
        val_pos_rmse = np.sqrt(val_pos_mse / n_val)
        val_vel_rmse = np.sqrt(val_vel_mse / n_val)
        val_rmse = np.sqrt((val_pos_mse + val_vel_mse) / n_val)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['train_pos_rmse'].append(train_pos_rmse)
        history['val_pos_rmse'].append(val_pos_rmse)
        history['train_vel_rmse'].append(train_vel_rmse)
        history['val_vel_rmse'].append(val_vel_rmse)
        
        epoch_time = time.time() - epoch_start
        
        # GPU内存使用情况
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        # 保存最佳模型
        if val_loss < best_val:
            best_val = val_loss
            model_path = os.path.join(out_dir, 'model.pt')
            
            # 保存模型
            torch.save(model.state_dict(), model_path)
            
            meta = {
                'd_model': 64,
                'n_layers': 4,
                'seq_len': seq_len,
                'input_dim': 4,
                'output_dim': 4,
                'epoch': ep + 1,
                'best_val_loss': best_val,
                'best_val_rmse': val_rmse,
                'best_val_pos_rmse': val_pos_rmse,
                'best_val_vel_rmse': val_vel_rmse,
                'total_parameters': total_params,
                'batch_size': batch_size,
                'use_amp': use_amp,
                'pos_weight': pos_weight,
                'vel_weight': vel_weight
            }
            meta_path = os.path.join(out_dir, 'meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

        # 打印训练信息
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {ep+1:3d}/{epochs} | Time: {epoch_time:.1f}s | "
              f"LR: {current_lr:.2e} | GPU: {gpu_mem:.2f}GB")
        print(f"  Train: Loss={train_loss:.6f} | "
              f"Pos={train_pos_rmse:.2f}m | "
              f"Vel={train_vel_rmse:.3f}m/s")
        print(f"  Val:   Loss={val_loss:.6f} | "
              f"Pos={val_pos_rmse:.2f}m | "
              f"Vel={val_vel_rmse:.3f}m/s")
        
        # 定期保存历史
        if (ep + 1) % 5 == 0:
            history_path = os.path.join(out_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)

    # ==================== 保存最终结果 ====================
    model_last_path = os.path.join(out_dir, 'model_last.pt')
    torch.save(model.state_dict(), model_last_path)
    
    history_path = os.path.join(out_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"{'='*70}")
    print(f"Best validation results:")
    print(f"  - Total RMSE: {min(history['val_rmse']):.4f}")
    print(f"  - Position RMSE: {min(history['val_pos_rmse']):.2f} m")
    print(f"  - Velocity RMSE: {min(history['val_vel_rmse']):.3f} m/s")
    print(f"\nResults saved to: {out_dir}")

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()