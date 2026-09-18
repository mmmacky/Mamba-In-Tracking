import torch
import os

# 1. 路径部分保持不变（这部分已经是正确的了）
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '3D_run', 'radar_filter_runs', 'model.pt')

print(f"正在尝试加载文件: {file_path}")

try:
    # 2. 修改这里：添加 map_location=torch.device('cpu')
    # 这也顺便解决了 weights_only 的警告
    model = torch.load(
        file_path, 
        map_location=torch.device('cpu'),  # 关键修改：强制映射到 CPU
        weights_only=False
    )
    
    print("\n成功加载模型！")
    print("-" * 30)
    print("Model architecture:")
    print(model)

    if hasattr(model, 'state_dict'):
        print("\nState dictionary keys:")
        for key in model.state_dict().keys():
            print(key)
            
except Exception as e:
    print(f"\n发生错误: {e}")