# 雷达轨迹滤波（含多普勒速度）实验报告

## 1. 仿真目的

本实验旨在评估基于 Mamba（SSM）模型对雷达极坐标量测（含多普勒径向速度）进行轨迹滤波的能力。重点关注：
- 对三维轨迹位置（$x,\;y,\;z$）与径向速度（$v_r$）的估计精度；
- 不同噪声水平与序列长度（$\mathrm{seq\_len}$）等超参数对滤波性能的影响；
- 在六类运动范式（`uniform`, `accelerate`, `decelerate`, `turn`, `hover`, `mixed`）下模型的稳健性与泛化性能。


## 2. 仿真场景与数据生成

- 场景：单雷达观测 3D 无人机目标，雷达坐标系原点在地面（默认 $\mathbf{r}_{\mathrm{radar}}=(0,0,0)$）。
- 轨迹类型：`uniform`、`accelerate`、`decelerate`、`turn`、`hover`、`mixed`（混合分段）。
- 每条轨迹长度 $T$（默认 $200$ 步），时间步间隔 $\Delta t = 1.0\,$s（或记作 $dt$）。
- 观测量测：极坐标 $(r,\;\mathrm{az\_deg},\;\mathrm{el\_deg})$ + 多普勒径向速度 $v_r$，测量噪声可配置：$\mathrm{noise\_std\_r}$（米），$\mathrm{noise\_std\_az}$ / $\mathrm{noise\_std\_el}$（度），$\mathrm{noise\_std\_vr}$（m/s）。
- 数据生成脚本：`data/traj.py::generate_trajectory` / `generate_dataset`。


## 3. 实验思路与方法

1. 数据准备：按指定分布生成训练集与验证集（脚本 `t.py` 中并行生成训练数据）。
2. 模型训练：使用 `models/trajectory_mamba.TrajectoryMamba`，输入投影到 $d_{\mathrm{model}}$，核心使用 Mamba，输出为 $(x,\;y,\;z,\;v_r)$。训练时位置与速度损失按权重 $\mathrm{pos\_weight}$ 与 $\mathrm{vel\_weight}$ 分开加权。
3. 评估：对验证集中每条轨迹使用滑动窗口进行预测，计算位置 RMSE（欧氏距离，记作 $\mathrm{RMSE}_{\mathrm{pos}}$）与径向速度 RMSE（$\mathrm{RMSE}_{v_r}$）；并按轨迹类型统计平均 RMSE。
4. 可视化：绘制 3D 空间对比图（真实轨迹、量测点、预测轨迹）、时间序列及误差分析（脚本 `v.py` / `v_simple.py`）。


## 4. 实验配置（建议的对比试验）

对以下参数做网格化试验，并记录每次训练的验证集平均性能：

- 测试组 A（量测噪声强度）:
  - A1: $\mathrm{noise\_std\_r}=5.0$ m, $\mathrm{noise\_std\_az}=1.0^{\circ}$, $\mathrm{noise\_std\_vr}=0.1\,$m/s
  - A2: $\mathrm{noise\_std\_r}=10.0$ m, $\mathrm{noise\_std\_az}=2.0^{\circ}$, $\mathrm{noise\_std\_vr}=0.2\,$m/s

- 测试组 B（序列长度）:
  - B1: $\mathrm{seq\_len}=20$
  - B2: $\mathrm{seq\_len}=40$

- 测试组 C（速度权重）:
  - C1: $\mathrm{pos\_weight}=1.0$, $\mathrm{vel\_weight}=0.1$
  - C2: $\mathrm{pos\_weight}=1.0$, $\mathrm{vel\_weight}=0.5$

说明：每组可组合（例如 A1+B1+C1），建议每个组合训练 1 次并记录结果。


## 5. 结果记录（示例三线表）

下面给出每次试验的结果记录表（请将实际运行得到的数值填入表中）。三线表格式如下：

| 实验 ID | $\mathrm{noise\_r}$ (m) | $\mathrm{noise\_az}$ (deg) | $\mathrm{noise\_vr}$ (m/s) | $\mathrm{seq\_len}$ | $\mathrm{pos\_w}$ | $\mathrm{vel\_w}$ | 样本数 | 平均 $\mathrm{Pos\_RMSE}$ (m) | 平均 $\mathrm{Vel\_RMSE}$ (m/s) |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A1-B1-C1 | 5.0 | 1.0 | 0.1 | 20 | 1.0 | 0.1 | 200 | TO_FILL | TO_FILL |
| A2-B1-C1 | 10.0 | 2.0 | 0.2 | 20 | 1.0 | 0.1 | 200 | TO_FILL | TO_FILL |
| A1-B2-C1 | 5.0 | 1.0 | 0.1 | 40 | 1.0 | 0.1 | 200 | TO_FILL | TO_FILL |
| A1-B1-C2 | 5.0 | 1.0 | 0.1 | 20 | 1.0 | 0.5 | 200 | TO_FILL | TO_FILL |

说明：表中每行为一次完整训练-验证结果，"样本数"为验证集轨迹数或样本数视角；"TO_FILL" 为待填写的实际数值。


## 6. 结果分析（示例要点）

- 噪声敏感性：
  - 随着 distance / angle / $v_r$ 噪声增大（A1 -> A2），位置 RMSE 通常上升。角度噪声对水平定位影响更显著，距离噪声对整体偏差增加直接可见。
- 序列长度影响：
  - 增大 $\mathrm{seq\_len}$（B1 -> B2）通常有利于捕捉更长时间窗口的动态，从而在转弯或加减速段改善位置估计，但可能导致模型更难训练（计算量与记忆需求增加）。
- 速度权重影响：
  - 增大 $\mathrm{vel\_weight}$（C1 -> C2）会使速度 RMSE 降低，但可能以位置 RMSE 的轻微上升为代价；需根据应用侧重权衡。
- 不同运动范式：
  - `uniform`、`accelerate` / `decelerate`：模型对匀速段表现稳定；加减速段若序列包含足够历史信息则能较好恢复。 
  - `turn`：若角速度较大且采样不密，角度噪声会显著影响定位；混合段（`mixed`）反映模型在非平稳序列下的鲁棒性。


## 7. 可视化与诊断建议

- 对每条轨迹保存：空间对比（3D）、分量时间序列（$X/Y/Z$）、径向速度对比、误差时间序列。脚本：`v.py` 与 `v_simple.py`。
- 对于表现异常的轨迹，检查量测分布（角度偏移、异常 $v_r$）与模型预测差异，定位 failure mode（例如：强角速度 + 角度噪声 -> 丢轨）。


## 8. 复现实验步骤（命令示例）

- 生成训练并训练（建议在有 GPU 机器上运行）：

  python t.py

- 可视化训练与验证结果（CPU）：

  python v.py

- 快速单轨迹测试与可视化：

  python v_simple.py


## 9. 结论与后续工作

- 初步实验可帮助量化 Mamba 模型在三维雷达数据上的滤波性能。若期望进一步提升：
  - 使用更强的模型正则化或数据增强；
  - 在模型中引入明确的物理运动先验（如常加速度、CT 模型融合）；
  - 增加观测维度（多基站或多视角融合）以提升定位鲁棒性。


## 10. 基于训练产物的关键结果（从训练输出自动提取）

以下数值直接来源于训练输出文件 `3D_run/radar_filter_runs/meta.json` 与 `history.json`：

- 最佳验证指标（meta.json）：
  - best epoch: 20
  - 最佳验证总 RMSE: 18.8114
  - 最佳验证位置 RMSE: 18.8046 m
  - 最佳验证径向速度 RMSE: 0.5068 m/s

- 训练过程概览（history.json）：
  - 训练过程中位置 RMSE 从初期数百米收敛到约 19 m，径向速度 RMSE 从几十 m/s 收敛到约 0.5 m/s，表明模型成功学习到了位置与速度的联合映射并稳定收敛。
  - 验证损失与 RMSE 在若干 epoch 出现波动（可能与训练数据分布或部分轨迹的高难度场景有关），但最终指标稳定在上述水平。


## 11. 可视化产物清单与初步观察

脚本已在 `3D_run/run_plots` 与 `3D_run/radar_filter_runs/plots_doppler` 中保存若干可视化图像。以下为工作区中文件名清单（统计自当前工作区）：

- `3D_run/run_plots`（逐轨迹选择保存的图）：共 28 张，按轨迹范式统计:
  - uniform: 4 张
  - accelerate: 4 张
  - decelerate: 4 张
  - turn: 4 张
  - hover: 4 张
  - mixed: 8 张

  注意：代码意图为 mixed 保存 4 张、其他每类保存 2 张，但实际目录中 mixed 的图像数量为 8 张，其他类各有 4 张，说明脚本在当前运行时多次保存或参数被多次调用，建议检查重复保存逻辑或根据需要清理/重命名。

- `3D_run/radar_filter_runs/plots_doppler`（训练过程与示例对比）:
  - `loss_curve.png`：训练 / 验证损失与 RMSE 曲线（含分离的 pos/vel 指标）。
  - `combined_trajectories_3d.png`：三条最佳轨迹的合并 3D 对比图。
  - `trajectory_comparison_1.png`、`trajectory_comparison_2.png`、`trajectory_comparison_3.png`：脚本选择的若干轨迹的 2x2 综合对比图。

初步观察建议（基于文件名与统计信息）：
- mixed 类型保存数量更多，说明在混合场景下模型的表现可能更有代表性或在选择逻辑中优先被采纳，建议对这些图进行人工检视以识别混合段常见失败模式。
- loss 曲线与历史指标显示训练已收敛，但仍有波动，需结合按类型的平均 RMSE（见下一节）进行进一步诊断。


## 12. 建议的下一步分析（可自动化）

我可以为你执行或帮你实现下列自动化分析脚本（当前我不能直接在你的环境中运行代码，但可以修改/添加脚本）：

1. 按轨迹范式计算平均 RMSE：
   - 已在 `v.py` 中添加了 `compute_average_rmse_by_type(model, val_data)` 函数，运行 `python v.py` 会打印并返回每类轨迹的平均位置 RMSE 和平均速度 RMSE（建议把结果写入 `3D_run/run_plots/avg_rmse_by_type.json`）。

2. 统计并去重保存图像：
   - 编写小脚本统计 `run_plots` 中按 traj id 的重复图片并根据需要只保留最佳 rank 或按类别分目录保存，方便人工审阅。

3. 自动生成带数值注释的 PDF 报告：
   - 读取 `history.json`、`meta.json` 和 `run_plots` 中的图像，并生成一份包含表格、训练曲线与代表性 3D 图像的 PDF 报告（使用 matplotlib + reportlab 或者 pandoc）。


## 13. 是否需要我帮你：

- 自动化计算并填充实验报告表格（我将读取 `history.json` 与执行 `compute_average_rmse_by_type`，并把结果写回 `EXPERIMENT_REPORT.md` 与 CSV）;
- 生成按类别汇总的 PDF 报告;
- 清理 / 重命名 `run_plots` 中重复图片并将每类图像放到独立子目录。

请告诉我你希望我执行哪项（我将修改或添加对应脚本），或若你允许我写入 avg 结果文件我会继续将计算结果保存到工作区。

## 14. 模型与仿真中使用的变量、计算式与损失定义（汇总）

下面将项目中涉及的主要变量、量测与计算公式集中列出，便于复现与讨论。

### 14.1 主要变量说明

- $t$：离散时间步索引（整数）。
- $T$：单条轨迹的时间步长度（例如 $200$）。
- $\Delta t$（或 $dt$）：时间步长（秒），本项目默认 $\Delta t = 1.0\,$s。
- $\mathbf{p}[t]=\big(x_t,\;y_t,\;z_t\big)$：目标在全局笛卡尔坐标系下的真实位置向量（米）。
- $\mathbf{v}[t]=\big(v_{x,t},\;v_{y,t},\;v_{z,t}\big)$：目标在全局笛卡尔坐标系下的真实速度向量（米/秒）。
- $\mathbf{a}[t]=\big(a_{x,t},\;a_{y,t},\;a_{z,t}\big)$：目标的真实加速度向量（米/秒$^2$）。
- $\mathbf{r}_{\text{radar}}=(x_r,\;y_r,\;z_r)$：雷达位置（本项目默认 $(0,0,0)$）。

观测量（每步 $t$ 的量测，以极坐标给出，角度以度为单位）：
- $r_{\text{meas}}[t]$：测得的径向距离（米），包含噪声。
- $\mathrm{az\_meas\_deg}[t]$：测得的方位角（度），包含噪声（对应极坐标中的 bearing）。
- $\mathrm{el\_meas\_deg}[t]$：测得的俯仰/仰俯角（度），包含噪声（对应 elevation）。
- $v_{r,\text{meas}}[t]$：测得的径向多普勒速度（m/s），包含噪声。

在数据结构中：$\mathrm{measurements\_polar}.\mathrm{shape}=(T,4)$ 对应 $(r,\;\mathrm{az\_deg},\;\mathrm{el\_deg},\;v_r)$。

模型输入 / 输出：
- 输入（模型前端）：一个长度为 $L=\mathrm{seq\_len}$ 的量测序列，形状为 $(L,4)$，元素为 $(r,\;\mathrm{az\_deg},\;\mathrm{el\_deg},\;v_{r,\text{meas}})$。
- 模型输出（out\_head）：预测最后步的笛卡尔位置与径向速度：$\hat{y}=(\hat{x},\;\hat{y},\;\hat{z},\;\hat{v}_r)$. 

### 14.2 极坐标到笛卡尔的变换（用于可视化与目标真值对齐）

角度在计算时先转换为弧度：

$$\mathrm{az\_rad} = \operatorname{deg2rad}\big(\mathrm{az\_meas\_deg}\big),\qquad
\mathrm{el\_rad} = \operatorname{deg2rad}\big(\mathrm{el\_meas\_deg}\big).$$

对应的测量坐标（相对于雷达）为：

$$x_{\mathrm{meas}} = r_{\mathrm{meas}}\cos(\mathrm{el\_rad})\cos(\mathrm{az\_rad}) + x_r,$$
$$y_{\mathrm{meas}} = r_{\mathrm{meas}}\cos(\mathrm{el\_rad})\sin(\mathrm{az\_rad}) + y_r,$$
$$z_{\mathrm{meas}} = r_{\mathrm{meas}}\sin(\mathrm{el\_rad}) + z_r.$$ 

真实的相对向量为：

$$\mathbf{p}_{\mathrm{rel}}[t] = \mathbf{p}[t] - \mathbf{r}_{\mathrm{radar}} = (x_{\mathrm{rel}},\;y_{\mathrm{rel}},\;z_{\mathrm{rel}})^\top.$$ 

真实径向距离：

$$r_{\mathrm{true}}[t] = \|\mathbf{p}_{\mathrm{rel}}[t]\| = \sqrt{x_{\mathrm{rel}}^2 + y_{\mathrm{rel}}^2 + z_{\mathrm{rel}}^2}.$$ 

### 14.3 径向（多普勒）速度的计算方法（真值）

径向单位向量：

$$\mathbf{u}_r[t] = \frac{\mathbf{p}_{\mathrm{rel}}[t]}{\|\mathbf{p}_{\mathrm{rel}}[t]\|}.$$ 

真值径向速度：

$$v_{r,\mathrm{true}}[t] = \mathbf{v}[t]\cdot\mathbf{u}_r[t] = \frac{v_{x,t}x_{\mathrm{rel}} + v_{y,t}y_{\mathrm{rel}} + v_{z,t}z_{\mathrm{rel}}}{\|\mathbf{p}_{\mathrm{rel}}[t]\|}.$$ 

备注：当 $\|\mathbf{p}_{\mathrm{rel}}[t]\|$ 非常接近 $0$ 时，实际实现使用小量 $\epsilon$（例如 $10^{-10}$）避免除零。

### 14.4 测量噪声模型（在 generate_trajectory 中实现）

测量噪声采用独立加性高斯噪声：

$$r_{\mathrm{meas}}[t] = r_{\mathrm{true}}[t] + \eta_r[t],\quad \eta_r[t]\sim\mathcal{N}(0,\sigma_r^2),$$

$$\mathrm{az\_meas\_deg}[t] = \mathrm{az\_true\_deg}[t] + \eta_{\mathrm{az}}[t],\quad \eta_{\mathrm{az}}[t]\sim\mathcal{N}(0,\sigma_{\mathrm{az}}^2),$$

$$\mathrm{el\_meas\_deg}[t] = \mathrm{el\_true\_deg}[t] + \eta_{\mathrm{el}}[t],\quad \eta_{\mathrm{el}}[t]\sim\mathcal{N}(0,\sigma_{\mathrm{el}}^2),$$

$$v_{r,\mathrm{meas}}[t] = v_{r,\mathrm{true}}[t] + \eta_{v_r}[t],\quad \eta_{v_r}[t]\sim\mathcal{N}(0,\sigma_{v_r}^2).$$

其中 $\sigma_r=\mathrm{noise\_std\_r}$（米），$\sigma_{\mathrm{az}}=\mathrm{noise\_std\_az}$（度），$\sigma_{v_r}=\mathrm{noise\_std\_vr}$（m/s）。角度噪声以度加到角度上，再在需要时转换为弧度。

### 14.5 损失函数与评估指标（用数学式给出）

设模型在时间步 $t$ 的预测为

$$\hat{y}[t]=\begin{bmatrix}\hat{x}[t]\\\hat{y}[t]\\\hat{z}[t]\\\hat{v}_r[t]\end{bmatrix},\quad y[t]=\begin{bmatrix}x[t]\\y[t]\\z[t]\\v_{r,\mathrm{true}}[t]\end{bmatrix}.$$ 

定义位置均方误差（按样本均值）：

$$\mathrm{MSE}_{\mathrm{pos}} = \frac{1}{N}\sum_{i=1}^N \|\hat{\mathbf{p}}[t_i] - \mathbf{p}[t_i]\|^2 = \frac{1}{N}\sum_{i=1}^N \big((\hat{x}-x)^2+(\hat{y}-y)^2+(\hat{z}-z)^2\big).$$

定义径向速度均方误差：

$$\mathrm{MSE}_{v_r} = \frac{1}{N}\sum_{i=1}^N (\hat{v}_r[t_i] - v_{r,\mathrm{true}}[t_i])^2.$$ 

加权总损失（训练时最小化）：

$$\mathcal{L} = w_{\mathrm{pos}}\,\mathrm{MSE}_{\mathrm{pos}} + w_{v_r}\,\mathrm{MSE}_{v_r},$$

其中 $w_{\mathrm{pos}}=\mathrm{pos\_weight}$，$w_{v_r}=\mathrm{vel\_weight}$。

评价指标采用 RMSE：

$$\mathrm{RMSE}_{\mathrm{pos}} = \sqrt{\mathrm{MSE}_{\mathrm{pos}}},\qquad \mathrm{RMSE}_{v_r} = \sqrt{\mathrm{MSE}_{v_r}}.$$ 

### 14.6 误差计算（可视化中的时间对齐）

模型基于滑动窗口 $L$ 输出对应窗口末步的预测，因此第一个可预测的时间索引为 $t=L-1$。若数据集中共有 $M$ 个窗口样本，则 $N=M$ 用于上式求平均。


## 15. t.py 的详细实现与工作流程说明

下面给出 `t.py` 的逐步说明，包含关键实现细节、参数和数据流：

1. 环境与 GPU 配置
   - 打开适用的 PyTorch 后端选项：
     - torch.backends.cudnn.benchmark = True（启用自动算法搜索以加速卷积/SSM等）
     - 允许 TF32（torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32）以加速矩阵乘法（在支持的硬件上）。
   - 检查 CUDA 可用性：若不可用脚本会抛出错误（本训练脚本面向 GPU）。

2. 数据生成（并行化）
   - 使用 `data/traj.generate_dataset` 生成仿真轨迹。为了提高生成速度，脚本使用 Python 的 multiprocessing.Pool 并将生成任务切分为若干批次（每个进程生成 N_i 条轨迹）。
   - `base_params` 包含：T, dt, noise_std_r, noise_std_az, noise_std_el, noise_std_vr, radar_pos, trajectory_distribution（6类的概率分布）。
   - 合并每个进程返回的轨迹批次，得到训练集（N_train）与验证集（N_val）。

3. 数据集与 DataLoader
   - 使用 `datasets.td.TrajectoryDataset` 将轨迹转换为滑动窗口样本：
     - 输入：每个样本为 (seq_len, 4) 的极坐标量测子序列。
     - 目标：每个样本的 target 为 (x_t, y_t, z_t, vr_true_t) 对应窗口末步 t。
   - DataLoader 优化配置：
     - batch_size 设为较大值（例如 256）用于加速 GPU 训练；
     - num_workers（数据加载进程数），pin_memory=True（加速主机到 GPU 的数据拷贝），persistent_workers=True（提高多 epoch 重复加载效率）。

4. 模型初始化
   - 创建 `TrajectoryMamba(d_model, n_layers, seq_len)`：
     - 输入投影层（Linear 4 -> d_model）将极坐标量测映射到模型嵌入；
     - 可选的位置信息嵌入（pos_emb）用于短序列编码；
     - 核心为 Mamba（状态空间模型实现），输出 head 映射 d_model -> 4（x,y,z,vr）。

5. 优化器与调度
   - 尝试使用 fused AdamW（若可用）；否则回退到标准 AdamW；
   - 学习率调度器使用 CosineAnnealingLR（T_max = epoch 数）。

6. 混合精度（AMP）与梯度缩放
   - 可选启用 AMP（use_amp），脚本能够兼容不同 PyTorch 版本的 autocast API；
   - 若启用 AMP，使用 GradScaler 以稳定数值并加速训练。

7. 训练循环（每个 epoch）
   - model.train()，在每个 batch 上：
     - 将输入 xb, 目标 yb 拷贝到 device（GPU）；
     - 在 autocast 上下文中前向传播得到 pred；分离 pred_pos = pred[:, :3] 与 pred_vr = pred[:, 3:4]；
     - 计算 loss_pos = mse(pred_pos, yb_pos) 与 loss_vr = mse(pred_vr, yb_vr)；
     - loss = pos_weight * loss_pos + vel_weight * loss_vr；
     - 反向传播：若使用 AMP 则使用 scaler.scale(loss).backward() 等流程；否则标准 backward + optimizer.step()；
     - 梯度裁剪（clip_grad_norm_）以防发散；
     - 累积 batch 级别损失用于 epoch 统计并在 tqdm 中显示当前损失/子指标。

8. 验证阶段
   - model.eval()，在验证集上禁用梯度计算：
     - 在每个验证 batch 上做前向并累积 loss_pos、loss_vr，最后计算整个验证集的平均 loss（val_loss）与 RMSE 指标；

9. 模型与元数据保存
   - 若当前 val_loss 比 best_val 更好，保存 model.state_dict() 到 out_dir/model.pt，并写入 meta.json（包含 d_model、n_layers、seq_len、epoch、best 指标等）；
   - 每隔若干 epoch 或训练结束时将 history.json 保存到 out_dir。

10. 训练结束后
   - 保存最终模型 `model_last.pt`，以及 `history.json`；
   - 在控制台打印训练与验证的最终统计信息。


---

以上为完整变量、计算式与训练脚本实现细节的补充，若需要我可以：
- 把第 14 节的公式用 LaTeX 格式保存为单独的 PDF 参考手册；
- 在 `EXPERIMENT_REPORT.md` 中把训练中的关键代码片段（例如 loss 计算、数据生成关键段）以代码块形式嵌入以便快速审阅。
