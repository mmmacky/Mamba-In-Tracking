# Mamba架构
Mamba 的核心是 Selective SSM（S6）+ 门控机制 + 硬件感知的并行扫描算法。
1. 从传统 SSM 开始
Mamba 的基础是状态空间模型（State Space Model），其连续形式为：
\[
\begin{aligned}
h'(t)&=Ah(t)+Bx(t)\\
y(t)&=Ch(t)
\end{aligned}
\tag{1}
\]

其中：
- \(x(t)\)：当前输入。
- \(h(t)\)：隐藏状态，负责存储历史信息。
- \(A\)：状态转移矩阵，决定历史信息如何演化。
- \(B\)：输入投影矩阵，决定新信息如何写入状态。
- \(C\)：输出投影矩阵，决定从隐藏状态中读取什么信息。
2. 离散化后进行序列建模
实际处理离散序列时，SSM 通过离散化转化为递归形式：
\[
\begin{aligned}
h_t&=\bar A h_{t-1}+\bar Bx_t\\
y_t&=Ch_t
\end{aligned}
\tag{2}
\]

其中 \(\bar A,\bar B\) 是离散化后的参数。
这和 RNN 很相似：通过隐藏状态 \(h_t\) 将历史信息传递到下一时间步，而不需要像 Transformer 一样显式计算所有 Token 之间的注意力。

S6 使用输入依赖的步长 \(\Delta_t\) 进行离散化：
\[
\begin{aligned}
\bar A_t&=\exp(\Delta_t A)\\
\bar B_t&=(\Delta_t A)^{-1}
(\exp(\Delta_t A)-I)\Delta_t B_t
\end{aligned}
\tag{4}
\]

这里给出的是零阶保持（ZOH）离散化的理论形式；实现时可以采用等价的数值方法或简化计算。
因此，状态更新变成：
\[
\begin{aligned}
h_t&=\bar A_t h_{t-1}+\bar B_t x_t\\
y_t&=C_t h_t
\end{aligned}
\tag{5}
\]

这就是 S6 的本质：根据当前输入内容，动态决定保留哪些历史信息、写入哪些新信息。
# 轨迹去噪与状态估计（基于 Mamba）

本实验目标为去噪/状态估计：给定时刻 t 的带噪测量，预测时刻 t 的真实位置。


在仓库根运行：
```powershell
python .\exp2\exp3\train.py
python .\exp2\exp3\eval.py
python .\exp2\exp3\visualize.py
```
### train.py
计算可学习的参数数量：权重、偏置、卷积核，筛选出需要计算梯度的参数（要训练的）  
将张量保存在输出目录中(.pt)  
history是训练历史记录  
生成数据：  
1. 训练数据：500个样本，每个样本100个时间步  
2. 验证数据：50个样本  
创建数据集：  
将train_data和val_data转换为模型可直接读取的模式，提供按索引访问数据的接口：  
1. train_data中包含极坐标量测信息  
2. seq_len=50 用连续50个时间步的量测数据作为输入，每个长度为50的序列都对应一个标签，即最后一个时刻的目标笛卡尔坐标  
创建数据加载器  
设置计算设备  
初始化模型  
设置优化器与损失函数  
训练参数  
epoch  
best_val最佳验证损失  
开始训练循环(每个循环是一个epoch)：  
设置为model中的训练模式(model.train)  
在训练一个epoch就用验证集进行一次评估model.eval,此时不再更新参数所以关闭梯度计算  
程序入口

