> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 矩阵与时间窗

完整模式针对五条策略分别抽取 64、96、84、76、72 个通过静态规划的配置。另保留无效和 NO_OP 控制。seed=20261119 的清单在 validation/matrix.json 中给出。参数由 stage_a/matrix.py 的 axes 管理，按各层取值的已采样次数选取，候选 ID 为实际参数的内容哈希。所有策略单独执行。Smoke 覆盖 17 个 workload/strategy 对。实际 GPU 覆盖低于目标时保留真实分母。

| 负载 | 计算、参数轴与策略 |
|---|---|
| W1 | Bias → 精确 GELU → Residual；4K、64K、1M、16M 与尾块；tile 128–1024，thread/lane/warp/跨 program |
| W2 | GEMM → Bias → GELU；M 16/128/512/1024 加尾块，N/K 512/2048/4096；BM/BN 32/64/128，BK 32/64，4/8/16 warps；S1、S2 |
| W3 | RMSNorm 或 LayerNorm → scale 或多输出线性投影，每个消费者有独立向量；行数 1–1024，隐藏维 256–8192，消费者 1/2/4/8；S1、S2、S3 |
| W4 | 稳定 Softmax → scale／加权归约；宽 128–4096 加尾部，片长 32–512；S3、S4 |
| W5 | 单头 QKᵀ → 稳定 Softmax → PV；batch 1/4，序列 128–2048 加尾块，维 64/128，因果与非因果；S3、S4。每个 B1/B2 对固定相同的归一化路径，覆盖整行及每 32 项更新最大值和指数和的在线算法。上游 FlashAttention 作为独立可选对照 |
| W6 | 线性、分叉汇合、多生产者、混合负载 DAG；16/64/256 个任务，1–16 workers，深度轴 2–16。链和单次扇入图记录其真实深度，消除重复参数 |
| W7 | 官方 896/4864 维 post-attention RMSNorm → 双投影 → SwiGLU → Down。batch 1/2，token 1/8/16/32/65/128；S1、S2、S3 |
| W8 | 注意力模块输入 → 带 bias 的 Q/K/V 投影 → RoPE → 14:2 GQA 因果注意力 → 无 bias 输出投影；prefill 全序列、decode 单 token。KV 前缀为确定性 post-RoPE 历史，当前 token 实际更新 cache。S3、S4、S5 |

W7/W8 的输入是对应子图边界激活。W8 使用 self-attention 的输入契约，不含 block 外的输入归一化与残差。历史 KV 张量属于测试输入，decode 不包含构造历史的时间。随机权重和真实权重均测同一子图，不能解释为整模型推理。

A0 检测环境及 H1–H6；A1 对每个候选先做数值和版本检查；A2 随机交错 B0/B1/B2；A3 导出编译产物与单独的调度插桩；A4 生成分层统计和 F1–F8；A5 按实际覆盖、数值与资源证据验收。Nsight Compute 不属于工具链。
