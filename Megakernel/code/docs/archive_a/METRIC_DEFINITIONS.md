> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 指标与标签

H3 报告包含读入、交接、GELU 与写回的完整模板耗时。单 warp 的跨 lane 对照要求 PTX 出现 shuffle，跨 warp 对照要求共享内存指令。两者的 B1 与 B2 使用相同 warp 数。每条 PTX 指令的周期数仍属未测。

对就绪调度，cross_program_dependencies 计不同逻辑任务间的数据边，execution_constraint_edges 另计执行顺序约束。instrumented_cross_worker_edges 用一次独立插桩中真实执行两个端点的 worker ID 计算。S3 的 estimated_operand_bytes 是重建所读原输入或已存 logits 的逻辑字节数。可选 Proton 在独立进程对一个已通过正确性检查的 B2 采集 CUPTI 活动树，保存在对应案例 IR 目录；不可用时记录具体原因。

计时单位为微秒。主要测量以同一 stream 的 CUDA Events 包围预先捕获的 CUDA Graph，输出为除以图内调用次数的平均单次耗时。Graph 不可用时改用相同重复数的 eager CUDA Events，并记录方法。两种模式均先 JIT、预热、同步。调度状态复位属于操作开销。Nsight 和 Proton 均非必需；正式测量不使用硬件计数器。A0 单独探测 do_bench_cudagraph。

- MEASURED：逐批设备时间、主机查询到的频率/功耗、独立插桩的任务起止时间和 worker 编号。ready_gap_ns 是数据已就绪到任务开始的间隔，包括调度和其他工作，不能解释成硬件 stall。
- COMPILER_REPORTED：寄存器/线程、静态 shared、驱动报告 n_spills（原始数量，不能解释为动态流量）、IR 与 CUBIN 哈希、静态 PTX 指令出现次数。
- STATIC_ESTIMATE：逻辑操作数、有效元素覆盖、张量级操作量和流量。flops_main_matmul 只累计主矩阵乘；算术近似不含地址计算和超越函数的指令展开。logical_io_lower_bound_bytes 描述给定逻辑边界的输入输出量；缓存、transactions 和调度元数据另计。逻辑字节不能代替 DRAM 流量。
- DERIVED：配对速比、节省时间、变异及 CUDA driver 的 cuOccupancyMaxActiveBlocksPerMultiprocessor 计算的理论 occupancy。occupancy 使用实际 CUfunction、动态 shared 和每 CTA 线程数。API 不可用时保留 UNAVAILABLE。
- UNAVAILABLE 表示适用但没有可靠来源；NOT_APPLICABLE 表示不适用。零只表达已知为零的量。

每配置给出中位数、P10/P90、均值、样本标准差及确定性 bootstrap 95% CI。增量速比为两中位数之比；paired_speedup_ci95 对同一交错批的比值中位数进行重采样，二者 estimand 不同。跨配置使用几何平均，退化率为增量速比小于 1 的有效配置占比。静态资源与收益的相关性图仅表明关联。H2 的 cold 表示定时前使用超过 5090 L2 容量的工作集施压；不声称逐 cache line 已驱逐。
