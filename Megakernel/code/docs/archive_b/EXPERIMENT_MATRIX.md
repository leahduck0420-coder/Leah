> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 区域与实验矩阵

Full 在每类工作负载中按历史增量收益选最优、中位和最差，再按共享内存与 spill 的压力指标选一点。随后补齐可用 dtype、阶段、DAG、归一化、布局和消费者。相同配置合并。历史结果用于确定代表性区域，不能据此推断总体无偏收益。当前输入得到 35 个区域、672 个唯一计划、2,565 条静态合法历史。静态计数含无策略根计划。Smoke 独立选取 8 个区域。

| 工作负载 | 可改变的实际位置 |
|---|---|
| W1 Bias/GELU/Residual | Bias 到 GELU 同 Program 数据交接，按布局核对 shuffle/shared 指令 |
| W2 GEMM epilogue | dot 到 bias 的 SSA 交接、输出列重分块 |
| W3 Norm 消费 | 行分组、归一化中间交接、行统计重建、重建端新产生的统计量边 |
| W4 Softmax | 行统计和 weighted sum 的切片域，scale 只使用适用的动作 |
| W5 Attention | softmax 统计、PV 归约和行内统计读取作用域 |
| W6 任务 DAG | 屏障放宽、Worker 数，依赖及执行完成次数 |
| W7 Qwen MLP | gate/up 两条边、行分组、RMS 统计，支持在两个位置重复应用 S1 |
| W8 Qwen Attention | 14 个查询头的任务分组、GQA、统计重建、PV 切片与就绪调度 |

P0 包括 S1+S2、S2+S3、S3+S4、S4+S5。P1 包括 S1+S3、S2+S5。所有十类异策略对都有按区域的适用性统计，合法的同策略多位置组合另存。T1 在 W3/W7，T2 在有真实调度位置的 W8 检查。W5 每个 query 单独执行，缺少跨任务发布。W8 的 S2+S4+S5 可以组合分头、头内切片与调度。需要每个归约切片作为独立任务发布 partial 的 T3 另标 NOT_APPLICABLE。详见 target_coverage.csv。
