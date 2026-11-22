> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 原始记录和结果表

主键为 Case ID 与 Candidate ID。Case 哈希覆盖工作负载、目标策略、精度、形状和全部参数。B0/B1/B2 在同一 Case 内共享输入。签名覆盖源码、环境、权重哈希和测量协议，用于恢复。统计时仅接纳两个编译产物存在、真实 CUDA 执行、所有预检通过、每个基线样本齐全且确有变换的行。NO_OP 和预期拒绝保留审计行。

00_environment 记录环境、配置、协议、微基准和计划分母。01_correctness 记录各模式误差与独立合法性检查。02_performance 保存每批样本、分位数与增量收益。03_cost 保存按来源分类的成本、资源、上下文和静态/动态 DAG 特征。04_ir 保存 B1/B2 计划、OwnershipMap、DDG、实际编译中间产物和差异。05_analysis 为八幅实际数据图；无证据时给出原因。06_reports 用 PASS/PARTIAL/FAIL 表示整轮覆盖。07_dataset 中 CSV、schema 和 manifest 可直接用 Python、R 或 Excel 读取。08_logs/cases 是逐案例证据，包含可选基线结果、执行次序、插桩和异常。

建模时先筛选 status=PASS。标签 gpu_latency_us 是 B2 延迟。按 shape / workload 分组留出；同一配置的批次不可跨训练/验证泄漏。本阶段没有拟合或执行联合模型。硬件计数器缺失值不能填为零。训练时应对不同的可用性设置特征掩码。
