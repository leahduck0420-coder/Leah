> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 数据关联与验收

00_environment 保存输入哈希、协议签名、环境对照及导入审计。01_legality、03_multistep 记录每个动作与前驱计划、重新验证结果及快照。02_pairwise 包含五组 factorial 对照与顺序观测。04_search 含选择历史、各预算和有限 Oracle。05_cost 与 06_ir 可取回资源、作用域、完整 IR 和逐前缀差异。08_reports 保存验收，09_dataset 提供含 Stage A 列的训练表、schema 和 SHA256。实际原始批次、负例、编译成本及完整校验在 10_logs/records/<region>/<plan>.json。

主表每个区域的唯一终态一行。相同计划的顺序历史、搜索预算和选中次数另用多对一表关联。Stage C 的监督标签应限定 correctness_status=PASS 且 measurement_status=MEASURED；保留回退。建议用 workload、功能 shape 和运行组划分训练/验证，避免同一计划的别名路径跨集合。UNAVAILABLE 表示缺少证据，NOT_APPLICABLE 表示无对应语义。合法性表可保留无法生成终态的尝试。

只有未过滤 Full 满足输入、环境、策略和组合、两种实际顺序证据、搜索预算、消融、Qwen、编译资源、图表和 dataset 等门禁，才能 PASS。输入/环境或数值失败会记录 FAIL；覆盖、实现、资源或其他门禁不足记录 PARTIAL，并给出 failed_gates。Smoke 有独立退出条件。失败或无数据仍输出可读的状态和缺数据说明。

Stage C 的输入特征应按决策时刻筛选。Oracle 时间、regret、目标执行时间、实测加速和插桩结果属于标签或事后分析；有实测编译成本的流程才可使用编译后资源元数据。
