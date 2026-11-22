> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# Stage A 输入与可信范围

交付基线为 stage_a_megakernel_rtx5090_qkvfix_20261119.zip。文件 SHA256 为 44eb215f49e0aa57b971527b21c5f6055c0669ca01953a03f0b48912f3ac6275。包内 stage_a/ 和 configs/ 的源码及配置与其原始字节一致。协议源码指纹为 56cad2d2221de752dd6b4a5fafe4a27abb666be95250e35754c79be6e13cef2a。

默认输入为 stage_b_inputs/stage_a_results 下唯一的运行，多个 manifest 时应指定确切目录。适配改名文件可以在该运行根下提供 stage_b_import_paths.json，键见 stage_b/importer.py 的 REQUIRED，值为输入文件相对路径。未提供映射时按标准路径及唯一路径名查找。

检查 Full、过滤标志、schema、CSV 内容 SHA256、每例 ID、矩阵参数、原始批次数及中位数、B0/B1/B2 的数值配置和重复执行、编译哈希、资源、OwnershipMap、DDG 及重新生成的单策略计划。历史的失败总裁决和逐例合格数量均保留。缺少 04_ir 时单列说明，能读取的历史 IR 要与元数据哈希一致。缺少逐例日志时只能使用记录齐全且能独立交叉核对的表格。版本无法确定时标记 UNAVAILABLE，当前编译缓存或时间不会倒填为历史数据。

附带索引记录来自 run 8054c1912f84dd7f。364 个合格配置中 S1/S2/S3/S4/S5 分别有 62/87/78/65/72 个。另有 26 个 FAIL_CORRECTNESS、2 个 COMPILE_ERROR、7 个 NO_OP、1 个 INVALID_REJECTED。正式执行时仍重新读取并验证用户提供的完整输入。
