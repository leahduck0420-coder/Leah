> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# Q/K/V 投影数值资格检查

## 服务器依据

accfix 的 seq=64 用例在 K cache 和最终输出上超限。随后统一采用 GEMM 后舍入、加 bias 后再舍入的 biasfix 使 W8-S4 通过，但 seq=16 的 W8-S3/S5 在 B0 检查出现新回归。

seq=16/S5 的 native V 相对 FP32 单次舍入 RMS 为约 5.44e-6，相对两段舍入为约 2.2867e-4。S3 的对应值约为 4.51e-5 和 2.3317e-4。由此排除所有配置统一采用两段舍入的做法。当前数据表明，原生低精度 linear 的数值行为需要按实际投影配置检查。

## 本版行为

1. 首先执行原有静态计划合法性检查。
2. 每个 W8 用例只用该用例的 normal 输入和相同权重进行一次数值资格检查。分别计算 native linear、FP32 GEMM 后加 bias 的单次舍入结果、GEMM 先舍入再加 bias 的两段舍入结果。
3. 对 Q、K、V 分别选择与 native 输出均方误差更小的模式。平局选择单次舍入。该过程使用投影数值，记录其耗时，不读取性能计时结果。
4. Q、K、V 的选择编码为 3 位 mask。B1 和 B2 以及 extreme/normal 检查、正式计时、单独插桩均固定使用同一个 mask。
5. 独立参考按选中边界执行 FP32 矩阵乘与显式转换，softmax 使用已修复的 FP64 移位参考。B0 继续运行原生 linear。所有全图输出、cache、重复性、任务完成检查按既有容差执行，失败继续报告 FAIL_CORRECTNESS。

该步骤用于兼容原生算术边界。它是逐用例数值资格检查，使用当前用例的 normal 数据；不能据此宣称与 native 在所有输入上逐位等价。

## 结果字段

- qkv_projection_contract_version 为 native_qualified_qkv_bias_v2。
- 每个用例的 qkv_projection_qualification 保存 mode、Q/K/V 两种误差、exact 标记、probe profile 和 elapsed_seconds。
- B1_plan.json 与 B2_plan.json 的 proof 保存相同 qkv_bias_mode。
- mode 的 bit 0/1/2 分别控制 Q/K/V 的 GEMM 后是否先转换为 FP16/BF16。
- CUBIN 的特化参数携带该 mask，launch/replay 内不重新探测。

## 复测

在配置好 SEAM、PATH、CPATH 的服务器终端执行完整 smoke：

```bash
bash run_stage_a.sh --mode smoke --venv-path "$SEAM_VENV" --resume
```

提供 17 项状态和 Acceptance。若 W8 仍失败，其 JSON 同时包含数值资格检查和分段误差。本版 Linux Triton 3.4.0 的实际 GPU 验证为 SERVER_PENDING。
