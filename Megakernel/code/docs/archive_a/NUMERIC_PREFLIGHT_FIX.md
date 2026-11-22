> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 数值预检修复

## 服务器事实

PAD 修复后，20 个 A0 条目完成，17 个 smoke 用例中 13 个 PASS。四个 FAIL_CORRECTNESS 全部发生于 extreme 档位的 B0 对独立参考检查，B1/B2 尚未执行：

- W7-S1：82 个 Inf、82 个超限。
- W7-S2：2617 个 Inf、2617 个超限。
- W7-S3：82 个 Inf、82 个超限。
- W8-S4：25 个超限，最大绝对误差 1.0，RMS 误差约 0.01267785，无 Inf/NaN。

## 输入契约

原 extreme 生成器向激活和全部合成权重都注入固定周期的正负极值。在 W7 的两次投影、SwiGLU 乘法和 Down 叠加后，这种相关权重模式可使最终结果超出 FP16 有限范围。新的输入契约 activation_stress_fixed_model_weights_v2 为 W7/W8 使用同一 seed 的合成权重，normal/extreme 档位的权重相同；极值注入作用于输入激活与输入 KV cache。实际官方权重由原有加载路径提供。

协议 JSON 和逐用例 JSON 保存输入契约版本，以区分旧失败记录。主结果需使用新输出目录。原逐元素容差以及所有非有限值拒绝规则继续用于验收。

## 独立参考

原 Attention oracle 用 FP32 的 exp(logits-logsumexp(logits))。大 logit 的共同偏移会消耗 FP32 有效位，logsumexp 再被减回时可能导致概率和偏离 1。新参考先转 FP64、减行最大值、计算 exp 和 sum，再做除法，最后按原概率存储边界转回 FP16/BF16。B0 使用原本的 torch.softmax 路径。

正确性记录增加 reference_nan 和 reference_inf，区分执行结果与参考本身的非有限值。

## CPU 验证与服务器复测

CPU 回归调用实际 Workload 构造入口，在 NumPy 后端验证 normal/extreme 权重一致、激活压力值生效，并重现相关极值权重导致超出 FP16 范围。softmax 回归重现 FP32 大偏移消减并验证移位 FP64 参考及因果 mask。

服务器 Triton 3.4.0 下重新执行完整 smoke：

```bash
bash run_stage_a.sh --mode smoke --venv-path "$SEAM_VENV" --resume
```

检查 17 个逐用例状态、06_reports/acceptance_report.json 的 missing_pairs 和 core_failures_absent。Smoke 本身范围有限，formal_scope 和 qualified_range 不满足正式 full 覆盖时会报告 PARTIAL。若有数值失败，提供 08_logs/cases/ 对应 JSON。本版数值修复在 RTX 5090 上仍为 SERVER_PENDING。
