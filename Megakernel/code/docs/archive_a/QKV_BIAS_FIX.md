> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# QKV 投影的 bias 舍入边界

## 服务器记录

accfix 的 smoke-9a40f753c3f6 为 16/17 PASS。剩余 W8-S4 的 B0 通过，B1 extreme 超限 8 个元素；K cache 超限 6 个元素，V cache 超限 0。重复执行一致，任务恰好完成一次。四个 matmul 精度开关均记录为 False。

分段记录中，kernel V 相对 CPU FP64 的 RMS 误差约 1.1262e-5，native V 相对 FP64 为 2.3238e-4；先把 GEMM 结果舍入再加 bias 的参考相对 FP64 为 2.3238e-4。这组数据提示 native bias 路径中存在中间舍入，但旧记录没有给出 native 与分段参考的直接逐位比较，因此该解释仍需本版新增字段验证。

## 实现契约

新 QKV 内核和独立参考明确采用两个存储精度边界：

```text
FP32 GEMM accumulator -> FP16/BF16 projection
projection + bias in FP32 -> FP16/BF16 biased projection
RoPE with the existing multiplication/addition boundaries
```

原内核将 acc+bias 一次舍入为 FP16/BF16。K 的 RoPE 存在乘积相减，输入的舍入差异可以在接近零的位置放大相对误差。

B0 继续调用现有的原生 F.linear(x,weight,bias)。独立参考使用 FP32 矩阵乘，显式舍入后加 bias。B0 与该参考需先通过既有逐元素容差，再运行 B1/B2。协议和用例记录 qkv_projection_contract_version=gemm_dtype_bias_dtype_v1。

新增诊断 v_native_vs_round_before_bias 和 v_native_round_before_bias_bitwise，直接验证 native 的实际舍入行为。原有 FP64 诊断提供数学误差对照。不能只用最终 PASS 推断所有中间阶段逐位相同。

## CPU 验证与复测

CPU 回归使用有效 FP16 输入 1.0009765625、相同权重和 bias=0.00048828125，证明单次舍入结果为 1.0029296875，两次舍入为 1.001953125，差异为一个 FP16 ULP。实际 reference helper 还验证三维 batch 形状和输出 dtype。

服务器先跑完整 smoke，并记录 17 个状态：

```bash
bash run_stage_a.sh --mode smoke --venv-path "$SEAM_VENV" --resume
```

该版在 Linux Triton 3.4.0、RTX 5090 上仍为 SERVER_PENDING。若失败，提供新 08_logs/cases/W8-S4-58b1260d138e27.json。
