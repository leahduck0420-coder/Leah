> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# FP32 累加契约与分段数值诊断

## 服务器记录

numericfix 的 smoke-0976bbb8ce06 有 15 个 PASS。剩余两项分别为 W7-S2 的 extreme/B0 超限 2 个元素，以及 W8-S4 的 extreme/B1 超限 18 个元素，K/V cache 分别超限 27/7 个元素。W8 B0 已通过，B1 重复执行一致且所有任务恰好执行一次。其 B2 尚未执行。

## 精度设置

原程序只设置 allow_tf32=False。PyTorch 2.8 的 FP16/BF16 GEMM 另有 reduced_precision_reduction 开关。此前没有完整记录和强制设置这些开关，因此尚不能从旧结果确认各路径使用一致的累加契约。

新 numeric.configure_matmul_precision 在环境探测、正式 worker、可选对照 worker 和 A0 worker 中逐进程设置：

- allow_tf32=False
- allow_fp16_reduced_precision_reduction=False
- allow_bf16_reduced_precision_reduction=False
- 当版本提供 allow_fp16_accumulation 时，设置为 False。

这些实际读回的值写入环境 JSON、协议和逐用例 JSON。Triton Qwen 路径的 GEMM 显式把 FP32 accumulator 传给 tl.dot，使用跨 K tile 的连续累加。该变更覆盖 W7 的 Gate/Up/Down 与 W8 的 QKV/Out，B1/B2 共用实现。

PyTorch 官方说明：
https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html
https://docs.pytorch.org/docs/2.8/backends.html

## 失败时保存的证据

W7 记录 Gate/Up 原生与 FP32 参考差异、SiLU 两种表达式差异、Down 原生与 FP32 参考差异。

W8 记录 V 投影的原生、FP32、CPU FP64 和先舍入后加 bias 结果的差异。若实际内核已执行，还会把 kernel V cache 与这些分段结果比较。失败内核的 IR/PTX/CUBIN 保存到 04_ir/<case>/failed_B1 或 failed_B2。

这些证据用于继续区分后端累加、舍入边界和实际 kernel 错误。逐元素容差按既有契约执行。改动是否消除剩余错误，需要 RTX 5090 复测。

## 服务器运行

在配置好 SEAM、PATH 和 CPATH 的终端，执行：

```bash
bash run_stage_a.sh --mode smoke --venv-path "$SEAM_VENV" --resume
```

保存该版新结果目录。若仍失败，提供对应 08_logs/cases/*.json，其中包含 numerical_diagnostics 和 matmul_precision。本版 GPU 结果为 SERVER_PENDING。
