> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# H6 首次启动停滞的针对性修复

服务器 micro-5b8fb9eaf8ea 的前 18 项完成。H6.fixed 的 Python 堆栈停在首次 dispatch 启动后的 torch.cuda.synchronize，随后被 120 秒 watchdog 终止。由此确认该次首次 GPU 执行未完成，具体的 warp/同步触发机制尚未由设备端证据证实。

W6 的每个任务处理 32 个元素。此次将 W6 的 dispatch 配置固定为 num_warps=1，使 readiness 归约、任务 CAS 和屏障都在同一个 warp 的 CTA 内执行。B1 和 B2 使用同一配置；任务数、worker CTA 数和 fixed/ready 的依赖表按原计划进入该内核。W8 的配置另行验证。

A0 在第一次同步前将 TTIR、TTGIR、LLIR、PTX、CUBIN 与 launch metadata 写入该项的 artifacts 目录。这样后续超时可以核对实际 Triton 3.4.0 产物。H6 首次执行还要验证每个 task 的 Count=1、State=2，并与独立数值参考比较，通过后才生成计时样本。

## 服务器复测

沿用 A0_RECOVERY.md 的 SEAM、PATH 和 CPATH 设置，在本包目录依次执行：

```bash
bash run_stage_a.sh --mode micro --micro-item H6.fixed --venv-path "$SEAM_VENV" --resume &&
bash run_stage_a.sh --mode micro --micro-item H6.ready --venv-path "$SEAM_VENV" --resume
```

两个单项均为 PASS 后，再运行整个 A0：

```bash
bash run_stage_a.sh --mode micro --venv-path "$SEAM_VENV" --resume &&
bash run_stage_a.sh --mode smoke --venv-path "$SEAM_VENV" --resume
```

若任一项超时，按终端打印的结果目录提供 08_logs/micro/H6.fixed.log 或 H6.ready.log，以及对应 .artifacts/kernel.ptx 与 .artifacts/kernel.ttgir。核对实际产物时以 .artifacts/metadata.json 的 num_warps 为准。

## 验证范围

新增 CPU 回归通过实际 Workload.bind 入口验证 B1/B2 均以一个 warp 启动，worker CTA 数分别为 1、4、16，且每次执行调用 reset。离线验证使用 Windows Triton 3.9.0 为 16/64/256 个任务生成 SM120 CUBIN，记录于 validation/h6_offline_sm120.json。服务器 Linux Triton 3.4.0 的 GPU 正确性、终止性及性能仍为 SERVER_PENDING。
