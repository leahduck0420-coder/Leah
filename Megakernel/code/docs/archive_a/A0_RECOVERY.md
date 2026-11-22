> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# A0 前置微基准恢复

修复版本为 20261119-a0fix。服务器原版在生成 matrix.json 后持续运行约 38 分钟，未生成 microbench.json，用例测量尚未开始。GPU 后来已恢复空闲。现有日志只能把问题定位到 A0，具体内核仍为 SERVER_PENDING。

## 改动

- A0 分成 20 项，每项在独立 Python 进程中初始化 CUDA、预热并测量。正式计时仍使用原协议的 CUDA Events 和 CUDA Graph。
- 每项打印 START、END 和耗时。请求、日志、结果保存在 08_logs/micro/。progress.json 在启动前写入，microbench.json 每完成一项更新。
- 单项默认超时 120 秒，包含该子进程的初始化和编译。到上限前的 80% 时间点，faulthandler 保存 Python 堆栈。
- Linux 超时或用户中断时终止该 worker 的进程组，并等待进程退出。TIMEOUT 如实保留，随后 A0 标记 BLOCKED，该轮 smoke/full 的 GPU 用例也会保留 BLOCKED。
- 普通 Python/编译错误保存 FAILED 和 traceback。明确的 CUDA 设备错误会阻止该轮后续 GPU 测量。失败不会输出成功性能样本。
- 删除每项旧结果文件后再启动 worker，避免 --resume 时误用过期的 MEASURED。
- 将 PyTorch 裸 UUID 转为 nvidia-smi 所需的 GPU-UUID。缺少 UUID 时按 CUDA_VISIBLE_DEVICES 解析设备顺序。
- 兼容 CUDA Version 和 CUDA UMD Version 驱动横幅字段。

## 服务器运行

使用 SEAM 的 Triton 3.4.0 / PyTorch 2.8.0+cu128，以及 nvcc 12.8.93。

```bash
base=/das/t01/homes/liqi/Leah/program_linear_layout
export SEAM_VENV="$base/seam_ll_package_v1.0/.venv"
cuda_tools="$HOME/.local/seam-cuda-12.8.1"
export PATH="$cuda_tools/cuda_nvcc/bin:$cuda_tools/cuda_cuobjdump/bin:$cuda_tools/cuda_nvdisasm/bin:$PATH"
pydev="$HOME/.local/seam-python3.12-dev"
export CPATH="$pydev/usr/include/python3.12:$pydev/usr/include${CPATH:+:$CPATH}"
export CUDA_VISIBLE_DEVICES=0

# 在修复包的 stage_a_megakernel 目录执行。
bash scripts/preflight.sh --device cuda:0 --venv-path "$SEAM_VENV" &&
bash run_stage_a.sh --mode micro --device cuda:0 --venv-path "$SEAM_VENV" \
  --micro-timeout-seconds 120 --resume
```

A0 成功后运行 smoke，再运行 full。每种模式使用自己的输出目录。

```bash
bash run_stage_a.sh --mode smoke --device cuda:0 --venv-path "$SEAM_VENV" --resume
# smoke 主验收通过后
bash run_stage_a.sh --mode full --device cuda:0 --venv-path "$SEAM_VENV" --resume
```

若 A0 失败，首先记录终端最后一项的 ID 和打印的结果目录。例如最后一项为 H6.ready：

```bash
cat results/micro-实际签名/08_logs/micro/progress.json
tail -n 100 results/micro-实际签名/08_logs/micro/H6.ready.log
cat results/micro-实际签名/00_environment/microbench_status.json
```

确认上一进程已退出、GPU 已空闲后，可独立复测失败项：

```bash
bash run_stage_a.sh --mode micro --micro-item H6.ready \
  --device cuda:0 --venv-path "$SEAM_VENV" --micro-timeout-seconds 120 --resume
```

## 验证范围

tests/test_a0_recovery.py 验证真实 CPU 子进程超时回收、失败退出码和日志、超时后状态传播、旧成功文件隔离、裸 UUID 和新版驱动横幅解析。POSIX 子孙进程组回收用例在 Windows 上标记 SKIP，需在 Linux SEAM 上执行确认。

本修复包保留原始交付的历史离线验证文件。新增验证结果见 validation/a0_fix_verification.json。原版导致 A0 停滞的 GPU 根因仍需修复版分项日志定位。Linux RTX 5090 上的运行与性能均为 SERVER_PENDING。
