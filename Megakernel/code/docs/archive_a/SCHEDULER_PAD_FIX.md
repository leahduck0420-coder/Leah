> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 调度状态向量的 CTA 补齐修复

## 实际服务器证据

用户提供的 micro-855bebbf3208/H6.fixed 产物来自 Linux Triton 3.4.0。metadata.json 确认 num_warps=1、grid=[4]、shared=4。堆栈停在首次 kernel 启动后的 torch.cuda.synchronize。GPU 在 worker 终止后恢复空闲。

PTX 中逻辑状态向量长度为 16，CTA 有 32 个线程，索引由 tid&15 构造。依赖数组在两个半 warp 中复制，而原子读取只由 tid&16==0 的 lane 执行。原子输出寄存器先初始化为 0，lane 16..31 没有接收 lane 0..15 的原子结果。

```text
and.b32 %r23, %r22, 15;      // replicated slot
and.b32 %r3, %r22, 16;
setp.eq.s32 %p3, %r3, 0;
mov.u32 %r27, 0x0;
@%p3 atom.global.gpu.acquire.add.u32 %r27, [...], 0;
```

后续 reduce 只做 xor 8、4、2、1 的 shuffle。两个半 warp 因而各自计算就绪条件。当前驱已完成时，低半 warp 看到状态 2，高半 warp 仍使用返回值 0，得到相反分支。进入 CAS lowering 的一侧执行 bar.sync 0，另一侧跳过该段。CPU 反例测试用这组实际 PTX 语义复现分歧。

此前将 W6 调为单 warp，仍留有半 warp 的冗余 lane，服务器复测继续超时。该历史尝试见 H6_RECOVERY.md。

## 本次修改

Workload.bind 为共享 dispatch 入口设置：

```python
PAD = next_power_of_two(max(task_count, 32 * num_warps))
```

因此 W6 的 16 个任务使用 PAD=32，W8 的 16/32 个任务、4 个 warp 使用 PAD=128。每个 physical lane 都有独立的向量槽。Order load 和 State atomic_add 都以 ids<TASKS 为 mask。补齐槽没有依赖，也没有状态原子访问。

调度仍遍历实际 TASKS 个任务。任务图、任务 ID 与 fixed/ready 的执行依赖由原计划提供。B1/B2 通过同一 bind 入口接收 PAD。离线编译脚本同步使用这一约束。

## 复测

设置好 SEAM、PATH、CPATH 和 CUDA_VISIBLE_DEVICES 后，在修复包目录执行：

```bash
bash run_stage_a.sh --mode micro --micro-item H6.fixed --venv-path "$SEAM_VENV" --resume &&
bash run_stage_a.sh --mode micro --micro-item H6.ready --venv-path "$SEAM_VENV" --resume
```

两项 PASS 后执行整个 A0 和 smoke：

```bash
bash run_stage_a.sh --mode micro --venv-path "$SEAM_VENV" --resume &&
bash run_stage_a.sh --mode smoke --venv-path "$SEAM_VENV" --resume
```

每项的 08_logs/micro/H6.*.artifacts 保存实际 PTX 与 TTGIR。H6 的 TTGIR 预期出现 tensor<32xi32，State atomic_rmw 带有效域 mask；就绪归约应覆盖完整 warp。W8 入口另由 smoke/full 验证。

## 验证口径

CPU 测试包括实际 PTX 语义的反例、W6 16/64/256 任务的正常 bind 入口，以及 W8 单/双 batch、B1/B2 的 CTA 补齐配置。离线 SM120 验证使用 Windows Triton 3.9.0，记录于 validation/scheduler_pad_offline_sm120.json。服务器 Linux Triton 3.4.0 上的正确性、终止性与性能为 SERVER_PENDING。
