> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 动作、状态与执行

stage_b/planner.py 中 Action 包含 strategy、location、parameter。每次扩展从所选原始图和整个有序前缀重建可执行计划，并重新求解与当前所有权或依赖有关的证明。OwnershipMap 沿用项目的 GF(2) 线性系统。可执行状态含 tile、合并因子、局部边位图、统计、切片、任务顺序、头组、依赖图及缓冲区约束。stage_b/runtime.py 将状态直接传入 @triton.jit 内核的编译常量、grid 和真实张量。

W3 在统计重建出现后才产生 stats_to_rebuild 位置。W7 可分别消除 gate 与 up 的写回，两步与一次同时消除双边的终态可以共享编译。W4/W5/W8 中先引入统计再分片时，最终统计读取放在切片循环外。逆序可将读取留在切片内，编译器生成不同专门化，两者分别计时。W8 按组收缩 head 节点与 O 投影的前驱集合，尾组只覆盖实际存在的头。

历史包含所有动作，即使有多个历史到达同一终态。终态键含 dispatch 参数、OwnershipMap、DDG、统计读取位置及调度；机器码和 dispatch 与严格前缀一致时，运行期记为 NO_OP。编译前后 IR、实际参数和数值检查随每一不同前缀保存。该集成位于计划层与公开 Triton JIT 接口，使用当前 .venv 的后端生成 TTIR、TTGIR、LLVM IR、PTX、CUBIN。预检记录已有自定义 pass 注册与 libtriton 指纹。
