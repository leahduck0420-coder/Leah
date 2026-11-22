# Triton 3.9.0 上的 RTX 5090 Stage A 与 Stage B 联合实验

这套入口先对原来的每个配置分别请求 S1、S2、S3、S4、S5，再用同一轮合格的实测记录启动组合搜索。两个阶段按顺序占用同一张 GPU，保留各自的原始数据、IR、图表、数据集和验收报告。

## 正式运行

上传 ZIP 到服务器，解压并进入 stage_ab_triton390。下列命令使用之前已在 RTX 5090 上通过 3.9.0 S0 的虚拟环境。当前路径不同可替换此参数。可复用之前已验证的 CPATH、PATH 和 CUDA_VISIBLE_DEVICES；设备编号为该进程可见的 cuda:0。

```bash
unzip stage_ab_all_strategies_triton390_rtx5090_seam.zip
cd stage_ab_triton390
bash run_all.sh --mode full --device cuda:0 --venv-path /das/t01/homes/liqi/Leah/program_linear_layout/triton390-s0/.venv --output-dir results/full_triton390 --resume
```

预检、Stage A 和 Stage B 都由最后一条命令执行。运行全程无需网络、模型下载、安装依赖或改动现有 .venv。默认使用包内 Qwen 结构参数和确定性随机权重。如果上一轮使用了本地真实权重，加上相同的 --real-weights 和 --model-dir；已有离线资源可通过 --offline-assets-dir 指定。保持两阶段使用相同资源。当前环境、教程版本与离线资源要求见 docs/TRITON_390.md。

若只验证管线，使用独立输出目录：

```bash
bash run_all.sh --mode smoke --device cuda:0 --venv-path /das/t01/homes/liqi/Leah/program_linear_layout/triton390-s0/.venv --output-dir results/smoke_triton390 --resume
```

Full 与 Smoke 必须使用不同的输出路径。中途停止后，在同一环境重新执行原命令。要重新尝试上一轮 Stage A 的编译、数值或设备失败，可在原命令末尾追加 --retry-stage-a。它会重新进入 A，复用通过校验的请求，重试其余请求。若证据变化，B 输出到另一个证据标识目录。两个独立的正式重复实验使用不同的 --output-dir。GPU 报错时先查日志，进程会隔离设备并保留状态。

## Stage A 的配对设计

默认种子 20261119。400 个原始配置与上一轮服务器的 matrix.json 一致，含原有控制项；每个配置发出 5 个单策略请求，共 2,000 个独立 Case ID。Smoke 使用原有 17 个配置，共 85 个请求，同样覆盖 40 格。同一 anchor_id 内使用相同功能输入、输入种子、形状、dtype、数值契约和采样协议。原有策略配置保留原 Case ID。新增请求只为目标变换选择所需参数，如合并因子或切片宽度。未激活策略的配置参数不启用其他变换。基线 B0 是 PyTorch，B1 是对应后端的无策略运行；B2 每次只开启一条策略。跨后端比较请同时看 lowering、B1 的绝对时间和 B1/B2。

| 工作负载 | 原配置数 | 请求数 | 默认 Full 中至少一个配置通过静态检查的策略 |
|---|---:|---:|---|
| W1 Bias → GELU → Residual | 17 | 85 | S1、S2 |
| W2 GEMM → Bias → GELU | 49 | 245 | S1、S2、S4 |
| W3 Norm → Scale / Linear | 66 | 330 | S1、S2、S3、S4 |
| W4 Softmax → Scale / Weighted Sum | 44 | 220 | S1、S2、S3、S4 |
| W5 QK → Softmax → PV | 43 | 215 | S1、S2、S3、S4 |
| W6 任务 DAG | 37 | 185 | S2、S4、S5 |
| W7 Qwen MLP | 66 | 330 | S1、S2、S3、S4 |
| W8 Qwen Attention | 78 | 390 | S1、S2、S3、S4、S5 |

以上是测试设计。默认 Full 在 CPU 端得到 1,472 次静态允许、451 次不适用、75 次无有效变化、2 次非法拒绝。GPU 编译、数值和实际作用检查均在服务器执行，可能进一步减少有效数。

新增作用位置包括 S1 对概率的局部交接、S2 对元素块/行/查询/任务和 W8 头组的重新划分、S4 对 GEMM K 维、Norm 统计归约、DAG 前驱和 MLP down projection 归约的切片。每个作用位置链接到实际 Triton lowering。W6 合并后以组发布，在 CTA 内顺序完成成员任务，校验组和原始任务都恰好执行一次。切片改变浮点累加次序，沿用已有的逐算子数值门禁；回退、超限和编译失败都会留下单独记录。

每个工作负载也向表中未列出的策略发出请求。S3 需要可重建中间值与充分统计量；W1 没有可供 S4 切分的归约；S5 需要现有运行时中独立发布的任务；W6 的独立单步 S1 需要已有的同 Program 所有权见证。保持原工作负载和基线时缺少这些前提的请求会给出具体原因。实际回退路径按相同协议测 B0、B1 和回退执行；回退执行时间不包含编译器判定时间。通过静态检查仍须在 GPU 上验证指令/启动几何或调度变化、正确性和可重复性，才记作有效策略性能。

## 测量与 Stage B

- Full 使用每项 12 个交错批次、每批 3 次重放。Smoke 为 10 个批次、每批 2 次。先编译预热，再由 CUDA Event 计时。有 CUDA Graph 时按原协议捕获实际执行；每个有状态候选包含必要的重置操作。
- 实验使用 Triton 3.9.0、单卡 RTX 5090、默认 24 GiB 显存上限、关闭 FP fusion，并保留 W8 的 constexpr 修复。此前通过 S0 的环境为 PyTorch 2.15.0+cu134 和 CUDA 13.4。运行时校验当前解释器导入的 Triton、实际 SM120 编译和发射，分别记录 PyTorch CUDA runtime、nvcc 和驱动。未依赖 Nsight Compute。
- Stage B 检查本轮 A 的设计、种子、全部请求、输入、编译产物哈希、数值、原始计时和数据集。可用于搜索的每条记录都必须是通过验证的真实单策略测量。A 有局部失败时，已有合格样本仍可供 B 使用。运行状态和排除原因见两个验收报告；联合 PASS 要求 A 和 B 都达到各自的正式门禁。
- B 沿用上次的按工作负载分层选区域、同语义契约单策略复测、无策略/单策略/固定组合/贪心/束搜索及有限候选集内的实测 oracle。Full 的搜索预算为 20、50、100，束宽 4、8、16，深度为 2 至 4。新 A 的候选数和作用位置变化可能改变入选区域与合法组合数。B 会在当前 GPU 上重新编译、执行并计时，不把 A 的时间或历史服务器结果回填为 B 的测量。
- 联合入口接收 --seed、--max-memory-gb、--offline-assets-dir、--model-dir、--real-weights、--resume 和 --retry-stage-a。单独阶段的诊断参数见各自 CLI。默认运行时长与磁盘占用取决于设备、合法组合数和保留的逐候选 IR。

## 查看及下载结果

| 产物 | 相对指定输出目录的位置 |
|---|---|
| 两阶段状态、当前 B 子目录、A 证据标识 | pipeline.json |
| A 的 40 格汇总、逐请求表与验收 | stage_a/06_reports/workload_strategy_8x5.csv、workload_strategy_cases.csv、acceptance_report.json |
| A 时间与回退原始样本 | stage_a/02_performance/raw_timings.csv、fallback_timings.csv |
| A 图表、数据集和 IR | stage_a/05_analysis/、07_dataset/、04_ir/ |
| B 已消费的 A 行与排除原因 | stage_b/证据标识/00_environment/stage_a_candidate_index.csv、stage_a_blockers.json |
| B 组合、搜索、成本与 IR | 同一 B 子目录的 02_pairwise/ 至 06_ir/ |
| B 图表、验收、完整数据集 | 同一 B 子目录的 07_analysis/、08_reports/acceptance_report.json、09_dataset/stage_b_dataset.csv |

逐请求 CSV 的 B0_us、B1_us、applied_us、fallback_us 是微秒中位数。有效变换行填入 applied_us，回退行填入 fallback_us，失败保留状态和原因。8×5 表单列各格的请求数、有效数、回退数、状态计数和有效样本加速比中位数。两种加速比分母均为实际 B2。每格中位数对配置等权，包含合法且慢于基线的配置。筛选同一个 anchor_id 可以在同一输入上做横向比较。

打包本轮完整结果（需要 Linux 的 zip）：

```bash
bash scripts/package_results.sh results/full_triton390
```

生成 results/full_triton390.zip，内含 A、B 及 pipeline.json。返回码 0 表示正式联合 PASS 或 Smoke 完成，1 表示已生成两阶段报告但有验收项未达标，2 表示交接或执行受阻。查阅 stage_a/08_logs/、B 子目录的 10_logs/ 和根目录 pipeline.json。若 B 因 A 的设备异常或缺少输入而阻塞，会另存 stage_b/blocked/08_reports/acceptance_report.json。

## 来源与本地验证

Stage A 基于 stage_a_megakernel_rtx5090_qkvfix_20261119.zip；Stage B 基于上一轮工程并恢复服务器实际使用的 W8 头分组整数 constexpr。docs/archive_a 与 docs/archive_b 保留历史方案，其中旧版本号和历史复现命令仅适用于旧轮次。当前运行入口和依赖以此文档及 docs/TRITON_390.md 为准。当前配对设计、入口和结果结构以本 README 为准。许可证与上游来源在 third_party/。validation/ 中的清单只记录本地测试与静态规划。可在解压后运行选定解释器的 scripts/verify_package.py 校验包内文件。用 python -m unittest discover -s tests 重跑 CPU 回归。
