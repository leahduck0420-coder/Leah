# Triton TC/SIMT 尾部协同优化 v2

本版本针对 v1 实验暴露的两个问题修改：模型在 `M=1, K=4096` 上误选全 SIMT hybrid；Python 热路径查表和 launch 元数据处理抵消了预选 kernel 的收益。目标是在预先声明的不能整除 Tile 的领域中寻找可复现收益，同时在没有收益的 shape 上回退到原编译器路径。

代码交付时只完成 CPU 静态与逻辑测试，没有在本地运行 GPU 编译或性能实验。实际收益必须由远程目标 GPU 的新结果确认。

## v2 改了什么

### 1. 新增 `hybrid_k`

v1 的 `hybrid` 把 M/N 空间边缘交给 CUDA Core 计算完整 K。当 M 很小或空间边缘较大时，完整 K 的 SIMT 循环会严重退化。

`hybrid_k` 保留 M/N 边缘的 Tensor Core `tl.dot`，只对 `K mod BK` 的 1–4 个元素使用 FP32 `tl.fma`：

\[
C_{mn}=\sum_{k=0}^{K_c-1}A_{mk}B_{kn}
+\sum_{k=K_c}^{K-1}A_{mk}B_{kn},\quad 1\le K-K_c\le 4.
\]

第一项由 Tensor Core 完成，第二项由 SIMT 完成。它避免因 M/N 尾部而执行完整 K 的 SIMT 循环。普通 `hybrid` 仍保留用于实验，但只有 SIMT 有效 MAC 不超过总 MAC 的 2%、K 尾部不超过 4、每个 group 都存在 16×16 TC 主体时才允许进入候选。

以下情况默认排除 hybrid：

- `K < BK`，没有 Tensor Core K 主体；
- 任一 group 的 M < 16 或 N < 16；
- `K mod BK > 4`；
- 估算的 SIMT MAC 占比超过 2%；
- 完全对齐、没有任何 SIMT 工作的冗余 hybrid。

`--unsafe-hybrid` 可以关闭这些保护，仅用于研究负例，正式结论不要使用。

### 2. 多 head Attention 合为一次启动

QKᵀ 和 PV 的 4 个等形状 head 现在使用 `program_id(2)` 构成三维 grid。只有输入确实共享 storage、shape/stride 相同且 storage offset 等距时才启用；否则自动使用逐 group 启动。MoE token 数不同，仍按非空专家分别启动。

Triton padding 基线和所有优化候选共用这套批处理启动基础设施，因此多 head 合并不会只给优化路径使用。Inductor 保持其默认行为。

### 3. 热路径只做 tuple 查询和已绑定 callable 调用

`Case.dispatch_key` 使用 `cached_property` 预先计算，不在每次调用中执行 `dataclasses.asdict` 或 JSON 序列化。`Plan` 在准备阶段绑定：

- kernel launcher；
- grid；
- shape、stride、epilogue 和策略参数；
- 输出张量；
- 多 head 的 storage stride。

`Plan.__call__()` 只遍历已经绑定的 launcher。`Dispatcher` 保存 `tuple key → callable`，`optimized_dispatch` 计入真实字典查询和一层 Python 调用成本。

### 4. 分开五条关键路径

|报告方法|含义|
|---|---|
|`inductor_default`|`torch.compile(..., backend='inductor')` 默认策略|
|`triton_padding_autotune`|Triton autotuner 包装仍在调用路径|
|`triton_padding_bound`|autotune 选完配置后，用相同配置和 Plan 直接调用；主要 Triton 原策略对照|
|`optimized_bound`|策略表选中的优化 kernel，提前绑定|
|`optimized_dispatch`|预计算 tuple 查表并调用优化 kernel，包含在线调度开销|
|`optimized_guarded`|validation 同时胜过 Inductor 与 padding 才采用优化，否则回退到更快基线|
|`model_bound`|局部模型选择，仅作为泛化诊断|

这样可以区分尾部算法、autotuner 包装、在线调度和原编译器之间的差异。

### 5. 保守部署回退

每个 shape 使用三个互不混用的阶段：

1. discovery：测完整合法候选，生成精确策略表；相对最优 padding 改善不足 3% 时仍选择 padding。
2. validation：比较 `optimized_dispatch`、`inductor_default` 和 `triton_padding_bound`。只有优化比两个基线的中位数都快至少 3%，且优化样本最大值低于两个基线的最小值，才启用优化。
3. holdout：随机顺序测量并报告；不再改变策略或 route。

`optimized_guarded` 是最终部署方案。验证回退降低退化风险，但 validation 的有限样本不是数学保证；holdout 仍可能观察到噪声或退化。

## 针对的应用领域与 shape

`targeted` 套件在代码修改前预先声明以下领域：

|算子|领域|基础 shape 示例|尾部变体|
|---|---|---|---|
|GEMM|一般矩阵尾部|128×128×128、512×512×512、128×1024×512|M+1、N+1、K+1、K+3、M/N/K+1|
|Linear|ragged token projection|128×768×768、256×1024×1024|同上|
|融合 Linear|ragged fused projection|同 Linear|同上|
|QKᵀ|不规则序列长度 Attention|128×128×64、512×512×128|同上|
|PV|不规则 reduction Attention|128×64×128、512×128×512|同上|
|MoE|ragged expert projection|128×768×768、256×1024×1024|同上|

每组包含对齐父 shape 和五个不能整除 Tile 的成对样本。六种算子还包含 `1×4096×4096` 负对照，确保 v1 的窄矩阵误选不会复现。正式报告既显示各个 shape，也显示对齐父 shape 与尾部 shape 的成对性能比。

## 模型

v1 的全局非负线性模型倾向于把所有 case 选成 32×32×32。v2 使用有边界的局部实测模型：每个 `op:epilogue:config` 保存校准坐标和 request 延迟，取最多三个相邻点做对数空间加权预测。坐标包括：

\[
(\log_2 M_{max},\log_2 N,\log_2 K,\log_2\sum M_g,
\log_2 G,r_M/B_M,r_N/B_N,r_K/B_K).
\]

前五个坐标任一相差超过 1 octave 时视为分布外，不做外推并回退到真实 padding 配置。模型不能替代精确策略表；`model_bound` 单独报告，最终方案使用精确表和 validation guard。

## 在远程服务器运行

假定目录为：

```text
~/Leah/TC_SIMT/
└── cooperation_triton_v2/
```

保持在 `~/Leah/TC_SIMT`，激活原来的 `tilelang-t4` 环境。

### 1. 安装 Python 依赖

沿用服务器上与 CUDA 匹配的 PyTorch/Triton，只补齐缺失依赖：

```bash
cd ~/Leah/TC_SIMT
python -m pip install -r cooperation_triton_v2/requirements.txt
```

Matplotlib 强制使用 `Agg` 后端，服务器不需要桌面、浏览器或 X11。

### 2. CPU 测试和 GPU 正确性

```bash
cd ~/Leah/TC_SIMT
python -m unittest cooperation_triton_v2.tests -v
CUDA_VISIBLE_DEVICES=0 python -m cooperation_triton_v2.gpu_checks
```

任何一条失败都先停止性能实验。

### 3. 查看正式计划

```bash
python -m cooperation_triton_v2.benchmark plan \
  --suite targeted --out results/v2_plan
```

`plan.json` 会列出每个 shape 的合法候选和被排除的原因，不执行 GPU 代码。

### 4. Smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m cooperation_triton_v2.benchmark run \
  --suite smoke --blocks 1 --repeats 5 \
  --discovery-blocks 1 --validation-repeats 3 \
  --out results/v2_smoke
```

Smoke 只检查流程，不能作为性能结论。

### 5. 校准局部模型

```bash
CUDA_VISIBLE_DEVICES=0 python -m cooperation_triton_v2.benchmark calibrate \
  --suite calibration --discovery-blocks 3 --repeats 20 \
  --out results/v2_calibration
```

若只关心部分算子，可先加 `--ops gemm linear qkt`，但未校准算子的 `model_bound` 会回退；精确策略表和最终 guard 仍可工作。

### 6. 正式 targeted 实验

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_LOGS=output_code \
python -m cooperation_triton_v2.benchmark run \
  --suite targeted \
  --model results/v2_calibration/model.json \
  --blocks 5 --repeats 20 \
  --discovery-blocks 3 --validation-repeats 9 \
  --out results/v2_targeted \
  2> results/v2_targeted_inductor_code.log
```

实验较大时可以按算子分开运行，每个输出目录必须不同：

```bash
python -m cooperation_triton_v2.benchmark run \
  --suite targeted --ops gemm linear linear_post \
  --model results/v2_calibration/model.json \
  --out results/v2_projection

python -m cooperation_triton_v2.benchmark run \
  --suite targeted --ops qkt pv moe \
  --model results/v2_calibration/model.json \
  --out results/v2_attention_moe
```

### 7. 使用冻结表复测

```bash
python -m cooperation_triton_v2.benchmark run \
  --suite targeted \
  --model results/v2_calibration/model.json \
  --table results/v2_targeted/table.json \
  --out results/v2_frozen_repeat
```

冻结表保留已选 config 和 validation route。为了生成消融图，复测仍会编译并测量候选，但不会用新 discovery 或 holdout 改写最终选择。

### 8. 旧 v1 shape 回归

```bash
python -m cooperation_triton_v2.benchmark run \
  --suite regression \
  --model results/v2_calibration/model.json \
  --out results/v2_regression
```

可加 `--profile` 导出关键方法的 profiler trace；可加 `--graph` 记录可用时的 CUDA Graph replay 对照。普通 request 结果仍保留。

## 自动生成的结果和图表

正式运行结束时，`report.py` 会直接在服务器用 Python 生成图表：

```text
results/v2_targeted/RESULTS.md
results/v2_targeted/figures/
```

不需要重新运行 GPU 实验，也可以从已有 CSV 重画：

```bash
python -m cooperation_triton_v2.plot_results \
  --run results/v2_targeted \
  --formats png svg pdf
```

`figures/` 中每组图同时提供 PNG、SVG、PDF，包含：

- `01_speedup_heatmap`：各算子 × shape 的最终方案加速比，同时对比 Inductor 默认和 Triton padding autotune；
- `02_latency_*`：两个基线、局部模型、策略表 kernel、含调度调用和最终回退的 request/device 延迟；
- `03_ablation_tile_*`：在相同 Tile 下比较 padding、small_mma、hybrid_k、hybrid，空白表示策略不合法；
- `04_strategy_selection`：每个 shape 选择的 policy 和 BM×BN×BK；
- `05_benefit_regression_summary` 和 `05b_win_regression_counts`：按算子的几何平均、最好、最差、收益数和退化数；
- `06_tail_pair_comparison`：每个不能整除 Tile 的 shape 与其对齐父 shape 的成对比较。

图中加速比大于 1 表示优化更快。所有主图使用 holdout 中位 request 延迟；缺失或失败结果保留为空，不按 0 填充。`figures/manifest.json` 记录图表名称、解释、格式和来源 CSV。

另外生成：

|文件|内容|
|---|---|
|`before_after.csv`|每个 shape 的 Inductor、Triton padding、优化 kernel、调度和最终方案|
|`comparison.csv`|全部合法候选和全部报告路径|
|`domain_summary.csv`|按算子/领域的几何平均、最好、最差、胜出数|
|`tail_pairs.csv`|对齐父 shape 与不能整除 Tile 的 shape 成对比较|
|`figures/*.png`|服务器生成、可直接预览的位图|
|`figures/*.svg`|适合论文或文档编辑的矢量图|
|`figures/*.pdf`|适合打印和归档的矢量图|
|`figures/manifest.json`|所有图表的文件清单与解释|
|`table.json`|精确 shape → config → validation route|
|`cases/*.json`|discovery、validation 和 holdout 原始记录|
|`artifacts/*`|TTIR/TTGIR/PTX 等编译产物及资源/指令审计|
|`completion.json` / `failures.json`|完成状态和完整错误|

报告中的 `optimized_bound` 可回答算法 kernel 是否有潜力；`optimized_dispatch` 可回答查表成本；`optimized_guarded` 才是采用整个方案后的最终性能。只有 `completion.json` 完整、没有失败，并且 holdout 仍显示收益时，才能说某个领域得到提高。

## 文件说明

|文件|用途|
|---|---|
|`kernels.py`|四种 Triton 尾部策略、统一 epilogue、batched grid|
|`specs.py`|不可变 Case、预声明领域、校准/正式/回归套件|
|`policy.py`|合法性保护、代价几何、局部模型、精确表、轻量 dispatcher、回退判定|
|`runtime.py`|输入、参考实现、预绑定 Plan、autotune 绑定基线、审计|
|`timing.py`|request、device span、可选 graph replay 计时|
|`benchmark.py`|plan/calibrate/run 和 discovery/validation/holdout 协议|
|`report.py`|聚合 CSV、Markdown，并触发服务器端绘图|
|`plot_results.py`|Matplotlib Agg 静态科研图，支持单独重画 PNG/SVG/PDF|
|`tests.py`|CPU 逻辑与报告回归测试|
|`gpu_checks.py`|远程 GPU 数值、batched head、hybrid_k 检查|
|`run_remote.sh`|顺序执行测试、smoke、校准和 targeted 实验|

## 结论边界

- `small_mma` 表示较小 `tl.dot` 块，不保证 ISA MMA atom 改变。
- hybrid/hybrid_k 在同一 kernel 中顺序执行，Triton 管理 accumulator 布局，不宣称 TC/SIMT 并发或 fragment-native 汇编。
- QKᵀ/PV 是 Attention 的两个 GEMM，不包括 softmax、路由、通信或反向传播。
- MoE 是多个专家 GEMM，不是 persistent/grouped MoE kernel。
- validation guard 是可部署选择机制的一部分，因此其 route 必须与策略表一起冻结；不能根据 holdout 重新挑赢家。
- 代码交付时没有目标 GPU 新结果，不能预先承诺一定加速。设计目标是在确认有收益的不能整除 Tile 领域启用优化，其余场景回退。
