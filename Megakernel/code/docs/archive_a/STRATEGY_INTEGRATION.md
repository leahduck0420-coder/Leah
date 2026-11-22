> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 源码审计、端口与适配

工作区 v1.0/config/experiment.yaml 固定 Triton 3.4.0 提交 c817b9b63d40ead1ed023b7663f5ea14f676f4bc。triton-v3.4.0-seam-audit.diff 在 passes.ttgpuir 下注册 seam_audit，作用是读取 TTIR/TTGIR。只读分析不能作为优化已经执行的凭证。v1.1/seam_harness/ownership 的原始 OwnershipMap、LinearSystem、pack 与 active owners 以源码保存在当前包。learn_cuda_mlp_benchmark/planner.py 的 GF(2) 见证、列迁移、重建与顺序分片规则是移植依据。此前 megakernel/graph.py、schedule.py、codegen.py 给出了版本化依赖、worker 轮次和物化／局部 SSA 两种 lowering。旧的 MLP lowering 具有复合策略、固定形状和跨 CTA 同时驻留前提，不能直接测五条单规则增量。

本工程在相同表示原理下新增八类工作负载的参数化 lowering。计划由 plan(case, enabled) 生成，Workload.bind 将计划交给标准 Triton JIT。所有权、重分块、稠密 tile、统计量和切片使用公开语言接口。没有要求向 SEAM 安装自定义 MLIR pass。A0 记录已知审计入口以及 libtriton 的 SHA256；未发现入口只能说明该特征不存在，不能证明安装包字节等于官方发行版。

| Pass | 变换证据 | 拒绝与执行约束 |
|---|---|---|
| S1 | 解局部列空间的前像；相同实例的物化 store/barrier/volatile-load 改成局部 SSA。W1 对局部索引施加已知 XOR 排列 | 跨 program 请求拒绝。lane/warp 标签需有相应 shuffle/shared 指令证据，否则 UNSUPPORTED |
| S2 | 把 1、2、3 个分块位移到 program 内，变更发射 grid 或本地行数。W2 合并输出列，W3 合并行，W7 合并物理 CTA 行块 | 唯一写入和尾块保持；factor=1、只剩一个有效块计 NO_OP；大 tile 的编译失败或资源超限保留 |
| S3 | 归约前后数值版本显式，写 FP32 统计量；从未被覆盖的原输入／logits 重建相同舍入值 | RMS 一标量、LayerNorm 两标量、稳定 Softmax 两标量；扣除重复读取及重建成本 |
| S4 | 对归约域形成不交叠的有效分片，产生完整片后顺序消费。W4、W5、W8 的累加单位固定为 32，再组合成不同发布片宽 | 保留最后部分片、多消费者与 dtype 舍入；只有一个有效片为 NO_OP |
| S5 | 通过真实 task-ID、数据边、发布 flags 和 worker 执行器，把固定串行/拓扑分层约束变换为实际数据就绪图 | 结构相同为 NO_OP。PTX 可相同，此时实际交给 dispatcher 的 GPU 前驱矩阵必须改变，且每个任务运行一次 |

W6、W8 复用同一有界调度器。worker 扫描任务，使用 acquire 读取前驱完成标记，CAS 认领后执行；CTA 写完数据、执行 barrier、release 发布。无任务可认领的 worker 可退出。仍在执行的最后一个 worker 能继续扫描所有任务。每次成功扫描至少完成一个尚未执行的任务，扫描次数至多为任务总数，因此没有跨 CTA 无限忙等或全网格同时驻留要求。每次调用清零状态与完成计数，CUDA Graph 包含复位成本。独立插桩记录 globaltimer 的开始/结束/worker 并检查完成计数。

普通布局转换、权重读入、显式精度边界、CUDA 后端合法化、寄存器分配及必要的 CTA barrier 属于基础 codegen。B1 与对应 B2 共用这些操作。输出的 DDG 对静态单 CTA 模板保留符号阶段边，调度负载额外保存逐任务数据与执行依赖。逻辑分块与物理 CTA grid 分列，不能把逻辑 panel 数解释成实际启动数。
