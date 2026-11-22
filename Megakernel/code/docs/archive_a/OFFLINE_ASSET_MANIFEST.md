> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 离线资源清单

| 状态 | 名称 / 来源及版本 | Linux 上的位置及验证 | 缺失影响 |
|---|---|---|---|
| REQUIRED | SEAM 已有 Linux venv，实际 Python ABI、torch 版本与 CUDA runtime 由预检打印。PyTorch 仓库 https://github.com/pytorch/pytorch | venv 的 pyvenv.cfg、bin/python、torch 导入、真实 CUDA 张量和 Graph 检查 | 所有 GPU 项 |
| REQUIRED | Triton 3.4.0，https://github.com/triton-lang/triton/tree/v3.4.0 。固定提交 c817b9b63d40ead1ed023b7663f5ea14f676f4bc | 优先安装包；A0 检查版本、语言接口、libtriton SHA256，编译并运行 SM120 探针 | 所有 GPU 项 |
| REQUIRED | CUDA Toolkit 12.8.93 和支持 SM120 的已有 NVIDIA 驱动，https://developer.nvidia.com/cuda-12-8-0-download-archive | PATH 中 nvcc、nvidia-smi 和 CUDA 实机探针 | 所有 GPU 项；不在实验进程安装 |
| BUILT_IN | 以上 Triton 提交的 02、03、05、06、09 五个教程，原始 Python 源码 | third_party/triton_tutorials/*.py；validation/package_manifest.json 中逐文件 SHA256，运行时只加载选中的已核验定义 | 可用 ZIP 中副本；替代同版本源码须与内置文件哈希相符 |
| BUILT_IN | Qwen2.5-0.5B-Instruct 结构字段，官方 config.json 文件修订 eaa56b5，https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/eaa56b5/config.json | configs/qwen2_5_0_5b.json 是需要字段的结构摘录，实际读取文件和哈希记录到环境清单 | 默认随机权重模式直接可用 |
| OPTIONAL | 完整官方 Qwen 配置及本地第一层 safetensors，https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main | offline_assets/qwen/config/config.json；真实权重推荐把同一修订的 config.json、model.safetensors 或全部 shards 与官方 index 放进 qwen/weights。记录实际各文件哈希，校验键和形状 | REAL_WEIGHTS 的 W7/W8 阻塞；不影响默认模式 |
| OPTIONAL | Liger-Kernel，https://github.com/linkedin/Liger-Kernel 。优先现场已验证的版本；手动评估的候选 tag 为 v0.6.2，不保证适配特定 torch | 先从 venv 导入，再尝试 offline_assets/liger_kernel 下已解压源码；前向接口和数值检查不通过则记录原因 | 可选强对照 |
| REQUIRED（绘图） | matplotlib 及其依赖，https://pypi.org/project/matplotlib/ 。选与现有 Python/NumPy 兼容的 Linux wheel | venv 中导入，F1–F8 用 Agg 生成 PNG | 仍保留原始数据，图表门禁阻塞 |
| OPTIONAL | safetensors，https://pypi.org/project/safetensors/ | REAL_WEIGHTS 时由 venv 本地导入 | 仅真实权重 |
| OPTIONAL | 已有 PyTorch SDPA、Triton CUDA Graph helper | 运行时特性检测和真实执行 | 对应控制项可缺 |
| NOT_AVAILABLE | 真实 DRAM 字节、L2 命中率、实际占用和硬件 warp stall 计数器 | 本工程不收集这些计数器 | 在数据集中写 UNAVAILABLE |

离线包没有 torch wheel、GPU 驱动和模型权重。允许用兼容的小依赖 wheel 补足报表；切换模型、架构或 Triton 版本属于另一实验。对外部大文件记录实际 SHA256；有可信发布方摘要时由用户核对，程序计算的本地摘要只用于追溯。
