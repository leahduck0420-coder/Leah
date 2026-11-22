# 3.9.0 运行环境与复现

联合入口和 Stage A、Stage B 的各个 GPU 测量进程使用所选 .venv 的 Python。选择顺序为命令行 --venv-path、SEAM_VENV、工程或启动目录祖先中唯一的 .venv。入口清除外部 PYTHONPATH。之前通过 S0 的环境为 /das/t01/homes/liqi/Leah/program_linear_layout/triton390-s0/.venv，其中记录了 PyTorch 2.15.0+cu134。请保留该环境已有的 PATH、CPATH 和设备可见性配置。实验规划及变换由包内代码完成，原版和带 seam_audit 的 Triton 3.9.0 均可接受；预检记录实际模块路径和 libtriton SHA256。

| 项目 | 规则 |
|---|---|
| 编译器 | 导入 triton 的版本必须为 3.9.0，允许同一版本的本地构建标识。微基准、单策略、组合搜索、可选基线和离线编译入口均检查实际加载的版本。 |
| 设备与工具链 | 单卡 RTX 5090、SM120。nvcc 版本至少 12.8，兼容性最终由真实探针判断。此前 S0 环境使用 CUDA 13.4。系统 nvcc、驱动、Torch CUDA runtime 和 wheel 中的编译工具可能使用不同版本。 |
| 数值与计时 | B0 使用所选 .venv 的 PyTorch，A/B 比较 PyTorch、CUDA runtime 和编译器身份。FP32 matmul 采用 IEEE 精度；保留已有 FP16/BF16 归约和累加策略、QKV 舍入检查、固定输入、采样与失败门禁。 |
| 核心依赖 | Linux .venv 中已有 torch、triton==3.9.0、numpy、matplotlib。使用随机权重时读取包内 Qwen 配置；真实权重需要用户提供本地 safetensors 和对应依赖。 |
| 模型与外部对照 | Qwen、Liger 的离线目录和加载方式沿用原工程。Triton 五个教程及 MIT 许可证已从 3.9.0 对应 Git 提交导出，文件哈希见 configs/tutorials_390.json。 |
| A → B | A 保存声明的目标版本、实测导入版本、GPU、libtriton 指纹，以及每个测量 worker 的版本。B 按协议、版本、源码和产物校验 A 的单策略记录，重新在同一环境编译、计时。版本迁移使用新的输出目录。 |

在 README 指定的解压目录执行正式命令。独立检查可以使用：

```bash
SEAM_VENV=/das/t01/homes/liqi/Leah/program_linear_layout/triton390-s0/.venv bash scripts/preflight.sh
```

A0 对单个 SM120 程序实际调用 JIT warmup 与 launch，核对编译身份、输出数值、TTIR、TTGIR、LLVM IR、PTX、CUBIN 和寄存器信息，并检查 CUDA Graph。正式执行继续使用逐候选数值和计时门禁。若 3.9.0 的运行环境缺少报告依赖，在同一 ABI 的 Linux 机器准备匹配的 wheel，上传 offline_assets/wheels，手工按需执行所选解释器的 python -m pip install --no-index --find-links ./offline_assets/wheels 包名；正式 runner 全程离线，不安装依赖。

固定的上游提交是 b7ddcf86c095de79527ba3098503e73158bb94bd。本地已有对应源码时可以执行 python scripts/check_api390.py /path/to/triton 查看静态核查结果。服务器运行无需额外上传 Triton 源码。验证材料见 validation/。Windows 离线生成的 SM120 编译记录用于检查代码生成，Linux 服务器上的数值和时间由本次独立运行产生。
