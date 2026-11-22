> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# Linux 离线部署

使用 unzip stage_a_megakernel_rtx5090_seam.zip，进入 stage_a_megakernel。完整性检查可用现有 venv 的 python scripts/verify_package.py。生成的 manifest 对源码、配置、文档与教程逐文件记录 SHA256，忽略后来上传的资产和结果。

```bash
/path/to/SEAM/.venv/bin/python -V
/path/to/SEAM/.venv/bin/python -m pip debug --verbose
uname -m
nvcc --version
nvidia-smi
bash scripts/validate_assets.sh --venv-path /path/to/SEAM/.venv
bash scripts/preflight.sh --venv-path /path/to/SEAM/.venv --device cuda:0
```

仅在你明确决定补齐独立报表依赖后，人工使用：

```bash
/path/to/SEAM/.venv/bin/python -m pip install --no-index --find-links ./offline_assets/wheels --no-deps matplotlib
```

缺少的传递依赖也必须是用户已准备、与现有依赖兼容的 Linux wheel，并应先比较版本冲突。不要把 torch、triton、驱动升级作为修复方法。runner 进程禁止 Python 网络连接，关闭 Hugging Face 在线访问；所有程序入口只加载本地软件和数据。系统驱动、CUDA、.venv、CPU 架构均保持由 SEAM 管理。
