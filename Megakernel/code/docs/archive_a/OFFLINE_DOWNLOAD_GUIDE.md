> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# Windows 准备与传输

默认 W1–W8 不需从模型站下载内容。先用现有 SEAM 环境运行 preflight；已有核心依赖可直接使用。只有确实需要可选对照或 REAL_WEIGHTS 才执行下列人工准备。

1. 浏览器打开 Triton v3.4.0 标签页，选择 Code → Download ZIP。文件通常为 triton-3.4.0.zip。需要替换教程时解压到 offline_assets/triton/；工具递归定位五个同名 .py，并与包内同提交文件核对。官方标签源 ZIP 为 https://github.com/triton-lang/triton/archive/refs/tags/v3.4.0.zip 。
2. 在 https://github.com/linkedin/Liger-Kernel/tags 选现场可兼容版本，下载 Source code (zip)，解压到 offline_assets/liger_kernel/，保留 src/liger_kernel 和许可证。记录 tag/commit。使用源码导入后做实际前向和数值检查；失败只影响可选 Liger。
3. 在 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main 选择一个固定 revision，分别下载 config.json 和 model.safetensors；若使用分片，下载 index.json 引用的所有分片。上传后把 config、index 和分片放在 offline_assets/qwen/weights/ 下。只要结构配置时，使用 qwen/config/config.json。无需 tokenizer、数据集或整模型推理框架。
4. 从服务器 venv 查询 Linux Python ABI、架构和 pip 支持标签。通过供应方的 PyPI Files 页面下载匹配该 Linux 平台的 wheel；Windows 本地只作文件中转。放到 offline_assets/wheels/。Wheel 标记如 cp311、cp312、manylinux、x86_64 必须与服务器匹配。
5. Windows 可通过 Get-FileHash -Algorithm SHA256 生成文件摘要。使用已有 SSH 地址／跳板或 SFTP 客户端上传。主机名必须是 Windows 可解析的 SSH 配置名或获授权的实际地址。

```powershell
Get-FileHash -Algorithm SHA256 .\stage_a_megakernel_rtx5090_seam.zip
scp ".\stage_a_megakernel_rtx5090_seam.zip" "user@your-ssh-host:/path/to/SEAM/"
```

服务器使用 unzip 解压，然后 bash scripts/validate_assets.sh --venv-path /path/to/SEAM/.venv。该命令打印实际资源和校验状态。不要执行教程文件自带的测试/benchmark main 作为本实验入口。
