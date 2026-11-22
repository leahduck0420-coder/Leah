> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 离线资源

| 资源 | 版本/来源 | 部署方式 |
|---|---|---|
| Stage A Full 输入 | 用户已有 qkvfix 单次运行 | --stage-a-results 指向保留完整目录结构的结果 |
| Stage A 源码和教程 | 用户最终 ZIP、Triton v3.4.0 | 已内置，来源和 MIT 许可证在 third_party |
| 编译器和运行库 | SEAM .venv 内 Triton 3.4.0、现有 PyTorch、NumPy、Matplotlib；系统 nvcc 12.8.93 | 沿用以往环境，脚本不会安装依赖 |
| Qwen2.5-0.5B-Instruct 配置 | 官方模型配置子集，blob eaa56b5 | configs/qwen2_5_0_5b.json 已内置，默认固定种子的随机权重 |
| Qwen 官方权重 | 可选本地 config.json、*.safetensors，分片时附索引 | --real-weights --model-dir /path/to/model；需要现有 safetensors |
| Linux wheels | 仅手工补齐缺少的工具包 | offline_assets/wheels，必须匹配 Linux 平台和现有 Python ABI |

Windows 本地人工取得资源时从官方模型库选择一致的配置、权重与索引，用 Get-FileHash -Algorithm SHA256 记录所下载文件；通过现有 SSH 别名/IP 或文件传输工具上传，Linux 用 sha256sum 对照。候选运行前会验证必需 config 字段及权重索引，加载校验所需 tensor 的名称和 shape。若选真实权重且缺资源，W7/W8 会记 BLOCKED。默认合成权重只用于真实维度的子图。自定义资源沿用 --offline-assets-dir。已安装环境优先，正式脚本不修改依赖。可选离线维护命令示例为 `"$SEAM_VENV/bin/python" -m pip install --no-index --no-deps --find-links /本地/wheels 包名==已验证版本`，请在维护环境中人工执行。

官方出处：[Triton v3.4.0](https://github.com/triton-lang/triton/tree/v3.4.0)、[PyTorch](https://github.com/pytorch/pytorch)、[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。

资源定位按显式 --offline-assets-dir、SEAM_OFFLINE_ASSETS、Stage A 运行及其最近三层祖先中的 offline_assets、包内目录依次查找。
