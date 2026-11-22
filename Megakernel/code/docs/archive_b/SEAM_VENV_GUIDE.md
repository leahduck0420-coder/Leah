> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# SEAM 环境

解释器按 --venv-path、SEAM_VENV、工程与工作目录祖先的唯一 .venv 依次定位。多个不同 .venv 时须明确给出绝对路径。启动器检查 pyvenv.cfg 和 bin/python，Python 内再次比对 sys.prefix。只选择逻辑 cuda:0；若作业已通过 CUDA_VISIBLE_DEVICES 分配设备，沿用该分配并按 UUID 收集 nvidia-smi 状态。

preflight 在当前环境记录 Python、torch.version.cuda、nvcc、驱动版本及支持 CUDA 版本、libtriton 二进制指纹、seam_audit 注册情况、CUDA graph 和真实 SM120 小 kernel。编译器变换由包内计划层驱动 .venv 中的 Triton，无须仅为本实验加载 3.9.0 动态库。当前工具链支持的 CUDA、Torch runtime 和驱动最大支持版本分别记录。
