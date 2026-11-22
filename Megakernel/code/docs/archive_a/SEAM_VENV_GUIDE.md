> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 选择解释器

启动器的选择顺序是 --venv-path，SEAM_VENV，脚本目录和启动目录的 SEAM 祖先 .venv。多个不同候选时终止并列明解决方式。脚本直接运行目标 bin/python，并验证 sys.prefix 与目录真实路径一致；不以 bin/python 符号链接解析后的系统二进制路径替代 prefix 检查。子进程使用完全相同的 sys.executable。启动时清除 PYTHONPATH 和 PYTHONHOME，禁用用户 site；以选定虚拟环境导入的软件为准。CUDA_VISIBLE_DEVICES 可由用户在启动前设置；只向 --device 指定的一张卡分配张量。建议在已确认对应物理设备的独占作业中运行。
