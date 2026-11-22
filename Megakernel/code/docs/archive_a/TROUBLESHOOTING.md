> 历史实验档案。当前 Triton 3.9.0 联合运行命令与环境要求见 [3.9.0 说明](../TRITON_390.md)。

# 排错与恢复

| 现象 | 检查 |
|---|---|
| 找不到解释器 | 提供 SEAM .venv 绝对路径，并核对 pyvenv.cfg、bin/python、Linux 架构 |
| A0 阻塞 | 打开 environment.json.errors。核对 nvcc、驱动支持 CUDA 上限、torch 构建 runtime 及实机 smoke；三者版本字段意义不同 |
| 未发现 seam_audit | 原生只读扩展可选。确认 libtriton 来源；本实验变换由单策略计划 pass 和 JIT lowerings 驱动 |
| S1 跨 warp 无物理证据 | 后端可能选择不同布局。看 B2 的 TTGIR 和 PTX；该层级保留 UNSUPPORTED |
| CUDA Graph 捕获或 JIT 失败 | 查看 worker traceback、固定版本和算子类型，保留失败分母。主进程记录 TIMEOUT、OOM、COMPILE_ERROR、RUNTIME_ERROR |
| 单个策略更慢 | 属于可用测量，比较基线、资源、worker 数和尾块，不能筛去该行 |
| 核心正确性异常 | 查看首次失败的输入类型、B1/B2 和误差；尤其留意 BF16 边界与归约次序 |
| 缺少真实 Qwen 权重 | 默认不加 --real-weights 即可测同维度子图。显式请求真实权重时不会静默改用随机数据 |
| GPU 超时或设备错误 | 终止该子进程，隔离余下 GPU 工作并写出报告。确认设备恢复后用原命令 --resume，必要时延长单候选预算 |
| 缺少 PNG | 查 acceptance_report.json.plot_error；离线补齐兼容 matplotlib 后恢复，已有测量可复用 |

静态 API 对照脚本可对本地 Triton 3.4.0 源码运行 python scripts/check_api34.py /path/to/triton。脚本不会下载内容。可用本地环境 python -m unittest discover -s tests -v 运行纯 CPU 测试。
