# 环境恢复与运行

更新：2026-10-13。

## 机器与基础环境

- Python 3.12.13；系统提供 Torch 2.11.0+cu128。2026-10-13 本机只识别到 1 张 RTX 4090（24 GB）；多卡训练前须先确认可用显卡数。
- 本地盘 60 GB，空间有限；大文件、pip 缓存与临时目录放 `/test` 盘。2026-10-12 曾因 pip 隔离构建把本地盘写满：编译扩展须关闭构建隔离（`--no-build-isolation`）并把 `TMPDIR` 指到 `/test`。
- 会话重启会终止由会话直接启动的后台任务，耗时任务须脱离会话启动。
- 命令行没有 GitHub 凭据，推送使用编辑器自带的 git 凭据助手。

## 主环境（训练、检索、测试）

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-research.txt
.venv/bin/python -m unittest discover -s tests -q
```

`--system-site-packages` 用于复用系统 Torch。`requirements-research.txt` 含 Qdrant 客户端、MCP、SciPy、pandas。测试全部在 CPU 上运行，不需要模型权重。

## 其他隔离环境

- `.venv-serving`：vLLM 推理服务（Torch 2.11.0+cu130、Transformers 5.19.0、vLLM 0.25.1），依赖见 `requirements-serving.txt`。启动时须把该环境的 `bin` 加入 `PATH`，并仅为服务进程设置 `LD_LIBRARY_PATH=$PWD/.venv-serving/lib/python3.12/site-packages/nvidia/cu13/lib`。
- `artifacts/deepspeed_env/site`：DeepSpeed 0.19.7（`requirements-distributed.txt`，安装须加 `--no-deps --no-build-isolation`），只在使用该后端时加入 `PYTHONPATH`。
- `artifacts/browsers`、`artifacts/browser-deps`：Playwright 的 Chromium 及其共享库（`requirements-dev.txt`），目前没有使用它们的测试入口。

## MinerU 解析环境

用于阶段 1 的中文财报表格解析，与主环境隔离：

```bash
python3 -m venv --system-site-packages .venv-mineru
TMPDIR=/test/Finagent/tmp PIP_CACHE_DIR=/test/Finagent/pip-cache \
  .venv-mineru/bin/python -m pip install "mineru==4.0.12"
MINERU_HOME=/test/Finagent/mineru_home MINERU_MODEL_SOURCE=modelscope \
  .venv-mineru/bin/mineru-kit models download --tier standard
```

- 环境约 583 MB，在本地盘；模型与配置在 `/test/Finagent/mineru_home`。两个 `MINERU_*` 环境变量每次运行都要带。
- 许可证为 Apache 2.0 加附加条款（月活过亿或月收入超 2,000 万美元需商业许可；对第三方提供在线服务须标明使用 MinerU），2026-10-13 按 4.0.12 的 LICENSE.md 核对。
- **只用 `mineru-kit parse` 做本地解析，不带 `--remote`**：该选项会把文件上传到 mineru.net。另一个命令 `mineru` 是带文档库、后台服务与使用统计的客户端，本项目不使用，也未启动其服务。
- MinerU 4.x 的命令与 2.x 不同（没有 `mineru[core]`，也没有 `-b pipeline`）；表格以 HTML 加整表位置框输出，不含单元格位置，单元格到原文字符位置的映射由 `src/data/table_map.py` 完成。
- 2026-10-13：安装完成，模型下载已启动；本地解析尚未在真实年报页上跑通，跑通后在此补充解析命令。

## 不在 Git 中的资产

| 位置 | 内容 |
|---|---|
| `models/bge-m3`、`models/bge-reranker-v2-m3` | 向量与重排模型，revision 见 `configs/jd_retrieval.json` |
| `/test/Finagent/models/Qwen3.5-9B` | 现有基座模型，路径与 revision 见 `configs/m0_inference.yaml`；阶段 4 可能更换 |
| `data/raw/chinese_expanded/` | 招商银行 2023、中国电信 2023、康美药业 2017 及更正公告、新纶科技 2018／2019 及更正公告的原件 |
| `data/raw/reports/`、`data/staged/reports_v2/` | 中国移动财报原件与分页资料库 |
| `data/staged/jd_chinese_v4/`、`data/staged/m5_chinese_v1/` | 从中文原页构造的算术题（原值、单位、页码、哈希） |
| `/test/Finagent/mineru_home/` | MinerU 模型与配置 |

原始 PDF 仅作本地研究与必要引用，不随仓库再分发。移机时须一并恢复上表资产，仅 clone 仓库不能运行。

## 训练启动方式

单卡直接运行 `src.training.m0_lora_check --config <训练配置.json> --output-dir <输出目录>`；多卡（`gradient_accumulation` 须能被进程数整除）：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python -m torch.distributed.run --nproc_per_node 4 -m src.training.m0_lora_check \
  --config <训练配置.json> --output-dir <输出目录> --stop-after <步数>
```

- 续训加 `--resume <检查点>`；进度与检查点由 rank 0 写入输出目录；同一检查点可在不同卡数之间续训。
- FSDP 加 `--fsdp-frozen-weights`（不与 `--offload-unused-vision` 同用）；DeepSpeed 使用 `src.training.deepspeed_lora` 并加上述 `PYTHONPATH`。
- 同机并发多个分布式训练须使用不同的 `--master_port`。
- 训练入口的数据装载目前只认旧格式的样本，核查任务的训练数据格式在阶段 4 确定后才能使用上述命令（见 [architecture.md](architecture.md)）。

## 本地服务

`configs/model_choice.json` 指向的适配器权重已不在本机，`src.service.server` 在新模型就绪前不能启动。
