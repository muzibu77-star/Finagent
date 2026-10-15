# 环境恢复与运行

更新：2026-10-14。

## 机器与基础环境

- Python 3.12.13；系统提供 Torch 2.11.0+cu128。2026-10-13 本机只识别到 1 张 RTX 4090（24 GB）；多卡训练前须先确认可用显卡数。
- 本地盘 60 GB，空间有限；大文件、pip 缓存与临时目录放 `/test` 盘。2026-10-12 曾因 pip 隔离构建把本地盘写满：编译扩展须关闭构建隔离（`--no-build-isolation`）并把 `TMPDIR` 指到 `/test`。
- 会话重启会终止由会话直接启动的后台任务，耗时任务须脱离会话启动。
- 命令行没有 GitHub 凭据，推送使用编辑器自带的 git 凭据助手。

## 主环境（训练、测试）

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-research.txt
.venv/bin/python -m unittest discover -s tests -q
```

`--system-site-packages` 用于复用系统 Torch。`requirements-research.txt` 只含 MCP（阶段 5 使用）。测试全部在 CPU 上运行，不需要模型权重。

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
- 解析命令（2026-10-13 已在真实年报页上跑通，3 页约 14 秒）：

```bash
MINERU_HOME=/test/Finagent/mineru_home MINERU_MODEL_SOURCE=modelscope \
  .venv-mineru/bin/mineru-kit parse <文件.pdf> -o <输出目录> -p <页码范围> -f middle_json --tier standard
```

  输出 `<输出目录>/<文件名>.json`（`docvortex.middle` 格式），交给 `python -m src.data.table_facts --pdf --middle --output` 做原文映射与事实绑定。模型在 `/test/Finagent/mineru_home/models`。
- **文档模型必须走 GPU 服务**：MinerU 自带的 llama.cpp 在本容器内只能用 CPU（每个区块 30—100 秒）。先启动服务（约 3 分钟就绪，占显存约 10 GB），再给解析命令加三个环境变量：

```bash
V=$PWD/.venv-serving
PATH=$V/bin:$PATH LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cu13/lib \
  nohup setsid $V/bin/vllm serve /test/Finagent/mineru_home/models/MinerU2.5-Pro-2605-1.2B \
  --served-model-name mineru-vlm --host 127.0.0.1 --port 30000 --gpu-memory-utilization 0.35 \
  > /test/Finagent/tmp/vlm_server.log 2>&1 &
# 解析时：
MINERU_MODEL_VLM_SERVER_URL=http://127.0.0.1:30000 MINERU_MODEL_VLM_MODEL=mineru-vlm NO_PROXY=127.0.0.1,localhost
```

  权重用 `mineru-kit models download MinerU2.5-Pro-2605-1.2B` 下载。实测 0.5—0.7 页／秒。停止服务：`pkill -f "vllm serve"`。

## 全量解析任务（2026-10-13 17:41 启动，2026-10-14 01:17 完成）

先按上节启动 GPU 上的文档模型服务，再执行：

```bash
MINERU_HOME=/test/Finagent/mineru_home MINERU_MODEL_SOURCE=modelscope TMPDIR=/test/Finagent/tmp \
MINERU_MODEL_VLM_SERVER_URL=http://127.0.0.1:30000 MINERU_MODEL_VLM_MODEL=mineru-vlm \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
nohup setsid .venv/bin/python -m src.data.parse_reports --output-dir data/staged/cninfo_parsed_v1 \
  > /test/Finagent/tmp/parse_reports.log 2>&1 &
```

- 输出：`data/staged/cninfo_parsed_v1/<来源 id>.json`，每份文件一个；失败记录在同目录 `failures.jsonl`。
- 进度：`tail -f /test/Finagent/tmp/parse_reports.log`，每完成一份输出已完成文件数、页数、已用时间、速度与预计剩余时间。
- 续跑：中断后原样重跑同一条命令，已完整解析的文件自动跳过；半成品为 `*.partial.json`，重跑时覆盖。
- 验收：日志末行 `finished: 92/92 documents parsed`，且 `failures.jsonl` 不存在或为空。
- 停止：`pkill -f "python -m [s]rc\.data\.parse_report"`，再停文档模型服务。

## 不在 Git 中的资产

| 位置 | 内容 |
|---|---|
| `/test/Finagent/models/Qwen3.5-9B` | 现有基座模型，路径与 revision 见 `configs/base_model.yaml`；阶段 4 可能更换 |
| `data/raw/cninfo/` | 30 家发行人 2023—2025 财年年度报告与更正公告（92 个文件、741 MB），清单与哈希见 `configs/cninfo_sources.json`，用 `python -m src.data.cninfo_reports --pause 2.5` 重新获取 |
| `data/raw/chinese_expanded/` | 康美药业 2017、新纶科技 2018 年报及更正公告原件（`configs/jd_restatement*.json` 引用，重建出处库需要）；同目录另有招商银行、中国电信 2023 等旧路线原件 |
| `data/raw/policy/` | 政策文件与统计发布 1,855 篇的原始网页和正文（206 MB），清单见 `configs/policy_sources.json`，用 `python -m src.data.policy_sources` 重新获取 |
| `/test/Finagent/mineru_home/` | MinerU 模型与配置 |
| `data/staged/cninfo_parsed_v1/` | 年报的 MinerU 解析结果（全量解析任务的输出） |
| `data/staged/seed_claims_v2.jsonl`、`data/staged/eval_v2/`（v1 保留作历史） | 种子断言与冻结的评测集（含 `lock.json`）；模型稿件的原始输出与 API 应答缓存在 `artifacts/stage2_drafts_v1/`（v2 目录是同一批应答在新资料库上的重判），重跑命中缓存不再计费 |
| `data/staged/eval_v2_labels/` | 点评数字的逐条核实标注（Codex，2026-10-14）。用户 2026-10-14 决定其中的标注文件、盲审记录、出处库问题清单与汇总进 Git（`git add -f`，因为 `data/` 默认被忽略）；`evidence/` 下的年报页面截图不进 Git。备份在 `/test/Finagent/backups/eval_v2_labels_20261014.tar.gz` |
| `.env` | DeepSeek API 密钥（`DEEPSEEK_API_KEY`），权限 600，不进 Git |
| `data/staged/library_v8/`（当前默认；v7 为验收后改进评测所用，v5 为阶段 3—6 评测所用，v3 保留，Codex 标注的事实编号指向 v3） | 出处库 `library.sqlite`（约 438 MB）与收据；由 `python -m src.data.library --restatement configs/jd_restatement.json --restatement configs/jd_restatement_xinlun.json --output-dir <新目录>` 从上面各项重建，约 6 分钟，不需要 GPU |

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
- 训练入口的数据装载只读训练配置里 `chat_samples` 指向的 JSONL（`{review_id, messages}`，见 [architecture.md](architecture.md)）；`--config` 为必填，阶段 4 的配置是 `configs/stage4_extraction_training.json`，基座模型配置为 `configs/base_model.yaml`。

## 本地模型服务与评测（阶段 4）

```bash
PATH=$PWD/.venv-serving/bin:$PATH CUDA_VISIBLE_DEVICES=0 .venv-serving/bin/vllm serve \
  /test/Finagent/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --served-model-name qwen3.5-9b --port 18090 --max-model-len 16384 --gpu-memory-utilization 0.88 \
  --max-num-seqs 8 --enforce-eager --language-model-only \
  --enable-lora --max-lora-rank 16 --lora-modules qwen3.5-9b-extract=/test/Finagent/runs/stage4_extract_v1/step_0072
```

- LoRA 内核运行时编译需要 `ninja`，它在 `.venv-serving/bin`，所以 PATH 必须包含该目录。端口 8000 被 sshd 占用。
- 机器设置了 HTTP 代理：访问本地服务的进程须设 `NO_PROXY=127.0.0.1,localhost`，否则返回 502。
- 评测：`run_reviews --model qwen3.5-9b-extract --url http://127.0.0.1:18090/v1/chat/completions --budget-usd 0 --workers 8`；单卡生成约 120 token/秒，开发集 66 篇约 1 小时，多卡时可每卡一个服务、按稿件分片后合并缓存再整集计分。
- 训练产物：`/test/Finagent/runs/stage4_extract_v1/`（适配器 `step_0072`、日志 `stage4_extract_v1.log`）。
- `.venv` 已安装 `langgraph==1.2.15`、`langchain-core==1.6.10`、`langchain-openai==1.7.1`（2026-10-15，阶段 5 编排用，见 `requirements-research.txt`）。
