# Claude ↔ Codex 任务交接

Claude 在这里派工并记录验收；Codex 在这里领任务并回报。项目事实以 `memory-bank/` 为准，工程规则以 [AGENTS.md](AGENTS.md) 为准，本文件只放任务与验收。

## 给 Codex 的约定

1. 开工前按 AGENTS.md 第 2 节读 `memory-bank/` 与相关代码；只做“待执行”状态的任务，一次一个。
2. 只改任务列出的文件；需要改别的文件、发现任务描述与代码不符、或验收标准无法达到时，停下在“回报”里写明，不自行扩大范围。
3. 不提交、不推送、不安装依赖、不删除数据资产、不启动训练／推理／大规模评测、不调用付费 API，除非任务明确写了允许。
4. 不新增无人调用的模块、配置项或兼容分支；能复用现有函数就不新写。
5. 完成后把状态改为“待验收”，在“回报”里写：改了哪些文件、验证命令及其输出摘要、未完成或有疑问的地方。

## 状态

`待批准`（用户尚未同意）→ `待执行` → `进行中` → `待验收` → `已验收`／`退回`（附原因，修正后回到 `待验收`）。

## 任务

用户 2026-10-14 批准整理：旧路线（英文财报问答 Agent）除受控计算 `src/tools/calculator.py` 与其依赖 `src/evidence/models.py` 外全部删除。按 T1 → T2 顺序执行。工作树里阶段 4 未提交的改动（`extraction_samples.py`、`m0_lora_check.py`、`run_reviews.py`、`api_client.py`、`configs/stage4_extraction_training.json` 及其测试）必须保留。

### T1 训练入口与旧路线解耦（已验收）

- **目标**：阶段 4 的训练入口不再依赖旧英文问答路线的模块。
- **现状问题**：`src/training/m0_lora_check.py` 的 `prepare_chat_samples` 从 `src/training/prepare_sft.py` 导入 `encode_messages`，后者顶层导入 `benchmark_tools`、`financial`、`compact`、FinQA vendor 代码；`prepare_samples` 仍保留旧 `manifest.jsonl` 分支及 `autonomous_sft`、`preference_data`、DPO 分支。
- **做法**：把 `encode_messages` 及其只为它服务的辅助函数原样移到 `src/training/chat_encoding.py`（行为不变），对应测试随之迁移；`m0_lora_check.py` 只保留 `chat_samples` 数据装载，删除 manifest、`autonomous_supervision`、`preference_data`／DPO 分支及不再使用的配置键读取；`src/training/deepspeed_lora.py` 继续可用（它从 `m0_lora_check` 导入 `prepare_samples`、`validated_initial_adapter`）。
- **禁止**：不改损失、掩码、采样顺序、检查点格式与 FSDP／DeepSpeed 逻辑；不加载模型、不启动训练。
- **验收**：
  - `pytest -q tests/test_chat_samples.py tests/test_training_loss.py tests/test_distributed_training.py` 通过；
  - `python -c "import sys, src.training.m0_lora_check; print(sorted(m for m in sys.modules if m.startswith('src.')))"` 只列出 `src.training.*`；
  - `configs/stage4_extraction_training.json` 无需修改；`git diff` 中 `m0_lora_check.py` 只有删除与导入路径变化。

### T2 删除旧路线模块（已验收）

**Claude 对回报两处问题的决定（2026-10-14）**：

1. `terms` 原样移入 `src/data/library.py`（只保留它这个调用方），删除 `from src.retrieval.report_index import terms`；然后按清单删 `src/retrieval/` 与 `src/data/leakage.py`。
2. 新建 `configs/base_model.yaml`，只含 `configs/m0_inference.yaml` 的 `model:` 段（`repo_id`、`revision`、`path`、`device` 原值照抄），文件头一行注释说明这是训练入口读取的基座模型配置；`configs/stage4_extraction_training.json` 的 `model_config` 改指向它（这是 T1“该配置无需修改”的例外，只改这一个值）。之后按清单删 `m0_inference.yaml` 与两个 `m0_smoke_*` 文件。
3. `src/training/m0_lora_check.py` 的 `--config` 改为 `required=True`（与 `deepspeed_lora.py` 一致），去掉指向已删 `m0_training.json` 的默认值。
4. `requirements-*.txt` 本次不改；回报中列出的依赖由 Claude 向用户申请后另行处理。

验收在下列原有标准外加一条：`.venv/bin/python -c "import yaml; print(yaml.safe_load(open('configs/base_model.yaml'))['model'])"` 与原 `m0_inference.yaml` 的 `model` 段完全相同。环境中没有 pytest 时，用 `.venv/bin/python -m unittest discover -s tests -q` 代替。

- **删除**（`git rm`，不动 `data/`、`artifacts/`、`models/` 与 `/test` 下资产）：
  - `src/agent/`、`src/harness/`、`src/service/`（含 `index.html`／`app.js`／`style.css`）、`src/retrieval/`、`src/evaluation/vendor/`、`skills/financial-research/`
  - `src/tools/benchmark_tools.py`；`src/evidence/compact.py`、`src/evidence/identity.py`；`src/model/agent.py`、`src/model/financial.py`、`src/model/http_driver.py`
  - `src/data/`：`build_manifest.py`、`tatqa_source_audit.py`、`corpus.py`、`report_corpus.py`、`rag_corpus.py`、`reports.py`、`prepare_chinese.py`、`expand_chinese.py`、`leakage.py`
  - `src/evaluation/`：`m0_inference_check.py`、`validate_agent_run.py`、`compare_agent_runs.py`、`statistics.py`、`restatement_run.py`、`serving_benchmark.py`
  - `src/training/`：`prepare_sft.py`、`autonomous_sft.py`、`complex_autonomous.py`、`teacher.py`、`preference_data.py`、`preference_loss.py`
  - `configs/`：`jd_action_3072_training.json`、`jd_chinese_sources.json`、`jd_chinese_sources_v2.json`、`jd_dpo_protocol.json`、`jd_evaluation_protocol.json`、`jd_retrieval.json`、`m0_data_lock.json`、`m0_inference.yaml`、`m0_smoke_tasks.jsonl`、`m0_smoke_tools.json`、`m0_training.json`、`m3_acceptance_lock.json`、`m5_chinese_tasks.json`、`model_choice.json`、`report_retrieval_choice.json`、`report_sources.json`、`tatqa_sources.json`
  - 只测试上述模块的测试文件；同时测试保留模块的测试文件只删相关用例并在回报中列出。
- **保留**：`jd_restatement.json`、`jd_restatement_xinlun.json`（出处库的更正案例输入）；`src/tools/calculator.py`、`src/evidence/models.py` 及其测试。
- **文档**：`memory-bank/architecture.md`、`environment.md`、`README.md` 删去被删模块与配置的描述；`memory-bank/note.md` 与 `result.md` 不改。`requirements-*.txt` 中只被已删模块使用的依赖列在回报里，不改文件。
- **验收**：
  - `pytest -q` 全部通过，回报测试数；
  - `grep -rnE "<被删模块名>" src tests configs README.md memory-bank/architecture.md memory-bank/environment.md` 无结果；
  - `python -m compileall -q src` 无错误；每个保留的命令行入口 `python -m <模块> --help` 正常退出。

### T3 删除旧路线资产与无用依赖（已验收）

用户 2026-10-14 批准删除（本任务明确允许删除下列数据资产，其余资产不动）。

- **删除磁盘资产**（均不在 Git 中，`rm -rf`，删除前逐个 `du -sh` 记录大小）：`models/bge-m3`、`models/bge-reranker-v2-m3`、`data/raw/reports/`、`data/staged/reports_v2/`、`data/staged/jd_chinese_v4/`、`data/staged/m5_chinese_v1/`。
- **不删**：`data/raw/chinese_expanded/` 整个目录（康美、新纶原件被 `configs/jd_restatement*.json` 引用）；`models/` 下其他内容；`/test/Finagent/` 下任何内容。
- **依赖**：`requirements-research.txt` 删去 `qdrant-client`、`scipy`、`pandas` 三行，保留 `mcp`（阶段 5 使用）。只改文件，不卸载已安装的包。
- **文档**：`memory-bank/environment.md` 资产表删掉“旧路线遗留资产”那一行；如该文件其他地方提到上述目录或三个依赖，一并删去。
- **验收**：
  - 上述 6 个路径均不存在，`data/raw/chinese_expanded/kangmei2017.pdf`、`xinlun2018.pdf` 仍在；
  - `grep -rnE "bge-m3|bge-reranker|reports_v2|jd_chinese_v4|m5_chinese_v1|data/raw/reports|qdrant|scipy|pandas" src tests configs requirements*.txt memory-bank/architecture.md memory-bank/environment.md` 无结果；
  - `.venv/bin/python -m unittest discover -s tests -q` 通过；回报中写明删除前各路径大小与释放的总空间。

### T4 训练损失省显存，并完成 4 卡 FSDP 试跑（已验收）

**本任务明确允许**：使用 4 张 GPU 运行下面的试跑与对拍命令；不允许正式训练（`--stop-after` 不得大于 1）、安装依赖或提交 Git。

- **现象**（Claude 2026-10-14 试跑，日志 `/test/Finagent/runs/stage4_probe.log`）：4×4090、`--fsdp-frozen-weights --checkpointing-min-tokens 1`，第一条样本（最长，9,735 token）在 `src/training/loss.py:23` 的 `F.cross_entropy` 申请 7.16 GiB 时 OOM（rank 0 已占 19.8 GiB）。原因：`assistant_loss` 先算出全序列 BF16 logits（词表约 24.8 万，约 4.8 GB），再把受监督位置整体转 FP32。
- **做法**（只改 `src/training/loss.py` 及其测试）：
  1. 不再生成全序列 logits：取最后一层隐状态（经过最终 norm、进入 `lm_head` 之前的那个张量），只取受监督位置 `positions`；
  2. 按固定块大小（如 1,024 个位置）分块：每块 `lm_head` → `.float()` → `cross_entropy(reduction='sum')`，每块用 `torch.utils.checkpoint` 包住，使反向时只重算当前块；最后总和除以受监督位置数。
  3. 函数签名、返回值含义（受监督目标上的平均 FP32 交叉熵）、`labels[t+1]` 的移位规则、报错条件保持不变；更新模块与函数 docstring，删掉已不成立的“保持全序列 GEMM 形状”注释。
  4. 取隐状态的方式必须对 PEFT 包装后的 Qwen3.5（`AutoModelForMultimodalLM`）成立；用 `model.get_output_embeddings()` 取输出层。不要用 `strict=False`、`.detach()` 等手段绕过问题。
- **验收**：
  - 单元测试：用一个很小的随机初始化因果语言模型（CPU 即可）对比新旧实现，损失与 LoRA 参数梯度在 FP32 下相对误差 ≤ 1e-5；
  - 真实模型对拍：单卡加载 Qwen3.5-9B，取训练集中一条约 2,000 token 的样本，新旧实现的损失相对误差 ≤ 1e-3（BF16 下），写出两个值；
  - 4 卡试跑：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m torch.distributed.run --nproc_per_node 4 --master_port 29512 -m src.training.m0_lora_check --config configs/stage4_extraction_training.json --output-dir /test/Finagent/runs/stage4_probe_t4 --stop-after 1 --fsdp-frozen-weights --checkpointing-min-tokens 1`，日志写到 `/test/Finagent/runs/stage4_probe_t4.log`。要求正常结束、损失有限，回报每个 rank 的峰值显存（`torch.cuda.max_memory_allocated`，若入口已记录就引用）与这一步的耗时；
  - 若试跑仍 OOM，再试一次 `--checkpointing-min-tokens` 以外不改任何参数的情况下把分块减小到 256；还不行就停下回报，不要改训练配置；
  - `.venv/bin/python -m unittest discover -s tests -q` 通过。
- **另报**：日志里 transformers 提示 `flash-linear-attention` 未安装、`chunk_gated_delta_rule` 走参考实现会很慢。只记录这一步的耗时，不要安装。

### T5 合并 LoRA 适配器，供 vLLM 评测训练后模型（已验收）

**背景**：阶段 4 正式训练正在 4 张 GPU 上运行（输出 `/test/Finagent/runs/stage4_extract_v1/`，日志同名 `.log`），**不要使用 GPU、不要碰这个进程和目录**。训练后要用 `.venv-serving` 的 vLLM 在开发集上评测（与免训练 9B 的 `artifacts/stage4_9b_zero_dev_v1/` 同一流程）。为避开 vLLM 对 Qwen3.5 多模态 LoRA 的兼容性风险，评测用合并后的完整权重。

- **做法**：
  1. 新建 `src/training/merge_adapter.py`（命令行：`--model-config configs/base_model.yaml --adapter <检查点目录> --output <目录>`）：在 CPU 上以 BF16 加载基座（`AutoModelForMultimodalLM`，`local_files_only=True`），加载适配器后 `merge_and_unload()`，`save_pretrained` 到输出目录，并把基座目录中 tokenizer、`chat_template.jinja`、preprocessor 等非权重文件原样复制过去；输出目录已存在时拒绝；写 `merge_receipt.json`（基座路径与 revision、适配器文件 sha256、输出权重文件列表）。
  2. `src/model/api_client.py` 的 `LOCAL` 增加 `'qwen3.5-9b-extract'`（价格 0），并补测试。
- **验证**（只用 CPU）：
  - 用试跑检查点 `/test/Finagent/runs/stage4_probe_t4/step_0001` 合并到 `/test/Finagent/models/qwen35-9b-extract-probe`；
  - CPU 上取一条约 50 token 的短输入，比较“基座+适配器（未合并）”与“合并后模型”最后位置的 logits：最大绝对差与 argmax 是否一致，写入回报；
  - 合并后的 `config.json` 与基座 `config.json` 相比只允许 `torch_dtype`／`transformers_version` 一类元数据不同，列出差异；
  - 验证完删除 `/test/Finagent/models/qwen35-9b-extract-probe`（本任务允许删除这个由你创建的目录）；
  - `.venv/bin/python -m unittest discover -s tests -q` 通过。
- **禁止**：不改训练、损失与核查代码；不启动 vLLM；不安装依赖；不提交 Git。

### T6 把“抽取结果 → 核查结论”抽成可复用函数（已验收）

**背景**：阶段 5 要把核查流程放进 LangGraph 编排、服务页面和 MCP 工具；这段逻辑现在写死在 `src/evaluation/run_reviews.py` 的 `main()` 里。GPU 正在跑评测，**不要使用 GPU，不要碰 `artifacts/stage4_9b_sft_dev_v1/` 与 vLLM 进程**。

- **做法**：
  1. 新建 `src/verify/pipeline.py`，提供 `judge(checker, text, items, code, published) -> dict[int, dict]`：输入一篇稿件的原文、抽取模型输出经 `parse()` 得到的 items、发行人代码与写稿截止日，输出“数字在原文中的起始位置 → 结论字典”。函数体就是 `run_reviews.py` 中 `for item in ... parse(answer['content'])` 那段循环（从 `locate` 到 `checker.check` 与各个 `unsupported` 原因），原样搬过去，键与值的结构不变。
  2. `run_reviews.py` 改为调用 `judge()`；`--direct` 分支与计分部分不动。
  3. 给 `judge()` 写单元测试：用一个内存 SQLite 小出处库（参照现有 `tests/test_checker.py` 的造库方式）覆盖 amount 一致、amount 不一致、growth、ratio、母公司、季度措辞、科目与措辞不符、数字写法不规范八种分支。
- **行为不变的验收**（零费用，全部命中缓存）：
  - `mkdir -p /test/Finagent/runs/t6_check && cp artifacts/stage3_reviews_dev_v6/api_cache.jsonl /test/Finagent/runs/t6_check/`，然后
    `.venv/bin/python -m src.evaluation.run_reviews --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --labels data/staged/eval_v2_labels/development_review_labels.jsonl --budget-usd 0.01 --output-dir /test/Finagent/runs/t6_check`；
  - `cmp /test/Finagent/runs/t6_check/records.jsonl artifacts/stage3_reviews_dev_v6/records.jsonl` 与 `summary.json` 均完全相同（`cost_usd` 应为缓存中已付金额，与原值相同）；如不同，停下回报差异，不要改核查逻辑；
  - `.venv/bin/python -m unittest discover -s tests -q` 通过。
- **禁止**：不改 `checker.py`、`extract.py` 的行为；不新增抽象层；不调用会产生费用的 API 请求（缓存未命中即视为失败，停下回报）。

### T7 LangGraph 核查流程与工作底稿（已验收）

**背景**：阶段 5 的产品形态。已安装 `langgraph==1.2.15`、`langchain-core==1.6.10`、`langchain-openai==1.7.1`（见 `requirements-research.txt`）。GPU 正在跑评测，**不要使用 GPU，不要碰 vLLM 进程与 `artifacts/stage4_9b_sft_dev_v1/`**。本任务允许调用 `deepseek-flash`，总花费 ≤ 0.5 美元（经 `ApiClient` 计量）。

- **做法**：
  1. `src/verify/pipeline.py` 的 `judge()` 输出里加入 `'evidence': verdict.evidence`（事实编号列表；只在调用了 `checker.check` 的分支出现），其余不变；补测试。
  2. 新建 `src/agent/llm.py`：一个 LangChain `BaseChatModel` 子类，内部持有现有 `ApiClient`（沿用缓存、计价、预算与本地模型规则），把 LangChain 消息转为 `ApiClient.chat` 的消息格式并返回 `AIMessage`。不另写 HTTP 客户端。
  3. 新建 `src/agent/graph.py`：用 `langgraph.graph.StateGraph` 编排五个节点，状态用 `TypedDict`：
     - `extract`（抽取 Agent，只看稿件和科目清单）：`extract.PROMPT` + `line_items` → 模型 → `parse()`；
     - `check`（代码）：`judge()`；
     - `consistency`（代码，稿内一致性）：同一稿件中 `type`、`label`、`year`、`unit` 都相同而书写数值不同的 items 互相标为 `intra_document`，记下双方位置；
     - `revise`（修订 Agent）：对每个 `inconsistent` 且有 `expected` 的数字，把它所在的句子（按中文句号／分号切分）连同应有值交给模型，要求只替换这个数字给出改写句；代码检查改写句包含应有值的书写且除该数字外与原句相同，不满足就丢弃建议并标“需人工改写”。修订 Agent 不能改变任何结论；
     - `report`（代码）：生成工作底稿。
     图入口函数 `review_draft(draft: dict, *, library: Path, model: str, url: str, budget_usd: float, cache: Path) -> dict`，`draft` 含 `text`、`code`、`company`、`fiscal_year`、`as_of`。
  4. 工作底稿（`report` 节点输出、可 `json.dumps`）：每个数字一条——在稿件中的位置与原文、结论、错因、应有值、证据（由 `evidence` 事实编号查出库中的文件、页码、字符区间、行标签、列表头、单位）、修改建议、稿内一致性发现；另加汇总计数（一致／不一致／无依据／存疑／抽取遗漏需人工）。再提供 `workpaper_markdown(workpaper) -> str`，给人读的表格版。
  5. 命令行 `python -m src.agent.graph --library ... --reviews <jsonl> --review-id <id> --output <json>`，从评测集稿件取 `draft`（`as_of` 用该年报的披露日，取法同 `run_reviews.py`）。
- **验收**：
  - 单元测试（不联网）：用假模型（返回固定 JSON）跑通整张图，覆盖修订建议被接受与被丢弃两种情况、稿内一致性发现、底稿证据字段；
  - 真实跑一篇开发集稿件（选一篇 `closed_book` 为真、含不一致数字的），flash，输出 `/test/Finagent/runs/t7_demo.json` 与 `.md`；回报其汇总计数，并与 `artifacts/stage3_reviews_dev_v6/records.jsonl` 中同一稿件的结论逐条比较（抽取若命中缓存应完全一致；说明任何差异）；
  - `.venv/bin/python -m unittest discover -s tests -q` 通过；回报花费。
- **禁止**：不改 `checker.py`、`extract.py`；不新增配置文件；不实现服务页面与 MCP（后续任务）。

## 回报

### T1（已验收）

改了哪些文件：

- 新增 `src/training/chat_encoding.py`：`encode_messages` 原样搬入（行为不变）。
- `src/training/prepare_sft.py`：删除 `encode_messages` 定义，改为从 `chat_encoding` 导入（旧路线模块在 T2 前仍可导入；该文件本身在 T2 删除）。
- `src/training/m0_lora_check.py`：`prepare_chat_samples` 合并为唯一的 `prepare_samples`（只保留 `chat_samples` 装载）；删除 manifest／`prepared_samples`／`autonomous_supervision`／DPO 分支及 `data_dir`、`prepared_samples`、`complex_training_*`、`training_objective`、`beta`、`baseline_root`、`reference_adapter_sha256` 等配置键读取；`encode_messages` 改为顶层从 `chat_encoding` 导入。`deepspeed_lora.py` 的 `prepare_samples`、`validated_initial_adapter` 导入不变且可用。
- 测试：删除 `tests/test_prepared_read_boundary.py`（整个文件只覆盖已删的 manifest／`prepared_samples` 分支）；`tests/test_training_initialization.py` 删去只覆盖已删分支的 `test_explicit_autonomous_parent_contract_and_legacy_rejection`，保留 `validated_initial_adapter` 的用例。`encode_messages` 没有专门测试，无可迁移的用例。

验证命令及输出摘要：

- 环境里没有 pytest（`.venv` 未安装，安装依赖未获授权），按 AGENTS.md 用等价 unittest 入口：`.venv/bin/python -m unittest tests.test_chat_samples tests.test_training_loss tests.test_distributed_training` → `Ran 9 tests ... OK`。
- `.venv/bin/python -c "import sys, src.training.m0_lora_check; print(sorted(m for m in sys.modules if m.startswith('src.')))"` → `['src.training', 'src.training.chat_encoding', 'src.training.distributed', 'src.training.loss', 'src.training.m0_lora_check']`，只有 `src.training.*`。
- 全量 `.venv/bin/python -m unittest discover -s tests -q` → `Ran 331 tests ... OK`（T1 前 333，差值为上面删掉的 2 个只覆盖已删分支的用例）。
- `.venv/bin/python -m compileall -q src` 无错误；`git diff --check` 无问题；`configs/stage4_extraction_training.json` 未改动（仍是未跟踪的新增文件）。
- `m0_lora_check.py` 相对工作树此前状态只有删除；相对 HEAD 除删除外只多 1 行导入，另有 3 处把仅服务 DPO 的条件表达式换成等价的 SFT 写法（`needs_offload` 的 `tokens`、`batch` 去掉 `side` 参数、nonfinite 失败的 `tokens`）。

未完成或有疑问：无。

### T2（已验收）

改了哪些文件：

- 新增 `configs/base_model.yaml`：只含原 `configs/m0_inference.yaml` 的 `model:` 段（`repo_id`／`revision`／`path`／`device` 原值照抄）加一行说明注释；`configs/stage4_extraction_training.json` 只把 `model_config` 改指向它。
- `src/data/library.py`：`terms` 原样移入（并加 `import re`），删除 `from src.retrieval.report_index import terms`。
- `src/training/m0_lora_check.py`：`--config` 改为 `required=True`。
- 删除（`git rm`，共 60 个 tracked 文件）：`src/agent/`、`src/harness/`、`src/service/`（含 `index.html`／`app.js`／`style.css`）、`src/retrieval/`、`src/evaluation/vendor/`、`skills/financial-research/`；`src/tools/benchmark_tools.py`、`src/evidence/compact.py`、`src/evidence/identity.py`、`src/model/agent.py`、`src/model/financial.py`、`src/model/http_driver.py`；`src/data/` 的 9 个模块；`src/evaluation/` 的 6 个模块；`src/training/` 的 6 个模块；`configs/` 的 17 个配置。剩下的空目录已 `rmdir`。
- 测试：删掉 31 个只测试已删模块的文件（`test_agent_audit`、`test_agent_comparison`、`test_agent_context`、`test_autonomous_sft`、`test_benchmark_tools`、`test_build_manifest`、`test_calculation_precision`、`test_calculation_references`、`test_compact`、`test_complex_parents`、`test_complex_trace`、`test_document_prompt`、`test_evidence_identity`、`test_financial_model`、`test_fixed_runner`、`test_harness_feedback`、`test_http_driver`、`test_inline_facts`、`test_leakage`、`test_m0_inference_check`、`test_model_lock`、`test_neural_retrieval`、`test_preference_data`、`test_preference_loss`、`test_prepare_sft`、`test_report_corpus`、`test_runner`、`test_service`、`test_service_model_choice`、`test_store`、`test_tatqa_source_audit`）。保留模块的测试文件本次无需再删用例——唯一涉及的 `tests/test_prepared_read_boundary.py` 与 `tests/test_training_initialization.py` 已在 T1 处理。
- 文档：`memory-bank/architecture.md`、`memory-bank/environment.md` 删去已删模块与配置的描述，并把基座配置指向改为 `configs/base_model.yaml`；`README.md` 无相关描述，未改；`note.md`、`result.md` 未改。`requirements-*.txt` 未改。

验证命令及输出摘要：

- `pytest` 不存在，按验收说明用 `.venv/bin/python -m unittest discover -s tests -q` → `Ran 119 tests ... OK`。
- grep（被删模块名与被删配置名，范围 `src tests configs README.md memory-bank/architecture.md memory-bank/environment.md`）→ 无结果（exit 1）。
- `.venv/bin/python -m compileall -q src` → 无错误。
- 13 个保留入口 `python -m <模块> --help` 全部 rc=0（`cninfo_reports`、`library`、`parse_reports`、`policy_sources`、`table_facts`、`draft_pilot`、`freeze_set`、`run_checker`、`run_reviews`、`seed_claims`、`splits`、`m0_lora_check`、`extraction_samples`）；`src.training.deepspeed_lora --help` 加 `PYTHONPATH=artifacts/deepspeed_env/site` 后 rc=0（不加则 `ModuleNotFoundError: deepspeed`，是 environment.md 既有的环境约定）。
- `.venv/bin/python -c "import yaml; print(yaml.safe_load(open('configs/base_model.yaml'))['model'])"` 与原 `configs/m0_inference.yaml` 的 `model` 段完全相同；`terms` 与 `report_index.py` 原定义经 AST 比较逐字相同。

未完成或有疑问：

- `memory-bank/progress.md` 与 `plan.md` 的“现有代码基础”仍写着已删模块（神经检索与重排、SQLite 任务持久化、服务页面等）；T2 的文档清单只列 architecture／environment／README，grep 验收也不含这两个文件，故按约定未改，需要时请下一条指令。
- `models/bge-m3`、`models/bge-reranker-v2-m3` 资产仍在磁盘（未删），但唯一使用方（检索）与其配置已删，故从 environment.md 资产表中移除该行。
- `requirements-*.txt` 中只被已删模块使用的依赖（不改文件）：`qdrant-client`（原仅 `src/retrieval/neural.py`）、`mcp`（原仅 `src/service/mcp_server.py`）；`scipy`、`pandas` 已无任何 import；`playwright` 无使用入口。`Pillow`／`accelerate`／`bitsandbytes` 未被直接 import。`vllm` 系列属 `.venv-serving`，仍用于 MinerU 文档模型服务（保留的 `parse_reports` 需要），不可删。

### T3（已验收）

改了哪些文件：

- 删除磁盘资产（均不在 Git 中，用 `rm -r`）：`models/bge-m3`、`models/bge-reranker-v2-m3`、`data/raw/reports/`、`data/staged/reports_v2/`、`data/staged/jd_chinese_v4/`、`data/staged/m5_chinese_v1/`。`models/` 现在是空目录（仍被 `.gitignore` 忽略），该目录本身未删。
- `requirements-research.txt`：删去 `qdrant-client==1.15.1`、`scipy==1.17.1`、`pandas==3.0.1` 三行，保留 `mcp==1.26.0`；只改文件，未卸载已安装的包。
- `memory-bank/environment.md`：删去资产表“旧路线遗留资产”那一行；“主环境”一节的 `requirements-research.txt` 含 Qdrant 客户端、MCP、SciPy、pandas 改为“只含 MCP（阶段 5 使用）”。该文件无其他相关提及。

验证命令及输出摘要：

- 删除前逐个 `du -sh`：`models/bge-m3` 2.2G、`models/bge-reranker-v2-m3` 2.2G、`data/raw/reports/` 3.4M、`data/staged/reports_v2/` 72K、`data/staged/jd_chinese_v4/` 2.7M、`data/staged/m5_chinese_v1/` 420K，合计 4.3G；精确字节合计 4,615,479,380 B（约 4.30 GiB）。
- 6 个路径均不存在；`data/raw/chinese_expanded/kangmei2017.pdf`（3.9 MB）、`xinlun2018.pdf`（15.6 MB）仍在。
- `grep -rnE "bge-m3|bge-reranker|reports_v2|jd_chinese_v4|m5_chinese_v1|data/raw/reports|qdrant|scipy|pandas" src tests configs requirements*.txt memory-bank/architecture.md memory-bank/environment.md` → 无结果（exit 1）。
- `.venv/bin/python -m unittest discover -s tests -q` → `Ran 119 tests ... OK`（与 T2 后一致）。

未完成或有疑问：

- `models/` 现为空目录，未 `rmdir`；如要一并删除请下指令。
- `requirements-research.txt` 删除的三行对应包仍装在 `.venv` 里（按约定“不卸载已安装的包”）；保留代码中已无 `qdrant`/`scipy`/`pandas` 的 import。

### T4（已验收）

改了哪些文件：

- `src/training/loss.py`：`assistant_loss` 不再生成全序列 logits。用 `model.get_decoder()` 取经过最终 norm、进入 `lm_head` 之前的 `last_hidden_state`（Qwen3.5 下即 `Qwen3_5TextModel` 的输出），只对受监督位置分块投影：每块 `CHUNK_POSITIONS = 1024` 个位置，`lm_head`（由 `model.get_output_embeddings()` 取）→ `.float()` → `F.cross_entropy(reduction='sum')`，每块用 `torch.utils.checkpoint(..., use_reentrant=False)` 包住，最后总和除以受监督位置数。函数签名、返回值含义、`labels[t+1]` 移位规则、两处报错条件不变；模块与函数 docstring 已更新，删去“保持全序列 GEMM 形状”的注释。
- `tests/test_training_loss.py`：改为用真实的小型 Qwen3.5（缩小配置）+ PEFT LoRA 包装后与旧的“全序列 logits”实现逐项对拍，并校验空标签与多批量报错。
- **计划外的一处实现增补（请重点复核）**：按 T4 原文实现后，4 卡试跑在第一个更新卡死。原因：`lm_head` 是 FSDP 分片单元，调用一次即 all-gather 一次权重，而各进程的样本受监督长度不同，调用次数随之不同（第一次更新实测 12／7／8／8），集合通信次序错位。已在 `loss.py` 内加 `agreed_chunk_count()`：先对本次样本的块数做一次 all-reduce 取最大值（复用 `src/training/distributed.py` 的 `reduce_scalar`），各进程按同一块数循环，位置不足的进程用空块补齐；空块照常调用 `lm_head`（保持集合通信同步、反向重算也同步），只贡献 0 到总和。`distributed.py` 未改。

验证命令及输出摘要：

**单元测试（CPU、FP32）**

- `.venv/bin/python -m unittest discover -s tests -q` → `Ran 119 tests ... OK`。
- `tests.test_training_loss` 在 4 种分块方案（1,024 单块、2 位置两块、1 位置四块、2 位置 3 块加 2 个空填充块）下与旧实现比较，损失与全部 LoRA 参数梯度满足 `torch.testing.assert_close(rtol=1e-5, atol=1e-8)`。

**真实模型对拍（单卡 4090、BF16）**，日志 `runs/stage4_compare_t4.log`、`runs/stage4_compare_t4_chunks.log`

- 训练集 192 条样本长度 1,680—9,735 token（中位 5,370），没有“约 2,000 token”的样本，取最接近的 `review-pro_open-601899-1225550234`（1,680 token、392 个受监督位置）。
- 旧实现 `0.13742344081401825`；新实现（块 1,024）`0.13742344081401825`；新实现（块 64，7 块）`0.13742344081401825`——相对误差 0（单卡且块数一致时逐位相同）。
- 该次单卡峰值 22,239 MiB（含基座 17.5 GiB 加载）。

**4 卡 FSDP 试跑**，日志 `runs/stage4_probe_t4.log`（命令与 T4 完全一致）

- `exit_code=0`；`metrics.jsonl`：`{'step': 1, 'total': 72, 'loss': 0.02925222204066813, 'elapsed_s': 38.44887132011354, 'eta_s': 0.0, 'peak_allocated_mib': 14306.11669921875, 'world_size': 4, 'peak_allocated_mib_any_process': 14306.11669921875, 'frozen_weights': 'fsdp'}`；损失有限，`step_0001` 检查点已保存。
- 单步（一次更新 8 个样本、每卡 2 个）耗时 38.4 秒。日志仍有 `flash-linear-attention`／`causal_conv1d` 未安装、`chunk_gated_delta_rule` 走参考实现的提示，按 T4 只记录耗时、未安装。
- 每 rank 峰值显存（`torch.cuda.max_memory_allocated`）：rank 0 = 14,306 MiB、rank 1 = 11,456 MiB、rank 2 = 11,363 MiB、rank 3 = 11,784 MiB。入口本身只记录 rank 0 峰值与跨进程最大值（均为 14,306 MiB），单 rank 数值来自同一命令的第二次运行（`PYTHONPATH` 加一个只在退出时打印峰值的 `sitecustomize`，日志 `runs/stage4_probe_t4_peaks.log`），该次 `loss` 相同、`elapsed_s = 38.637`。各 rank `max_memory_reserved` 约 23.4 GiB，是分片前整模型 17.5 GiB 加载留下的缓存，不是训练峰值；`nvidia-smi` 采样同样在加载阶段见到约 23.2 GiB、训练步内 12.3—14.3 GiB。

未完成或有疑问：

1. 第一次试跑（未改的 `loss.py`）是集合通信错位导致的死锁，不是慢：NCCL watchdog 在 600 秒后报 `_ALLGATHER_BASE` 超时并打印 FSDP `unshard` 堆栈，各 rank 已完成序号 39／42／40 不一致（摘录 `runs/stage4_probe_t4_deadlock_excerpt.txt`；该次日志已被同命令的第二次运行覆盖）。任务要求不中途杀进程，我等到 watchdog 确认错位后才 SIGTERM 结束它，随后改成上面带空块的方案。
2. 空块的代价：短样本的进程要按最长样本的块数多调用几次 `lm_head`（本步每 rank 12 次），每次约 2 GiB 权重 all-gather。想免掉这部分需要让 `lm_head` 不参与 FSDP 分片（要改 `distributed.py`，超出 T4 文件范围）；我试过在 loss 里用 `DTensor.full_tensor()` 只聚合一次，FSDP2 下直接段错误，不可行。
3. 若认可 `agreed_chunk_count` 这处增补，`loss.py` 即可定稿；若希望换成与数据无关的固定块数等其他同步方式，请下指令。
4. 试跑输出在 `runs/stage4_probe_t4`（T4 指定目录）；为取单 rank 峰值又跑了一次到 `runs/stage4_probe_t4_peaks`，两次都只跑 `--stop-after 1`。
5. `memory-bank/` 未改动（T4 只列 `loss.py` 及其测试）；未提交 Git、未安装依赖。
6. `src/training/deepspeed_lora.py` 也从 `loss.py` 导入同一个 `assistant_loss`，改成 `get_decoder()`／`get_output_embeddings()` 后我没有跑过 DeepSpeed ZeRO-3 路径（需要 `artifacts/deepspeed_env` 与多卡，T4 未授权该后端）；如需冒烟测试请另行授权。

### T5（已验收）

改了哪些文件：

- 新建 `src/training/merge_adapter.py`：必填项 `--model-config`／`--adapter`／`--output`；`--output` 已存在时在加载模型前直接拒绝。流程：CPU 上以 BF16 `AutoModelForMultimodalLM.from_pretrained(..., local_files_only=True)` 载入基座 → `PeftModel.from_pretrained` → `merge_and_unload()` → `save_pretrained` 到新目录；再把基座目录中除权重（`*.safetensors`／`*.bin`／`*.pt`／`*.pth`／`*.ckpt` 及 `*.index.json`）与 `config.json` 之外的文件原样复制（tokenizer、`chat_template.jinja`、preprocessor、video preprocessor 等）；写 `merge_receipt.json`（基座路径与 revision、`adapter_*` 文件 sha256、输出权重文件列表、复制的文件列表）。
- `src/model/api_client.py`：`LOCAL` 增加 `'qwen3.5-9b-extract'`（价格 0）。
- `tests/test_api_client.py`：新增 `test_extract_local_model_is_free`。

验证命令及输出摘要（全程 CPU，命令都带 `CUDA_VISIBLE_DEVICES=`；未使用 GPU，未触碰正在运行的训练进程与目录）：

- 合并：`CUDA_VISIBLE_DEVICES= .venv/bin/python -m src.training.merge_adapter --model-config configs/base_model.yaml --adapter /test/Finagent/runs/stage4_probe_t4/step_0001 --output /test/Finagent/models/qwen35-9b-extract-probe` → 成功。输出目录为 18.8 GB 单个 `model.safetensors`、`config.json`、`generation_config.json`、13 个从基座复制的文件与 `merge_receipt.json`；收据里 `adapter_config.json` sha256 `6b681e19cc2d93e3892e21bbaa01d2133d6e60d9223050b58e18a50babc38d5d`、`adapter_model.safetensors` sha256 `139b35909fcd8ef6efd9fe90768a9b47f49117b6e601e836ed4e0d950c34a3fd`。
- 拒绝重跑：`--output` 指向已存在的 `/test/Finagent/models/Qwen3.5-9B` → `refusing to overwrite /test/Finagent/models/Qwen3.5-9B: it must not exist`，exit=1，未加载模型（约 7 秒，基本是 import 耗时）。
- 对拍（CPU、BF16）：58 token 中文短输入，最后位置 logits——基座+适配器（未合并）与合并后模型最大绝对差 `0.125`、平均 `0.0153`，argmax 均为 `198`（一致）；logits 幅度最大 14.75。
- 权重层面核对合并算式：抽 4 个受 LoRA 的模块，合并后权重与 `W_base + (B@A)·(alpha/r)` 最大残差 `1.8e-5`；这 4 个模块 84,017,152 个元素中有 5,860,602 个（约 7%）与基座不同。
- `config.json` 与基座的差异：只在合并后出现的键 `dtype`、`text_config.bos_token_id=null`、`text_config.pad_token_id=null`、`text_config.partial_rotary_factor=0.25`、`text_config.tie_word_embeddings=false`、`vision_config.dtype`、`vision_config.rope_parameters.rope_theta/rope_type`；值变化 `transformers_version` 4.57.0.dev0→5.18.0、`vision_config.model_type` qwen3_5→qwen3_5_vision；没有只在基座里的键。
- 测试：`CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 120 tests … OK`（比之前多 1 项，即新增的 api_client 用例）。
- 已按任务删除 `/test/Finagent/models/qwen35-9b-extract-probe`，`/test/Finagent/models/` 只剩 `Qwen3.5-9B`。未提交 Git、未安装依赖、未启动 vLLM；`memory-bank/` 未改动。

未完成或有疑问：

1. `config.json` 的差异里除元数据外还有 `vision_config.model_type` 由 `qwen3_5` 变成 `qwen3_5_vision`：这是 transformers 5.18 载入时自己的规范化（基座 `config.json` 由 4.57.0.dev0 写出），不是权重或结构变化，训练入口今天也用 5.18 读同一基座。vLLM 0.25.1 是否接受这个值要等真正起服务才知道；若被拒，可改成“保存后把 `config.json` 换回基座原件”，请指示。
2. 试探检查点只训了 1 步，LoRA 的 `ΔW` 最大约 `1.7e-5`，低于 BF16 在 |W|≈0.1—0.5 上的分辨率，所以合并后约 93% 的权重与基座逐位相同，合并模型与未合并模型在 logits 上的差别与 bf16 噪声同量级（最大 0.125），argmax 相同；合并算式本身没错（残差 ≤1.8e-5 即 `ΔW` 本身）。含义：若正式训练 72 步后的 `ΔW` 仍小于 bf16 分辨率，合并保存为 BF16 会把 LoRA 效果量化掉。正式检查点出来后可先做一次同样的权重层面核对；如需保真可让输出用 FP32（约 35 GB 磁盘），T5 没有要求，我没有改。
3. `merge_receipt.json` 的 `adapter_sha256` 只覆盖 `adapter_*` 文件（`adapter_config.json`、`adapter_model.safetensors`），不含同目录的训练状态 `state.pt` 与 peft 生成的 `README.md`；若也想记录这两项的哈希请说一声。
4. `save_pretrained` 在 transformers 5.18 下把 18.8 GB 权重写成单个 `model.safetensors`（不是 5 GB 分片）。vLLM 能读单文件；若需要分片可加 `max_shard_size`，T5 未提，我没有加。

### T6（已验收）

改了哪些文件：

- 新建 `src/verify/pipeline.py`：`judge(checker, text, items, code, published) -> dict[int, dict]`。函数体就是原 `run_reviews.py` 主循环里 `for item in parse(answer['content'])` 那段（`locate` → 类型／年份校验 → `checkable` → 母公司 → ratio 与 amount／growth／change 分支 → 数字写法校验 → `checker.check` 与各 `unsupported` 原因），逐字搬移；只把三处 `review['code']`／`review['text']` 与闭包里的 `published` 换成同名参数 `code`／`text`／`published`，键与值结构不变。
- `src/evaluation/run_reviews.py`：删去 `import re` 与只服务该循环的导入（`SCALES`、`Claim`、`checkable`、`matches`），新增 `from src.verify.pipeline import judge`；循环替换为 `if not args.direct: verdicts = judge(checker, review['text'], parse(answer['content']), review['code'], published)`。`--direct` 分支、`--oracle`／`--workers` 与计分／写出部分未动，`git diff` 里该文件的其余差异都是阶段 4 已存在的未提交改动。
- 新建 `tests/test_pipeline.py`：内存 SQLite 小出处库（`document`／`fact` 造库方式同 `tests/test_checker.py`），8 个用例覆盖八个分支。

验证命令及输出摘要：

- `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest tests.test_pipeline -v` → `Ran 8 tests ... OK`：amount 一致 `consistent`；amount 不一致 `('inconsistent','digit')`；growth `consistent`；ratio `consistent`；母公司 `('unsupported','parent-company claim')`；季度措辞 `('unsupported','quarter, segment or own-method ratio')`；科目与措辞不符 `('unsupported','line item does not fit the wording')`；数字写法不规范 `('unsupported','number not written plainly')`。
- 行为不变验收（零费用，全部命中缓存）：
  - `mkdir -p /test/Finagent/runs/t6_check && cp artifacts/stage3_reviews_dev_v6/api_cache.jsonl /test/Finagent/runs/t6_check/`；
  - `CUDA_VISIBLE_DEVICES= .venv/bin/python -m src.evaluation.run_reviews --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --labels data/staged/eval_v2_labels/development_review_labels.jsonl --budget-usd 0.01 --output-dir /test/Finagent/runs/t6_check` → 正常结束，2223 个数字；`summary.json`：`wrong_passed_as_consistent` 23、`wrong_flagged` 623、`correct_confirmed` 918、`cost_usd` 0.2785。
  - `cmp /test/Finagent/runs/t6_check/records.jsonl artifacts/stage3_reviews_dev_v6/records.jsonl` 与 `cmp .../summary.json ...` 均无输出（完全相同）；sha256：records `33ca4e71fa05be6ce0a00a8f23552b6b91f5caae08341f57fab0fad5d8e95c27`、summary `422d373cae6b98a6023d785ce1fcce79d38bdd856d2b01223662c6ae4e865c04`，与原文件一致。
  - 运行后 `/test/Finagent/runs/t6_check/api_cache.jsonl` 与 `artifacts/stage3_reviews_dev_v6/api_cache.jsonl` 逐字节相同——没有新请求，也没有缓存未命中（未命中会因 `spent ≥ $0.01` 抛 `BudgetExceeded`）。
- `.venv/bin/python -m unittest discover -s tests -q` → `Ran 128 tests ... OK`（比 T5 的 120 项多 8 项，即新增的 pipeline 用例）。
- `.venv/bin/python -m compileall -q src` 无错误；`git diff --check` 无空白问题；`run_reviews.py` 内已无 `re`／`SCALES`／`Claim`／`checkable`／`matches` 的残留引用。

未完成或有疑问：

1. `memory-bank/architecture.md` 的“核查”一节仍只写 `run_reviews.py` 串起抽取与核查，没有提 `src/verify/pipeline.py`。T6 的文件清单只列新模块、`run_reviews.py` 与测试，按约定第 2 条我没有改文档；阶段 5 引用这个函数前建议补一行，需要时请下指令。
2. 本次全程未用 GPU（纯 CPU 读 SQLite ＋ 命中文件缓存），未触碰 `artifacts/stage4_9b_sft_dev_v1/` 与 vLLM 进程；未提交 Git、未安装依赖、未调用会产生费用的 API。

### T7（已验收）

改了哪些文件：

- `src/verify/pipeline.py`：只给经过 `checker.check` 的那个分支加 `'evidence': verdict.evidence`（事实编号列表），其余字段、其余分支与结构不变。
- 新建 `src/agent/llm.py`：`ApiChatModel(BaseChatModel)`，字段 `client: ApiClient` 与 `max_tokens: int = 8000`；`_generate` 按 `human／system／ai → user／system／assistant` 把 LangChain 消息转成 `ApiClient.chat` 的格式并返回 `AIMessage`（缓存、计价、预算、本地模型规则都沿用客户端）。没有第二个 HTTP 客户端。
- 新建 `src/agent/graph.py`：`ReviewState(TypedDict)`、`build_graph(checker, llm)`（extract → check → consistency → revise → report 线性五节点）、`review_draft(draft, *, library, model, url, budget_usd, cache) -> dict`、`workpaper_markdown(workpaper) -> str`，以及命令行 `python -m src.agent.graph --library --reviews --review-id --output [--model --url --budget-usd]`（模型应答缓存写在 `--output` 同目录的 `api_cache.jsonl`）。
  - extract：`extract.PROMPT` ＋ `line_items` → 模型 → `parse()`；check：`judge()`；consistency：按 `type／label／year／unit` 分组，同组内书写数值不同即互相标 `intra_document` 并记下双方位置（按 Decimal 比较，故 `1,234.56` 与 `1234.56` 视为同一个数）；revise：对每个 `inconsistent` 且有 `expected` 的数字，取它所在的句子（按 `。`／`；` 切分、不含分隔符，并去掉句首的段落换行）连同应有值交给模型，只有“该数字写成应有值、其余逐字相同”的改写句才接受（比较时去掉千分位），否则丢弃并记“需人工改写”；修订不写回任何结论；report：生成底稿（每个数字一条：位置、原文、结论、错因、应有值、由 `evidence` 事实编号查出库中的文件／页码／字符区间／行标签／列表头／单位、修改建议、稿内一致性发现；另加汇总计数）。
  - 底稿统计的数字用 `evaluation.draft_pilot.STATED` 枚举稿件全文的金额与百分比（与评测集 `numbers` 同一口径），没有结论的记为“抽取遗漏需人工”。
- 新建 `tests/test_agent_graph.py`：8 个用例，`FakeMessagesListChatModel` 返回固定 JSON ＋ 内存 SQLite 小出处库，不联网。
- `tests/test_pipeline.py`：加 2 个用例（经 checker 的断言带事实编号；未到 checker 的分支没有 `evidence` 键）。

验证命令及输出摘要：

- `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 138 tests ... OK`（T6 为 128 项，新增 10 项）；`tests.test_agent_graph`＋`tests.test_pipeline` 单独跑 18 项 OK。
- 真实跑一篇 `review-pro_closed-000858-1223311527`（五粮液 2024 年点评，`closed_book` 为真，稿件 25 个数字）：先把 `artifacts/stage3_reviews_dev_v6/api_cache.jsonl` 复制到 `/test/Finagent/runs/`，再
  `CUDA_VISIBLE_DEVICES= .venv/bin/python -m src.agent.graph --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --review-id review-pro_closed-000858-1223311527 --output /test/Finagent/runs/t7_demo.json --budget-usd 0.5`
  → `{"numbers": 25, "consistent": 4, "inconsistent": 20, "unsupported": 1, "ambiguous": 0, "missed": 0}`；输出 `/test/Finagent/runs/t7_demo.json` 与 `t7_demo.md`。
- 与 `artifacts/stage3_reviews_dev_v6/records.jsonl` 中同一稿件逐条比较（按 `start` 对齐）：25 条位置与原文全同，`verdict／error／expected／label／year／type／reason` 七个字段差异 0 —— 抽取命中缓存，结论完全一致。
- 同一篇里 20 次改写作答全部接受（首轮跑时 4 条合理改写被丢弃：句子的第一个字符是段落换行，模型不会照抄；已在切句后去掉段落换行并重跑这一篇）；稿内一致性发现 4 组（891.75／212.60、7.1／2.5、318.24／69.22、5.4／1.2，共 8 个数字被标，双方位置都记下）；证据示例：`2024年年度报告 第60页 [894,911) 其中:营业收入／2024年度／元`。
- 花费：为挑一篇能触发稿内一致性的稿件，先跑过 `review-flash_closed-601728-1222894586`（中国电信，37 个数字，结论与 records 差异同样为 0），再换到上面这篇；三次共新增 49 个请求，全部是改写作答（两篇的抽取都命中缓存，没有新抽取请求），**新增 0.0034 美元**，未超过 0.5 美元。
- `.venv/bin/python -m compileall -q src` 无错误；`git diff --check` 无空白问题。

未完成或有疑问：

1. 稿内一致性只对 `label` 非空的断言分组：ratio 类的 `label` 为 null（指标由分子／分母给出），若一并按 null 分组会把同一年份不同类型的比率算成同一个数。如果希望比率也参与稿内比对，需要先给分组定一个更细的键，请下指令。
2. 修订的验收是“除该数字外逐字相同”（比较时去掉千分位），模型若改动标点或增删文字仍会被丢弃并记“需人工改写”（单元测试覆盖这一分支；上面那 4 条的丢弃原因是段落换行，不是标点）。若希望更宽松（例如只比数字与文字、忽略标点），请下指令。
3. `memory-bank/architecture.md` 未改（T7 文件清单没有它，按约定第 2 条停手）：新增的 `src/agent/llm.py`、`src/agent/graph.py` 与 `python -m src.agent.graph` 入口属于模块边界与入口变化，阶段 5 文档更新时建议补一行。
4. `judge()` 的输出多了一个 `evidence` 键，而 `run_reviews.py` 把整份结论字典（`**mine`）写进 `records.jsonl`：再跑 T6 的行为不变验收时，`records.jsonl` 每条会多出 `evidence`（结论与 `summary.json` 都不变），与 `artifacts/stage3_reviews_dev_v6/` 的逐字节相同不再成立。本次没有重跑 `run_reviews`（避免改动 v6 产物），是否把那条验收改成“忽略 `evidence` 后相同”或重生成产物，请下指令。
5. 本次全程未用 GPU、未触碰 vLLM 进程与 `artifacts/stage4_9b_sft_dev_v1/`；未提交 Git、未安装依赖；除 `deepseek-flash` 的 49 次改写作答（共 0.0034 美元）外没有其他付费调用。

## 验收记录

| 日期 | 任务 | 结论 | 说明 |
|---|---|---|---|
| 2026-10-14 | 阶段 2 点评数字逐条核实（`src/evaluation/review_labels.py`、`data/staged/eval_v2_labels/`） | 已验收 | 12,309 个数字完成首遍与 10% 盲审，一致率 96.9%—99.1%；标签作为阶段 3、4 金标使用。模块 391 行，仅被其测试引用，属离线工具，保留。 |
| 2026-10-14 | T1 训练入口解耦 | 已验收 | 复核 diff：`encode_messages` 函数体与原版逐字相同；`prepare_samples` 只剩 chat 样本装载，样本校验、最长样本置首、按种子打乱与原 `prepare_chat_samples` 一致；删去的 DPO 条件表达式在 SFT 下等价。导入 `m0_lora_check` 只加载 `src.training.*`。停在 T2 并上报两处冲突的做法正确。 |
| 2026-10-14 | T2 删除旧路线模块 | 已验收（Claude 补正一处） | 复核：60 个文件按清单删除，src 只剩新路线模块与 calculator／evidence.models；`terms` 与 `base_model.yaml` 按决定处理；我重跑 `unittest discover` 119 项通过，grep 无残留。问题：`environment.md` 资产表删掉了 `data/raw/chinese_expanded/`，但康美、新纶更正案例原件在其中，重建出处库需要——已由 Claude 补回，并把仍在磁盘上的旧路线资产单列为“无使用方、删除须批准”。以后删文档条目前先 grep 保留配置是否引用。 |
| 2026-10-14 | T3 删除旧路线资产与无用依赖 | 已验收 | 首次由 Claude 用 `codex exec -p deepseek` 驱动。复核：6 个路径已删（释放约 4.3 GB），康美、新纶原件仍在；`requirements-research.txt` 只剩 `mcp`；grep 无残留；我重跑 unittest 119 项通过。空的 `models/` 由 Claude 删除。 |
| 2026-10-14 | T4 损失分块与 4 卡试跑 | 已验收 | 复核 diff 与日志：只对受监督位置分块投影词表、每块 checkpoint；小模型对拍损失与 LoRA 梯度 rtol 1e-5 内一致，真实 9B 新旧损失完全相同；4 卡试跑 1 步 38.4 s、峰值显存 14.3 GiB。Codex 发现各 rank 分块数不同会使 FSDP 分片的 `lm_head` 集合通信错位并死锁，加 `agreed_chunk_count` 取各 rank 最大块数、以空块补齐——判断正确，采纳。`flash-linear-attention` 不装：72 步约 46 分钟，可接受。 |
| 2026-10-14 | T5 合并适配器工具 | 已验收 | 复核：合并在 CPU 完成，权重层面 `W+BA·α/r` 残差 ≤1.8e-5，短输入 argmax 一致；`LOCAL` 新增 `qwen3.5-9b-extract`，120 项测试通过。Codex 提出的 BF16 合并可能抹掉小幅 LoRA 更新的风险成立：正式检查点合并后先在开发集抽样对比未合并与合并的抽取输出，不一致则改用 vLLM 原生 LoRA。 |
| 2026-10-15 | T6 抽出 `judge()` | 已验收 | 复核：开发集 flash 缓存重跑 `records.jsonl`、`summary.json` 与原产物逐字节相同，零新请求；新增 8 项分支测试，共 128 项通过。architecture.md 由 Claude 在阶段 5 文档更新时补。 |
| 2026-10-15 | T7 LangGraph 流程与底稿 | 已验收（附后续任务） | 复核代码：结论只来自核查器，修订 Agent 的改写由代码核对；五粮液样例 25 个数字与阶段 3 结论逐条一致，花费 0.0034 美元，139 项测试通过。四个疑问的决定：`evidence` 键保留（新产物带证据，旧产物不重算）；修订严格逐字核对保留；architecture.md 由 Claude 更新。发现问题：稿内一致性把“四季度收入／归母净利润”与全年同名科目归为一组，产生错误的稿内冲突——转 T8 修正并在开发集上测精度。 |
