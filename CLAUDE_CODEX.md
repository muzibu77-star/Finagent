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

### T8 稿内一致性精度修正与开发集全量运行（已验收）

**背景**：T7 样例（五粮液 2024）中，“其中四季度收入 212.60 亿元，同比增长 2.5%，归母净利润 69.22 亿元”被抽取为全年同名科目，`intra_document()` 因此把它们与全年数字报成稿内冲突，这是误报。本任务允许调用 `deepseek-flash`，总花费 ≤ 1 美元；GPU 不需要。

- **做法**（只改 `src/agent/graph.py` 与 `tests/test_agent_graph.py`，并新增一个评测脚本）：
  1. `intra_document()` 分组前排除：数字所在句（`sentence_at`）在该数字之前的部分含 `src/verify/extract.py` 的 `NARROWER` 词（季度、上半年、分部等）的 item；`checkable(phrase)` 为假的 item；`type` 不在 amount/growth/change 的 item。分组键再加上 `norm(phrase)`（`src/verify/checker.py` 的 `norm`），只有措辞指向同一指标的数字才互相比较。
  2. 新增 `src/evaluation/run_graph.py`：对一个稿件集（如 `data/staged/eval_v2/development_reviews.jsonl`）逐篇运行 `review_draft()`，`as_of` 取法同 `run_reviews.py`；把每篇底稿写成 `<output>/workpapers/<review_id>.json`，并汇总：
     - 结论计数，并按 `data/staged/eval_v2_labels/development_review_labels.jsonl` 计分，口径与 `run_reviews.py` 的 summary 相同；
     - 稿内一致性发现数；
     - 修订建议数、被代码接受数、丢弃数；接受的建议中应有值确实出现在改写句里的比例（应为 100%，作为自检）。
     缓存：先把 `artifacts/stage3_reviews_dev_v6/api_cache.jsonl` 复制到输出目录作为初始缓存，使抽取全部命中（请求体与 `run_reviews` 相同时哈希一致）；只有修订调用产生新费用。
  3. 运行：`.venv/bin/python -m src.evaluation.run_graph --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --labels data/staged/eval_v2_labels/development_review_labels.jsonl --budget-usd 1 --output-dir artifacts/stage5_graph_dev_v1`。
- **验收**：
  - 计分结果必须与用当前核查器重算的 flash 结果一致：`wrong_flagged 618`、`wrong_passed_as_consistent 23`、`correct_confirmed 917`、`correct_flagged 10`（若不同，说明抽取未命中缓存或流程有差异，停下回报）；
  - **人工核对全部稿内一致性发现**（数量不多时全部，超过 60 条时固定种子抽 60 条）：逐条看原句，判定两个数字是否确实指同一指标、同一期间、同一口径。回报真冲突数／误报数及误报原因；把核对表写到 `artifacts/stage5_graph_dev_v1/consistency_audit.jsonl`（每条：review_id、两个位置与原句片段、判定、理由）；
  - 抽 30 条被接受的修订建议（固定种子），核对改写句只改了该数字；
  - 单元测试覆盖“季度句不参与分组”“措辞不同不分组”；`unittest discover` 通过；回报花费。
- **禁止**：不改核查器与抽取提示词；不改 `judge()`。

### T9 单 Agent 对照组：一个带工具的 Agent 直接审稿（已验收）

**背景**：plan.md 要求“多智能体是否优于单 Agent 须在同一评测集上对照后决定”。现有流程（抽取 Agent → 核查器 → 修订 Agent）在开发集上的结果见 `artifacts/stage5_graph_dev_v1/summary.json`。对照组是一个 ReAct Agent：拿到稿件和查资料的工具，自己检索、计算并逐个数字给结论。本任务允许调用 `deepseek-flash`，总花费 ≤ 8 美元；GPU 正在跑验收评测，**不要使用 GPU、不要碰 vLLM 进程和 `artifacts/stage6_*`**。

- **做法**：
  1. `src/model/api_client.py`：`chat()` 增加可选参数 `tools: list | None = None`，非空时放进请求体（OpenAI 格式）；缓存行与返回值增加 `tool_calls`（原样保存）。不传 `tools` 时请求体与哈希必须与现在完全相同（已有缓存继续命中），补测试。
  2. `src/agent/llm.py`：`ApiChatModel` 支持 `bind_tools()`（转成 OpenAI tools 格式）以及消息中的 `tool_calls`／`ToolMessage` 往返转换。
  3. 新建 `src/agent/single.py`：`langgraph.prebuilt.create_react_agent`，工具为：
     - `find_number(text)`：`Library.find_number`，限定该发行人与截止日；
     - `search_pages(query)`：`Library.search_pages`，同上限定；
     - `page_text(document_id, page)`：只允许截止日前已披露的文件；
     - `calculate(expression)`：只允许数字、`+ - * / ( )` 的算式，用 Decimal 计算（可复用 `src/tools/calculator.py` 中合适的部分；不能复用就写最小的安全求值，禁止 `eval`）。
     系统提示：说明任务（逐个检查稿件中以亿元、万元、百万元、% 表示的数字是否与年报一致或能由年报数据算出），要求最后只输出与 `run_reviews.DIRECT` 相同格式的 JSON 数组（number、occurrence、verdict 为 正确／错误／无法判断）。设最大步数（如 40 次工具调用），超出就按已有输出计分、未给结论的数字记为 missed。
  4. 新建 `src/evaluation/run_single.py`：在开发集上逐篇运行，计分口径与 `run_reviews.py` 完全相同（`verdict` 映射同 `--direct` 分支），输出 `artifacts/stage5_single_dev_v1/`（records、summary、每篇的工具调用轨迹 `traces/<review_id>.json`），并在 summary 里加：平均工具调用次数、平均输入／输出 token、花费。
- **验收**：
  - 先跑 3 篇试点，检查轨迹合理（确实调用了工具、最终 JSON 能解析），回报一篇的轨迹摘要；再跑全部 66 篇；
  - 回报与 `artifacts/stage5_graph_dev_v1/summary.json` 的对照表：错误数字判出、非舍入放行（`truth=wrong` 且判一致、`reason` 不是 `rounding difference`；单 Agent 没有 reason，故全部判一致的错误数字都算非舍入放行，另单独列出其中与应有值相差在末位一个单位以内的个数）、正确确认、误报、遗漏、平均花费／篇；
  - `unittest discover` 通过（工具与 LLM 往返用假模型测试，不联网）。
- **禁止**：不改核查器、抽取提示词、`judge()` 与 `graph.py` 的节点逻辑。

### T11 混合方案：Agent 复查交给人工的数字，只能加报警（已验收）

**背景**：见 `memory-bank/result.md`“阶段 5：多 Agent 流程与单 Agent 对照”。规则已定：工具 Agent 只复查核查器交给人工的数字（结论为 `unsupported`、`ambiguous` 或抽取遗漏 `missed`）；Agent 判“错误”→ 结论改为 `inconsistent`，来源记为 `agent`；Agent 判“正确”→ 结论不变（仍交给人），附提示 `agent_hint: consistent`；Agent 永远不能产生 `consistent`。本任务允许调用 `deepseek-flash`，总花费 ≤ 6 美元；不使用 GPU。

- **做法**：
  1. `src/agent/graph.py` 在 `check` 与 `consistency` 之间加 `investigate` 节点（可由 `review_draft(..., investigate=True)` 打开，默认打开；关闭时图与现在完全相同）。节点收集交给人工的数字（用底稿同样的 `numbers_in()` 枚举，按起始位置），没有则跳过；有则调用 `src/agent/single.py` 的 Agent，提示里只列出这些数字（原文写法与第几次出现），要求输出 JSON 数组：`number`、`occurrence`、`verdict`（正确／错误／无法判断）、`expected`（判错误时的应有值，可空）、`basis`（依据：文件、页码与原文片段，可空）。答案用 `locate()` 回到位置，只接受落在“交给人工”集合里的位置，其余忽略。Agent 失败或答案无法解析：不改任何结论，`errors` 记录原因。
  2. 底稿每条加 `source`（`checker`／`agent`／空）、`agent`（Agent 的结论、应有值、依据）；汇总加 `agent_flagged`、`agent_hint_consistent`。`revise` 不为 Agent 标出的数字生成改写（应有值未经核查器核实），底稿显示“Agent 认为有误：应为 X（依据……），需人工确认”。Markdown 版同步。
  3. `src/evaluation/run_graph.py` 增加 `--no-investigate`；计分时 `inconsistent` 不区分来源，另外单列 Agent 来源的 TP／FP（按真值 wrong／correct／unverifiable 拆分）。
  4. 测试（假模型）：Agent 判错误→不一致且来源 agent；判正确→仍为交给人工且有提示；Agent 对核查器已判一致／不一致的数字给出的结论被忽略；Agent 失败不改结论。
- **运行**：开发集 `run_graph`（输出 `artifacts/stage5_hybrid_dev_v1/`，先复制 `artifacts/stage5_graph_dev_v1/api_cache.jsonl` 作初始缓存）。
- **验收**：回报与 `artifacts/stage5_graph_dev_v1/summary.json` 的对照（判出、放行与其中非舍入、确认、误报、遗漏、`agent_flagged` 的真值拆分、每篇新增花费）；非舍入放行必须仍为 0、`consistent` 计数必须与关闭 investigate 时完全相同（Agent 不能放行的直接检验）；`unittest discover` 通过。
- **禁止**：不改核查器、抽取提示词、`judge()`；不改 `single.py` 的工具权限。

### T10 审核页面、MCP 工具与故障处理（已验收）

**背景**：阶段 5 剩余的产品化部分。流程已含 T11 的 investigate 节点（默认打开）；GPU 不需要。另一个验收评测正在用 flash 跑（`artifacts/stage6_accept_hybrid/`），不要碰。核查流程入口是 `src/agent/graph.py` 的 `review_draft()`。本任务允许调用 `deepseek-flash`，总花费 ≤ 0.5 美元；GPU 可能在跑验收评测，**不要使用 GPU，不要碰 vLLM 进程和 `artifacts/stage6_*`**。依赖只用已安装的（标准库 `http.server`、`mcp==1.26.0`），不装新包。

- **故障处理**（改 `src/agent/graph.py`，结论仍只来自核查器）：
  1. 抽取节点调用模型失败（网络、HTTP 错误、`BudgetExceeded`）或答案无法解析：不抛出，所有数字在底稿中记为“抽取遗漏需人工”，底稿新增 `errors` 列表写明节点与原因；
  2. 修订节点单条调用失败：该条建议记“需人工改写”，错误写入 `errors`，其他条照常；
  3. 不变量：稿件中每个金额／百分比在底稿里恰好出现一次；模型失败不会产生“一致”或“不一致”结论。
  4. 单元测试逐条覆盖以上情况（假模型抛异常、返回乱码、预算用尽）。
- **审核页面**：`src/service/app.py`（`ThreadingHTTPServer`，只监听 127.0.0.1，默认端口 18100）+ `src/service/index.html`（原生 HTML/JS，单文件，无外部 CDN）：
  - 表单：发行人（下拉，来自出处库中有年报的发行人）、财年、写稿截止日、稿件正文、模型（`deepseek-flash` 或本机 `qwen3.5-9b-extract`，后者 URL 由命令行参数给出）；
  - `POST /api/review` 运行 `review_draft()`，同一时间只跑一篇（锁），返回底稿 JSON；参数不合法（发行人或财年不在库中、正文为空、截止日格式错）返回 400 与原因；
  - 页面展示：汇总计数（含 Agent 标出数与“Agent 认为一致”的提示数）；逐数字表格（原文、结论、来源〔核查器／Agent〕、错因、应有值、出处“文件 第 N 页”、修改建议或 Agent 依据），稿件正文中按结论给数字着色、点击跳到对应行；`errors` 非空时醒目提示；可下载底稿 JSON 与 Markdown（`GET /api/workpapers/<id>.json|.md`，底稿存到 `--data-dir`）。
  - 命令行：`python -m src.service.app --library ... --data-dir ... [--local-url http://127.0.0.1:18090/v1/chat/completions] --budget-usd ...`。
- **MCP**：`src/service/mcp_server.py`，stdio，`mcp` 包的 FastMCP；工具：`check_draft(code, fiscal_year, as_of, text)`（返回底稿）、`find_number(text, code, as_of)`、`search_pages(query, code, as_of, limit)`、`page_text(document_id, page, as_of)`（截止日后披露的文件拒绝）。模型与预算由启动参数决定。
- **验收**：
  - 单元测试：服务端点（用假模型，起在随机端口）——正常、400 三种、下载；MCP——用 `mcp` 客户端经 stdio 启动服务，列出四个工具并调用 `find_number`、`page_text`（含截止日拒绝）；故障三种；
  - 真实冒烟：用 flash 起服务，提交开发集稿件 `review-pro_closed-000858-1223311527`（五粮液 2024，截止日取年报披露日），返回的 `summary` 与 `artifacts/stage5_graph_dev_v1/workpapers/` 中该篇一致；把返回 JSON 与 Markdown 存到 `/test/Finagent/runs/t10_smoke.*`；
  - `unittest discover` 通过；`architecture.md` 不改（由 Claude 更新）。
- **禁止**：不改核查器、抽取提示词与 `judge()`；不新增配置文件。


### T12 移除稿内一致性节点（已验收）

**背景**：`memory-bank/result.md` 记录，`consistency` 节点在真实稿件上的发现 0／19 正确（开发集修正前 7 组、验收集 12 组全为误报）。决定从流程中移除，底稿不再给出未经核实的“稿内冲突”。不调用付费 API，不用 GPU。

- **做法**：删除 `src/agent/graph.py` 的 `consistency` 节点、`intra_document()` 及只为它服务的辅助函数与常量；状态、底稿行与汇总中去掉 `consistency` 相关字段；图变为 extract → check → investigate → revise → report。同步删掉 `src/evaluation/run_graph.py` 汇总里的 `consistency_findings`、页面 `src/service/index.html` 与 Markdown 底稿里的稿内一致性列、相关测试。
- **验收**：`grep -rn "intra_document\|consistency" src tests` 只剩与本节点无关的结果（逐条列出说明）；用开发集缓存零花费重跑 `run_graph`（输出到 `/test/Finagent/runs/t12_check/`，先复制 `artifacts/stage5_hybrid_dev_v1/api_cache.jsonl`），计分与 `artifacts/stage5_hybrid_dev_v1/summary.json` 的判出、放行、确认、误报、Agent 计数完全相同，且无新请求；`unittest discover` 通过。

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

### T8（已验收）

改了哪些文件：

- `src/agent/graph.py`：`intra_document()` 分组前排除 ①`type` 不在 amount／growth／change 的 item、②`checkable(phrase)` 为假的 item、③数字所在句在该数字之前的部分含 `extract.NARROWER`（季度、上半年、分部等）的 item；分组键由 `(type, label, year, unit)` 改为再加 `norm(phrase)`。为取“该数字之前的部分”，把 `sentence_at` 里的区间计算拆成 `sentence_span`，`sentence_at` 行为不变（其余节点、`judge()`、`checker.py`、抽取提示词都没动）。
- `tests/test_agent_graph.py`：新增 `test_a_part_year_sentence_takes_no_part_in_grouping`（季度句不参与分组）与 `test_a_different_wording_is_not_grouped`（措辞不同不分组），另加 `run_text` 辅助与两段稿件、两组 items 的固定装置。两个用例都先用旧规则复算过：旧规则下确实会报出误报，因此它们锁的是本次修正。
- 新增 `src/evaluation/run_graph.py`：逐篇调用 `review_draft()`，底稿写 `<output-dir>/workpapers/<review_id>.json`；计分口径照 `run_reviews.py` 的 summary（按 `review['numbers']` × 核实标签、同一 `by_truth` 结构），另汇总稿内一致性发现数、修订建议／接受／丢弃数、“接受的建议中应有值出现在改写句里”的自检计数、`cost_usd`（缓存内累计）与 `cost_usd_new`（本次新增）。缓存按任务说明在运行前手工复制，脚本本身不新建配置项。
- 产物（`artifacts/` 不进 Git）：`artifacts/stage5_graph_dev_v1/` 下 66 份底稿、`summary.json`、`api_cache.jsonl`、`consistency_audit.jsonl`。

验证命令及输出摘要：

- `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 140 tests ... OK`（T7 为 138，本次 +2）。
- 抽取全命中缓存：先按 `ApiClient` 的请求体构造逐篇计算哈希，66 篇的抽取请求体与 `artifacts/stage3_reviews_dev_v6/api_cache.jsonl` 一致；全量运行只新增改写作答请求。
- `cp artifacts/stage3_reviews_dev_v6/api_cache.jsonl artifacts/stage5_graph_dev_v1/` 后
  `CUDA_VISIBLE_DEVICES= .venv/bin/python -m src.evaluation.run_graph --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --labels data/staged/eval_v2_labels/development_review_labels.jsonl --budget-usd 1 --output-dir artifacts/stage5_graph_dev_v1`
  → `summary.json`：`wrong_flagged 618`、`wrong_passed_as_consistent 23`、`correct_confirmed 917`、`correct_flagged 10`，四项与验收数字逐项一致；`numbers 2223`、`by_truth` 与 `/test/Finagent/runs/rescore/stage3_reviews_dev_v6/summary.json` 完全相同。
- 逐条对拍（单篇 `review-flash_closed-000858-1219873658`，31 个位置）：`verdict／error／expected／label／year／type` 与 rescore 的 `records.jsonl` 全部相同，只有 1 条 `reason` 在旧产物里缺键（`''` vs 缺键），无实质差异。
- 稿内一致性（修正后）：`summary.json` 的 `consistency_findings 0`，66 份底稿里每条数字的 `consistency` 均无发现。按旧规则在同一份缓存上复算，开发集共 7 组（五粮液 2024 四组、恒瑞医药 2024 一组、恒瑞医药 2025 两组），逐条看原句核对：**真冲突 0、误报 7**。误报原因两类：①其中一个数字在季度／单季句里（“其中四季度收入212.60亿元”“单四季度实现收入85.12亿元”），4 组由句首 `NARROWER` 排除、2 组由 `checkable(phrase)` 排除；②“同比增长22.6%”与“收入增速重回20%以上”是同一增速的两种措辞，由 `norm(phrase)` 排除。核对表 `artifacts/stage5_graph_dev_v1/consistency_audit.jsonl` 写的这 7 组（每条：review_id、两个位置与原句片段、判定、理由、被哪条规则排除），因为修正后没有发现了。
- 修订抽样：本次 645 次改写作答、628 条建议（见疑问 1）、接受 627、丢弃 1（`review-flash_closed-601728-1222894586` 的“销售费用为586.34亿元，同比增长4.1%”，模型未给出合规改写）；自检 `expected_in_rewrite 627/627`。固定种子 0 抽 30 条被接受的建议逐条核对：非数字文字完全相同、只改了一个数字、且改成的正是应有值 → 30/30 通过（脚本与明细在 `/test/Finagent/runs/t8/`）。
- 花费：只调用 `deepseek-flash`。输出目录缓存累计 0.3195 美元（其中 0.2785 是阶段 3 已付的抽取缓存，本次未重复计费以外的调用）；本次新增 0.0410 美元，在 1 美元预算内（含首次后台运行被会话回收前已作答的 19 次；这些应答已被全量运行复用）。

未完成或有疑问：

1. 修订节点按抽取 items 逐条调用，而底稿与计分按 `numbers_in`／STATED 枚举的数字：开发集 645 次改写作答对应 628 条被记入底稿的建议，差的 17 次是①抽取把带负号的数字写成不带负号的（如原文 “-450.75亿元” 抽成 “450.75”，`locate` 落在数字上而不是负号上，偏移与 STATED 差一个字符——开发集 94 个负数里 16 个如此）；②两个 item 落到同一位置（如 “4” 与 “4.1”）。这类位置在底稿里显示为“抽取遗漏需人工”，而模型其实已作答，多花约 2.6% 的改写作答费用；阶段 3 的计分是同一口径，所以验收数字不受影响。改 `revise` 的去重或让 `locate` 吃下负号都超出 T8 写明的做法，未改动。
2. `consistency_audit.jsonl` 记的是修正前的 7 组（每行带 `rule` 字段说明）；修正后开发集 0 条发现，按验收“人工核对全部稿内一致性发现”无处可核。若希望这个文件在修正后为空、或改成“修正后为空＋修正前 7 组另存”，请下指令。
3. 本次改动只会让稿内一致性发现变少（只排除、只细分，不新增分组），所以开发集 7 组降为 0 是预期方向；但这也说明修正后该检查在开发集上不再产出任何发现，“能查出真冲突”这一点无法用开发集验证，需要阶段 6 验收集或专门的对照样例。
4. 工作树里 `src/model/api_client.py`（重试多捕获 `HTTPException`）与 `memory-bank/result.md` 出现了不是本次任务产生的未提交改动——我开工时 `git status` 只有 `CLAUDE_CODEX.md`，这两处我没有触碰，请不要记在 T8 账上。
5. 本次全程未用 GPU、未触碰 vLLM 进程与 `artifacts/stage4_9b_sft_dev_v1/`；未提交 Git、未安装依赖；未改 `checker.py`、`extract.py` 的提示词与 `judge()`。

### T9（已验收）

改了哪些文件：

- `src/model/api_client.py`：`chat()` 增加可选 `tools`，非空时按 OpenAI 格式放进请求体；省略或传空时请求体与哈希与改动前逐字节相同（66 篇开发集抽取请求复算后仍全部命中 `artifacts/stage3_reviews_dev_v6/api_cache.jsonl`）。缓存行与返回值增加 `tool_calls`（原样保存，可能是 `null`；改动前写下的缓存行没有这个键，读取方一律用 `.get`）。
- `src/agent/llm.py`：`ApiChatModel.bind_tools()` 用 `convert_to_openai_tool` 转成 OpenAI 格式再 `bind`；`_generate()` 把 `ToolMessage` 转成 `role=tool`（带 `tool_call_id`）、把 `AIMessage.tool_calls` 转成请求里的 `tool_calls`，并把应答的 `tool_calls` 解成 `{'name','args','id'}`（`arguments` 不是合法 JSON 时给空参数，模型随即收到工具的参数报错）。另加 `used` 字段累计本模型调用的输入／输出 token，供汇总用。
- 新增 `src/agent/single.py`：`langgraph.prebuilt.create_react_agent` ＋ 4 个工具（`find_number`、`search_pages`、`page_text`、`calculate`），全部限定该发行人与写稿截止日；`page_text` 先查该文件的披露日，晚于截止日直接拒绝；`calculate` 用 `ast` 只接受数字与 `+ - * / ( )`，全程 Decimal，没有 `eval`。入口 `review_draft_single()` 返回逐数字结论、工具调用次数、token 用量与消息轨迹；步数上限用 `recursion_limit = 2 × 40 + 2`（一个模型回合可并行多个工具调用）。工具在线程池里执行，所以每次调用自己开只读连接。
- 新增 `src/evaluation/run_single.py`：逐篇跑开发集，计分与 `run_reviews.py --direct` 分支完全相同（`parse` ＋ `locate` ＋ verdict 映射，没给结论记 missed），写 `records.jsonl` 与 `traces/<review_id>.json`，summary 增加平均工具调用次数、平均输入／输出 token、花费；`--limit` 跑试点、`--workers` 并发、缓存跨次复用。
- 测试：`tests/test_single_agent.py` 新增 6 项（消息与 tool_calls 的往返、`bind_tools` 的请求体、工具的发行人与截止日范围、Decimal 计算与非法算式、假模型跑通整条 ReAct 回路并核对轨迹与工具调用请求体）；`tests/test_api_client.py` 新增 3 项（tools 进请求体、不带 tools 的哈希不变、`tool_calls` 落缓存与旧缓存行兼容）。

验证命令及输出摘要：

- `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 151 tests ... OK`（T8 为 140，本次 +11）。全程不联网。
- 缓存回归：按 `ApiClient` 的请求体逐篇复算 66 篇抽取请求哈希 → 66/66 仍命中阶段 3 缓存。
- 试点 3 篇（`--limit 3 --workers 3`）：每篇 45—57 次工具调用，最终 JSON 都能解析。轨迹摘要（`review-flash_closed-000858-1223311527`，26 个数字）：62 条消息（1 human ＋ 16 ai ＋ 45 tool）；工具分布 `search_pages` 9、`page_text` 16、`find_number` 6、`calculate` 14；第一回合并行发出 3 个 `search_pages`（营业收入／归母净利润／销售费用），中间用 `find_number` 复核单个数字、用 `calculate` 复算同比，最后一条 ai 消息是完整的 verdict JSON 数组。
- 全量 66 篇（`--workers 4`，约 10 分钟）→ `artifacts/stage5_single_dev_v1/`（`records.jsonl`、`summary.json`、66 份 `traces/`）。

对照表（同一开发集与标签，graph 侧取 `artifacts/stage5_graph_dev_v1/summary.json`）：

| 2,223 个数字（错误 959／正确 1,206） | 单 Agent（flash＋工具） | 多 Agent 流程 |
|---|---|---|
| 错误数字判出（判不一致） | 823（85.8%） | 618（64.5%） |
| 错误放行（判一致） | 32，其中 18 个与应有值相差在末位一个单位以内 | 23，reason 全部是 `rounding difference`（按同一末位口径 19 个） |
| 正确确认 | 1,158（96.0%） | 917（76.0%） |
| 正确误报 | 1（0.08%） | 10（0.83%） |
| 遗漏（完全没给结论） | 138（错误 91／正确 47） | 18（错误 13／正确 5） |
| 交人工（无依据／无法判断） | 13（均为错误数字） | 593（错误 305／正确 274／无法核实 58） |
| 平均花费／篇 | $0.0440（缓存累计 $0.0463） | $0.0006（缓存累计 $0.0048，含阶段 3 抽取缓存） |

- 按稿件类型拆开更说明问题：不给原文写的稿（closed_book，错误 883 个）单 Agent 判不一致 774、放行 5，graph 是 570／5；给原文写的稿（错误 76 个，都是细小错误）单 Agent 判不一致 49、放行 27，graph 是 48／18。单 Agent 的优势几乎全部来自粗错很多的稿件；在最关心的“给原文写成的细小错误”上两者判出数持平（49 对 48），单 Agent 放行还多 9 个。
- 平均每篇 51.03 次工具调用、输入 134,889 token、输出 4,836 token。全程只用 `deepseek-flash`，缓存累计 3.0539 美元（本次新增 2.9025），在 8 美元内；未用 GPU，未碰 vLLM 进程与 `artifacts/stage6_*`；未提交 Git、未安装依赖。

未完成或有疑问：

1. 66 篇里有 3 篇最终答复不是可解析的 JSON（`review-flash_closed-600276-1222961962`、`review-pro_closed-600276-1222961962`、`review-pro_closed-000858-1223311527`）：两篇在同一段文字里写了多个 JSON 数组，`extract.parse` 的贪婪 `\[.*\]` 因此报 Extra data；一篇模型以散文收尾（最后一条 ai 消息 111 字、没有数组）。这 3 篇占 138 个遗漏中的 89 个。`parse` 是 DIRECT 对照臂与本次共用的解析器，我按“计分口径与 `run_reviews.py` 完全相同”没有动它；是否给单 Agent 加一次“输出不合格式就重发”的重试（会与 DIRECT 的可比性略有出入），请你定。
2. `create_react_agent` 在 langgraph 1.2.15 会打一条弃用告警（建议改用 `langchain.agents.create_agent`）。T9 指定用 `langgraph.prebuilt.create_react_agent`，我照做；是否换 import 由你定。
3. `page_text` 需要判断“披露日不晚于截止日”，`Library` 没有这个查询，我在 `single.py` 里直接读 `Library.db` 的 `documents.published`，没有改 `library.py`。
4. 我只改了 T9 列出的文件与测试；`memory-bank/architecture.md` 的“编排与底稿”一节还没有这两个新入口，按“只改任务列出的文件”的约定没有自行扩大范围，是否补由你决定（T7、T8 也是你更新的文档）。

### T11（已验收）

改了哪些文件：

- `src/agent/single.py`：把 ReAct 运行抽成 `run_agent()`／`final_answer()`（`review_draft_single` 行为不变，四个工具与它们的权限一字未改）；新增 `INVESTIGATE_SYSTEM`／`INVESTIGATE_TASK` 与 `investigate_draft(draft, numbers, *, library, model, url, budget_usd, cache)`——提示里只列传入的数字（原文写法＋第几次出现），返回 `items`（number／occurrence／verdict／expected／basis）、`tool_calls`、`usage`、`messages`；`正确／错误／无法判断` 到 `consistent／inconsistent／unsupported` 的映射提为模块级 `MEANING`。
- `src/agent/graph.py`：`investigate` 节点插在 check 与 consistency 之间，由 `build_graph(..., investigator=<callable>)` 打开（`investigator` 为 `None` 时不建节点、边的顺序与原来完全相同），`review_draft(..., investigate=True)` 默认打开、关闭时走五节点原路。节点用 `handover()` 收集结论为 missed／unsupported／ambiguous 的数字（`numbers_in()` 枚举，`occurrence_at()` 用 `locate()` 反推第几次出现），没有则跳过；Agent 答案用 `locate()` 回位置，落在该集合之外或已占用的位置一律忽略；Agent 抛异常或答案解析不出任何条目 → 结论全不变、`errors` 记原因。只有“错误”把结论改成 `inconsistent`（`source='agent'`、`expected` 取自 Agent），只有“正确”记 `agent` 提示（结论仍交给人工）。底稿每条加 `source`（`checker`／`agent`／空）、`agent`（结论、应有值、依据），顶层加 `errors`，汇总加 `agent_flagged`、`agent_hint_consistent`；`revise` 跳过 `source='agent'` 的数字（这些数字 `suggestion` 为空），Markdown 的“修改建议”列改显示 `Agent 认为有误：应为 X（依据……），需人工确认` 或 `Agent 认为一致，仍需人工确认`。
- `src/evaluation/run_graph.py`：新增 `--no-investigate`（透传给 `review_draft`）；汇总新增 `wrong_passed_nonrounding`、`agent_flagged` 与 `agent_flagged_by_truth`（wrong／correct／unverifiable）、`agent_hint_consistent` 与 `agent_hint_by_truth`、`agent_errors`；`inconsistent` 的计分不区分来源。
- `tests/test_agent_graph.py`：更新 summary 断言，新增 `InvestigateTest` 6 项（handover 只交人工集合且 occurrence 正确；判错误 → 不一致、`source='agent'`、`revise` 不给它生成改写；判正确 → 结论仍为交给人工并带提示；对核查器已判一致／不一致的数字给出的结论被忽略；Agent 抛异常、答案不可解析都不改结论并记 `errors`）。

验证命令及输出摘要：

1. `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 157 tests ... OK`（T9 为 151，本次 +6）。全程 CPU、不联网、不装依赖。
2. 关闭 investigate 的零费用对照：`cp artifacts/stage5_graph_dev_v1/api_cache.jsonl /test/Finagent/runs/t11_noinvest/` 后
   `.venv/bin/python -m src.evaluation.run_graph --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --labels data/staged/eval_v2_labels/development_review_labels.jsonl --budget-usd 6 --no-investigate --output-dir /test/Finagent/runs/t11_noinvest`
   → 66 篇跑完，`by_truth`、`revisions 628/627`、`consistency_findings 0`、`correct_confirmed 917` 与 `artifacts/stage5_graph_dev_v1/summary.json` 逐项相同，`cost_usd_new 0.0`，缓存仍 711 行（全命中）。
3. 先导 1 篇（真实 `deepseek-flash`）：`.venv/bin/python -m src.agent.graph --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --review-id review-flash_closed-601728-1219408348 --output /test/Finagent/runs/t11_pilot/review-flash_closed-601728-1219408348.json --budget-usd 6` → 45 个数字：一致 6／不一致 27／无依据 11／存疑 1，`agent_flagged 14`、`agent_hint_consistent 4`、`errors []`；底稿 `.md` 的“修改建议”列出现 `Agent 认为有误：应为 4649.65亿元（依据：中国电信2023年年度报告 第172页 …），需人工确认` 与 `Agent 认为一致，仍需人工确认`；本篇 0.2593 美元。
4. 开发集全量：`cp artifacts/stage5_graph_dev_v1/api_cache.jsonl artifacts/stage5_hybrid_dev_v1/` 并把先导的 25 行追加进去后，
   `.venv/bin/python -m src.evaluation.run_graph --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --labels data/staged/eval_v2_labels/development_review_labels.jsonl --budget-usd 6 --output-dir artifacts/stage5_hybrid_dev_v1`
   → 66 篇约 30 分钟，产物在 `artifacts/stage5_hybrid_dev_v1/`（`summary.json`、66 份底稿、`api_cache.jsonl`、`run.log`）。

对照表（同一开发集与标签；控制组＝`artifacts/stage5_graph_dev_v1`）：

| 2,223 个数字 | 控制组（investigate 关） | 混合（investigate 开） |
|---|---|---|
| 错误 959：判不一致 | 618（64.4%） | 896（93.4%） |
| 错误放行（判一致） | 23，全部在末位一个单位内 | 23，同一批；非舍入 0 |
| 正确 1,206：确认 | 917 | 917 |
| 正确：误报 | 10 | 11 |
| 遗漏（完全没给结论） | 18（错误 13／正确 5） | 6（错误 1／正确 5） |
| 交人工（无依据／存疑／遗漏） | 655 | 360 |
| 无法核实 58 | 无依据 58 | 无依据 42／不一致 16 |
| 每篇新增花费 | 0（全部缓存） | 约 0.042 美元 |

- `agent_flagged` 295：wrong 278（TP）、correct 1（FP）、unverifiable 16。`agent_hint_consistent` 284：correct 275、wrong 5、unverifiable 4。两个方向的交叉与 `memory-bank/result.md` 里“判出可升至约 902／959、误报约 11、非舍入放行 0”的估计基本一致，实际判出 896／959。
- 两条硬性检查都通过：`wrong_passed_nonrounding = 0`；`consistent` 计数与控制组完全相同（correct 917、wrong 23，Agent 既没有放行也没有新增确认）。
- 花费：本次运行新增 2.7825 美元，先导 0.2593 美元，T11 合计 3.0418 美元，在 6 美元内；未用 GPU、未碰 vLLM 进程与 `artifacts/stage6_*`；未提交 Git、未装依赖。
- 那 1 个 FP 是中信证券“自营投资收入 263.45亿元”（标签为正确），Agent 取“证券投资业务 营业收入 240.40亿元”当应有值——属科目近义误配，与开发集交叉估计的“只把 1 个正确判错”一致。

未完成或有疑问：

1. 16 个 `unverifiable`（人工也没核实）的数字被 Agent 判成不一致；按现有标签口径它们既不是 TP 也不是 FP，实际正误未知。要不要单独过一遍这 16 条由你定（我没有把它算进误报）。
2. 4 篇的 Agent 答案解析不出条目（`review-flash_closed-600030-1219411922`、`review-pro_closed-000858-1223311527`、`review-pro_closed-002352-1219410699`、`review-pro_closed-601088-1222870380`）：一篇是步数用尽（`Sorry, need more steps…`），其余是答案被截断或夹带说明文字使 `parse()` 取不到完整数组。这些篇保留核查器结论并记 `agent_errors`，未加重试（与 T9“按原样计分”一致）；若要覆盖率可只对这 4 篇重跑一次（其余命中缓存）。
3. Agent 提示带整篇稿件（同 T9），每篇花费随工具调用次数近似平方增长（本任务 66 篇 2.78 美元）。若要压成本可在提示里要求“优先 `find_number`、少读整页”，但这会改变 Agent 行为，我没有自行加。
4. 按约定只改了 T11 列出的文件；`memory-bank/architecture.md` 的“编排与底稿”一节还没写进 investigate 节点，是否补由你决定。

### T10（已验收）

改了哪些文件：

- `src/agent/graph.py`：故障处理。`extract` 节点把模型调用包进 try／except，抛异常（网络、HTTP 错误、`BudgetExceeded`）时记 `extract failed: <类型>: <原因>`、`parse()` 取不到条目时记 `unparsable extract answer: '<前 200 字>'`，两种都不抛出、`items` 记空，于是稿件里每个数字在底稿中都是 `missed`（抽取遗漏需人工）。`revise` 节点逐条 try／except，单条失败该条记 `{'accepted': False, 'sentence': None, 'note': '需人工改写'}` 并把 `revise failed: …` 写进 `errors`，其余条照常处理。新增 `note()` 把错误追加到已有列表（不再覆盖），`extract`、`revise`、`investigate` 三处都用它，所以一次运行里多个节点的错误都会保留；文件头 docstring 补了一句“失败的模型节点不中断运行、不产生结论”。结论仍只来自核查器。
- `src/service/app.py`（新增）：`ThreadingHTTPServer`，只绑 127.0.0.1，默认端口 18100。`GET /` 返回同目录的 `index.html`；`GET /api/issuers` 返回出处库中有年报的发行人及其财年与披露日；`POST /api/review` 在锁内跑一篇 `review_draft()`、把底稿写成 `<data-dir>/<id>.json|.md` 后返回 `{id, workpaper}`；`GET /api/workpapers/<id>.json|.md` 从 `--data-dir` 读（文件名只允许字母数字与 `_-`，不接受路径分隔符）。400 的原因：模型不在白名单、正文为空、截止日不是真实的 `YYYY-MM-DD`、(发行人, 财年) 在库中没有年报、选本机模型但启动时没给 `--local-url`；请求不是 JSON 对象也返回 400。命令行 `--library --data-dir [--port 18100] [--local-url] [--budget-usd 0.5]`。
- `src/service/index.html`（新增）：原生 HTML／JS 单文件，无外部 CDN。表单为发行人下拉（来自 `/api/issuers`）→财年下拉（切换发行人时重填，并把该年报披露日自动填进截止日）、写稿截止日、模型、稿件正文；结果区为汇总计数（含 `agent_flagged` 与 `agent_hint_consistent`）、`errors` 红框逐条列出、稿件正文按结论着色（点击滚动到对应行并高亮）、逐数字表格（原文、结论、来源〔核查器／Agent〕、错因、应有值、出处“文件 第 N 页”、修改建议；`source='agent'` 的行显示 `Agent 认为有误／一致……需人工确认`，与 Markdown 版一致）、底稿 JSON 与 Markdown 的下载链接。
- `src/service/mcp_server.py`（新增）：`mcp` 包的 FastMCP，stdio。工具 `check_draft(code, fiscal_year, as_of, text)`（发行人不在库中返回 `{'error': …}`，否则返回底稿）、`find_number(text, code, as_of)`、`search_pages(query, code, as_of, limit)`（都返回 JSON 数组，同 `single.py` 的工具）、`page_text(document_id, page, as_of)`（披露日晚于 `as_of` 的文件拒绝，复用 `single.REFUSED`）。模型与预算由 `--model／--url／--budget-usd` 决定，底稿与缓存写到 `--data-dir`。
- `tests/test_agent_graph.py`：新增 `FailureTest` 5 项——抽取抛异常／答案乱码／预算用尽（真的走 `ApiClient('deepseek-flash', 0.0, …)`，抛 `BudgetExceeded`）三种都是全部 `missed`、`errors` 记因、`consistent` 与 `inconsistent` 均为 0；修订单条抛异常时该条“需人工改写”、另一条正常接受、结论不变；失败运行里每个数字恰好出现一次且不出现一致／不一致。
- `tests/test_service.py`（新增）：9 项，起在随机端口，只把 `review_draft` 换成一个用 `FakeMessagesListChatModel` 跑真实 `build_graph` 的函数（不需要密钥）：页面、发行人列表、正常核查＋JSON／Markdown 下载、发行人不在库、财年不在库、正文为空、三种坏截止日、本机模型没给 URL、未知底稿。
- `tests/test_mcp_server.py`（新增）：3 项，用 `mcp` 客户端经 `stdio_client` 启动 `python -m src.service.mcp_server`（模型指向测试内起的本机假模型端点）：列出四个工具并调用 `find_number`／`search_pages`／`page_text`（含“截止日后才披露”的拒绝）、`check_draft` 返回底稿、发行人无年报时返回错误。

验证命令及输出摘要：

1. `.venv/bin/python -m unittest discover -s tests -q` → `Ran 174 tests ... OK`（T11 为 157，本次 +17）。全程 CPU，未用 GPU、未联网、未装依赖。
2. 真实冒烟（用 flash 起服务，数据目录的 `api_cache.jsonl` 先复制 `artifacts/stage5_hybrid_dev_v1/api_cache.jsonl`）：`NO_PROXY=127.0.0.1,localhost .venv/bin/python -m src.service.app --library data/staged/library_v5/library.sqlite --data-dir /test/Finagent/runs/t10_smoke_data --port 18100 --budget-usd 0.5`，再 `POST /api/review {code:"000858", fiscal_year:2024, as_of:"2025-04-26", text:<开发集 review-pro_closed-000858-1223311527 全文>, model:"deepseek-flash"}` → `summary {"numbers": 25, "consistent": 4, "inconsistent": 20, "unsupported": 1, "ambiguous": 0, "missed": 0, "agent_flagged": 0, "agent_hint_consistent": 0}`；与 `artifacts/stage5_graph_dev_v1/workpapers/review-pro_closed-000858-1223311527.json` 的 `numbers／consistent／inconsistent／unsupported／ambiguous／missed` 六项逐个相同，25 个数字的位置与结论序列逐项一致。返回 JSON 与 Markdown 存到 `/test/Finagent/runs/t10_smoke.json`、`/test/Finagent/runs/t10_smoke.md`（服务端的同一份在 `/test/Finagent/runs/t10_smoke_data/`）。本次 `cost_usd_new = 0`（缓存仍 1313 行，抽取、修订与 investigate 全部命中），底稿里 25 个数字各出现一次。
3. `GET /api/issuers`（library_v5）→ 32 家发行人；示例 600309 万华化学 2023／2024／2025 三个财年与披露日 2024-03-19／2025-04-15／2026-04-21。
4. `git diff --check` 无输出；`python -m compileall` 通过。

未完成或有疑问：

1. 冒烟是零花费的：为了让 `summary` 与 T7 的底稿逐项可复现，缓存先复制了 T11 的产物，所以三次调用都命中缓存、没有真正发起 flash 请求（服务本身按 `deepseek-flash` 的正常路径运行）。若你要“不经缓存真跑一次”作为验收证据，我可以在 0.5 美元内再跑一次；但重新采样会让结果与 T7 底稿不同——本篇唯一交给人工的数字若被 Agent 判成“错误”，`summary` 会变成 `inconsistent 21／unsupported 0` 且来源为 Agent。请定以哪一种为准。
2. 本篇 Agent 的缓存答案是 T11 里那条不可解析文本，因此 `errors` 为 `["unparsable investigate answer: '四季度营业收入 …"]`、`agent_flagged` 为 0。这是 T11 定下“不重试”的直接结果，我没有改提示词或加重试。
3. `--budget-usd` 我给了默认值 0.5（任务里写作可选项）；页面与 MCP 都用它。若你要必填或别的默认值，我改。
4. `page_text` 的截止日判断是字符串比较（与 `single.py`、`review_draft` 一致，披露日为日粒度），没有改 `library.py`；`GET /api/workpapers` 只从 `--data-dir` 读文件，未做鉴权（服务只监听 127.0.0.1）。
5. 未改核查器、抽取提示词与 `judge()`，未新增配置文件，未碰 `artifacts/stage6_*` 与正在运行的 `run_graph` 进程；`memory-bank/architecture.md` 按约定留给你更新（“编排与底稿”一节需要补服务页面与 MCP 工具）。

### T12（已验收）

改了哪些文件：

- `src/agent/graph.py`：删除 `consistency` 节点、`intra_document()` 与只为它服务的 `written_value()`；`ReviewState` 去掉 `consistency` 字段，底稿每条去掉 `consistency` 键，Markdown 底稿去掉“稿内一致性”列；不再使用的导入 `norm`、`NARROWER`、`checkable` 一并去掉；节点顺序变为 extract → check →（investigate）→ revise → report，模块与 `build_graph`／`review_draft` 的说明文字同步改成“四节点”。核查结论来源、`judge()`、抽取提示词、修订逻辑与故障处理一字未改。
- `src/evaluation/run_graph.py`：计分汇总去掉 `consistency_findings` 键及只为它服务的 `groups` 计数。
- `tests/test_agent_graph.py`：删掉 3 个只测该节点的用例（同指标写两次、季度句不参与分组、措辞不同不分组）及其专用常量 `QUARTER_*`／`WORDING_*` 与辅助函数 `run_text`；文件首行说明改为 extract／check／investigate／revise／report。
- 未改：`src/service/index.html` 没有稿内一致性列（表头为原文／结论／来源／错因／应有值／出处／修改建议，无 `consistency` 引用），无可删内容；`memory-bank/architecture.md` 第 35 行仍写“六节点…consistency…”，按约定留给你更新。

验证命令及输出摘要：

1. `grep -rn "intra_document\|consistency" src tests` → 无输出（exit 1），即无任何残留结果（文本与 `.pyc` 都已随重新编译更新）。
2. `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 171 tests ... OK`（T10 为 174，本次 −3，正为删掉的 3 个用例）。全程 CPU、未联网、未装依赖。
3. 开发集缓存零花费重跑：`cp artifacts/stage5_hybrid_dev_v1/api_cache.jsonl /test/Finagent/runs/t12_check/`（1313 行），再
   `CUDA_VISIBLE_DEVICES= .venv/bin/python -m src.evaluation.run_graph --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --labels data/staged/eval_v2_labels/development_review_labels.jsonl --budget-usd 0 --output-dir /test/Finagent/runs/t12_check` → exit 0，66 篇跑完，输出在 `/test/Finagent/runs/t12_check/`（`summary.json`、66 份底稿、`api_cache.jsonl`、`run.log`）。与 `artifacts/stage5_hybrid_dev_v1/summary.json` 逐键对比（旧产物独有的 `consistency_findings` 按本次删除处理后，两边键集合相同）：除 `cost_usd_new`（2.7825 → 0）外全部相同；判出 `wrong_flagged 896`、放行 `wrong_passed_as_consistent 23`、非舍入放行 `0`、确认 `correct_confirmed 917`、误报 `correct_flagged 11`、`agent_flagged 295`（wrong 278／correct 1／unverifiable 16）、`agent_hint_consistent 284`、`revisions 628/627`、`by_truth` 各计数全部一致。缓存运行后仍 1313 行，`cost_usd_new = 0`，`run.log` 无 `failed`／`BudgetExceeded`；66 份底稿中无 `"consistency"` 键。

未完成或有疑问：

1. 为硬性保证“不调用付费 API”，这次把 `--budget-usd` 设为 `0`：缓存中有已花费 3.36 美元，任何未命中的请求都会被 `BudgetExceeded` 拒绝而不是真的发出去（该异常由各节点按故障处理捕获）。结果与旧 summary 完全一致，说明 66 篇的抽取、修订与 investigate 请求全部命中缓存、没有新请求。若你希望以别的预算值复跑，结果应当相同。
2. `src/service/index.html` 实际没有“稿内一致性列”，与任务描述不符，我没有为它编造改动，也未扩大范围；请确认是否还有别处（如页面汇总）需要一并去掉。
3. `memory-bank/architecture.md` 第 35 行的节点描述（六节点、consistency）与 `plan.md` 第 9 行“输出”里仍提到稿内一致性，均未改（不在 T12 文件清单内）。

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
| 2026-10-15 | T8 一致性精度与开发集全量 | 已验收 | 复核 summary：计分与 flash 重算逐项一致；旧规则 7 条稿内冲突人工核对全为误报（季度句 6、措辞不同 1），新规则 0 条；修订 628 条接受 627，抽 30 条全对；新增花费 0.04 美元；140 项测试通过。决定：负数丢符号导致的定位偏差属抽取与 `locate`，验收期间冻结不改，记为已知问题；核对表保留修正前 7 组作为依据。`api_client.py` 的并发改动是 Claude 加的断连重试。 |
| 2026-10-15 | T9 单 Agent 对照组 | 已验收 | 第一次启动 `codex exec` 因 stdin 未关闭空等 3 小时超时，加 `< /dev/null` 后重跑。复核：抽取请求哈希不变（66/66 命中旧缓存）；轨迹合理（每篇约 51 次工具调用）；抽查放行的错误数字（研发费用 30.27 实为 30.07 等）确属真错。151 项测试通过，花费 2.90 美元。决定：3 篇最终答复不可解析不加重试（对照组按原样计分）；弃用告警暂不处理；architecture.md 由 Claude 更新。基于结果设计混合方案 T11。 |
| 2026-10-15 | T11 混合方案 | 已验收 | 复核 summary：判出 618→896、确认 917 不变、误报 10→11、非舍入放行 0，Agent 来源 295 条中 wrong 278／correct 1／unverifiable 16；关闭 investigate 零花费复现旧结果；157 项测试通过，花费 2.78 美元。决定：unverifiable 被标出的单列报告、不计 TP／FP；4 篇不可解析不重试（保守，结论不变）；不加“优先 find_number”提示，以免在验收前改变组件。 |
| 2026-10-15 | T10 页面、MCP 与故障处理 | 已验收 | 复核：174 项测试通过；我自己起服务，首页、`/api/issuers` 正常，未知发行人返回 400 与中文原因；冒烟底稿与 T7 同篇逐项一致。决定：缓存复现即可作冒烟证据，不再真跑。 |
| 2026-10-15 | T12 移除稿内一致性节点 | 已验收 | 复核：grep 无残留；171 项测试通过；缓存零花费重跑开发集，判出、放行、确认、误报、Agent 计数与 T11 完全相同。architecture.md 由 Claude 更新。 |
