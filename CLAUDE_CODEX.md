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

### T13 补全出处库的报表绑定（已验收）

**背景**：验收集上约一半正确数字交给人工，主因是部分发行人的合并报表事实没有绑定到报表或单位（`memory-bank/result.md` 阶段 6）。Claude 初步定位了两类根因：
1. **比亚迪**（`002594-1222881496` 第 142 页）：报表上方一行写“财务附注中报表的单位为：千元”，随后是“1、合并资产负债表”，但该页 57 条事实 `unit`／`scale` 全为空，`line_items()` 与核查器都要求 `scale` 非空，因此整张报表不可用。
2. **格力电器**（`000651-1223330631` 第 111、112 页）：PDF 文本层完整（“合并资产负债表……单位：人民币元”，数值清晰），但这两页在库中 0 条事实，需查 MinerU 解析结果（`data/staged/cninfo_parsed_v1/000651-1223330631.json`）里这两页有没有表格块、表格为何没有映射到文本层（`table_map`）或没有生成事实（`table_facts`）。
不调用付费 API，不用 GPU（除非需要重新解析个别页面，那时先停下回报）。

- **做法**：
  1. 写一个只读诊断（可放在 `src/data/library.py` 之外的一次性命令，结果写进回报即可，不新增模块）：对 `library_v5` 中每份年报统计“合并三大报表（`STATEMENT` 匹配的表题）且 `scale` 非空的事实数”，列出为 0 或明显偏低（低于该发行人其他年份一半）的文件，并按根因归类（单位未识别、表格未映射、表题未识别、其他）。
  2. 针对能在 `table_facts.py`／`table_map.py` 规则层修的根因修正（如上面的单位句式），每条规则配单元测试，用真实页面文本片段做用例。不要为单个文件写特例。
  3. 用修正后的代码重建出处库到新目录 `data/staged/library_v6/`（命令见 `memory-bank/environment.md`，含两个 `--restatement`）；`library_v5` 不动。
- **验收**：
  - 回报修正前后（v5→v6）每份受影响年报的上述事实数，以及全库事实总数、带单位比例的变化；任何事实数减少的文件逐个说明；
  - 人工核对：从新增的“报表＋有单位”事实中按固定种子抽 40 条（覆盖所有受影响发行人），逐条对照页面文本确认数值、行标签、列（本年／上年）与单位正确，写入 `/test/Finagent/runs/t13_audit.jsonl`，回报正确数；
  - 不改变已正确绑定的事实：在 4 个未受影响的发行人（如万华化学、宁德时代、招商银行、贵州茅台）上比较 v5 与 v6 的合并报表事实，`(document_id, page, start, row_label, column_header, unit, scale)` 应完全相同，不同之处逐条说明；
  - `unittest discover` 通过。
- **禁止**：不改核查器、抽取、评测集与标签；不删除 `library_v5`。

### T14 合并报表页的文本层兜底（已验收）

**背景**：格力电器三份年报的合并报表页（如 `000651-1223330631` 第 111、112 页）PDF 文本层完整清晰（`项 目 附注 2024 年 12 月 31 日 2024 年 1 月 1 日`，每行“科目 附注 本期 上期”），但 MinerU 把整列挤进单个 `<td>`，`table_map` 映射不出事实。决定：对“页面文本含合并报表表题、但该页没有任何映射出的事实”的页面，直接从 PDF 文本层按行生成事实。不调用付费 API，不用 GPU。

- **做法**（在 `src/data/table_facts.py` 或 `src/data/library.py` 中选更自然的一处，最少代码）：
  1. 触发条件：页面文本（与库中 `pages.text` 相同的 PDFium 文本）匹配 `checker.STATEMENT` 的合并报表表题（含续页：上一页触发且本页无新表题、仍是“科目＋数字”行），且该页 `table_map` 没有产出事实。
  2. 表头：从“项 目 … 附注 … 年 月 日 …”一类表头行解析两列的期间文字，作为两列的 `column_header`（保持与现有事实同样的写法，使 `checker.role()` 能判本年／上年）；单位来自页内 `单位：` 声明（复用现有 `UNIT` 规则）。
  3. 行：形如“科目 [附注编号] 数字 [数字]”的行生成事实；每个数字的 `start`／`end` 是它在页面文本中的字符区间（与现有事实一致的约定）；科目行标签按现有 `norm` 习惯保留原文；只有一个数字时按它在行中的水平位置无法判断就跳过（不猜）；附注编号（如 `五、1`）不得被当成数值。
  4. `title` 用页内表题（续页继承），`scope` 按表题判断合并／母公司；这些事实另加可区分的来源标记（如 `unit_source` 或问题清单中注明 `text_layer`），以便统计与回溯。
  5. 单元测试：用格力第 111 页真实文本片段（含只有一个数字的行、附注编号、负数与括号数）。
- **重建**：到新目录 `data/staged/library_v7/`（v5、v6 不动），命令同 environment.md。
- **验收**：
  - 回报格力三份年报合并报表有单位事实数（v6→v7），以及全库因兜底新增的事实数与涉及的文件清单（兜底不应在已有映射事实的页面触发）；
  - 人工核对：新增兜底事实按固定种子抽 40 条（覆盖三份年报），对照页面文本核对数值、科目、本年／上年列与单位，写 `/test/Finagent/runs/t14_audit.jsonl`，回报正确数；
  - 用核查器做功能检查：对格力 2024 年报构造 5 条“营业收入／净利润／总资产……金额”的 `Claim`（取库中值），核查结论应为 consistent；
  - v6 中已有的事实在 v7 中完全不变（元组比较）；`unittest discover` 通过。
- **禁止**：不改核查器、抽取、评测集与标签。

### T15 科目近义误配：“营业收入”不能对到“营业总收入”（已验收）

**背景**：验收集上 9B LoRA 唯一一个非舍入放行：稿件写“营业收入 740.19 亿元”，实为营业总收入（营业收入 732.22 亿元）；抽取把措辞“营业收入”对到科目“营业总收入”，`src/verify/extract.py` 的 `matches()` 因共有“收入”而接受。金融企业与部分集团同时披露“营业总收入／营业收入”“营业总成本／营业成本”，两者数值不同。不调用付费 API，不用 GPU；另有评测进程在用 `data/staged/library_v7`，不要动它和 `/test/Finagent/runs/lib7/`。

- **做法**：`matches()` 增加一条通用规则：科目名含“总”而措辞不含“总”（或反之）、且去掉“总”后两者相同或互相包含时，判不匹配（如 营业总收入／营业收入、营业总成本／营业成本、利润总额 不受影响——给出你的规则如何处理“利润总额”与“利润”的说明并测试）。只改 `matches()` 与 `tests/test_extract.py`。
- **验收**（零费用，全部命中缓存）：
  - 用 `library_v5` 与当前代码，对开发集（缓存 `artifacts/stage3_reviews_dev_v6/api_cache.jsonl`）、训练集点评（缓存 `artifacts/stage3_reviews_train_v1/api_cache.jsonl`）、验收集 LoRA（`artifacts/stage6_accept_9b_lora/api_cache.jsonl`，模型名 `qwen3.5-9b-extract`，URL 随便填，缓存命中不发请求）各重跑 `run_reviews` 到 `/test/Finagent/runs/t15/<名称>/`；
  - 回报每组与修改前（同库同缓存）相比：结论改变的数字清单（真值、前后结论、原文片段），非舍入放行与误报的变化；验收集 LoRA 那条“740.19”应不再判一致；
  - 任何“正确数字从一致变为非一致”的情况逐条说明是否合理；`unittest discover` 通过。

### T16 复查 Agent 降本（已验收）

**背景**：开发集上复查 Agent 每篇约 0.04 美元，主要花在多轮工具调用中不断累积的上下文（单 Agent 对照每篇 13.5 万输入 token）。`page_text` 每次返回整页原文是最大来源。目标：在不放行错误的前提下把复查花费明显降低。本任务允许调用 `deepseek-flash`，总花费 ≤ 4 美元；不用 GPU；不要动 `data/staged/library_v7` 与 `/test/Finagent/runs/lib7/`。

- **做法**（只改 `src/agent/single.py` 及其测试）：
  1. `page_text(document_id, page, keyword="")`：给了 `keyword` 时只返回包含该关键词的行及其上下各 1 行（另附该页表题、单位声明行），未命中时返回“未找到”和该页前 5 行；不给时行为不变。工具说明里写明优先带关键词。
  2. `search_pages` 只返回文件、页码、表题与命中片段（不超过 80 字），不返回整页。
  3. 两个系统提示加一句：先用 `find_number` 按数字查找，查不到再检索页面。
  4. 截止日限制、权限与 `calculate` 不变。
- **评测**：用 `library_v5`，开发集 `run_graph`（混合，默认 investigate），输出 `artifacts/stage5_hybrid_dev_v2/`，先复制 `artifacts/stage5_hybrid_dev_v1/api_cache.jsonl` 作缓存（抽取与修订命中，复查因提示改变会重新调用）。
- **验收**：与 `artifacts/stage5_hybrid_dev_v1/summary.json` 对比：判出、非舍入放行（必须仍为 0）、确认（必须相同）、误报、Agent 标出的真值拆分、每篇复查花费与输入 token；若判出下降超过 3 个百分点，回报原因（不要自行再调）；`unittest discover` 通过。

### T17 报表单位在附注中声明的情况（比亚迪 2023）（已验收）

**背景**：比亚迪 2023 年报（`002594-1219412018`）合并资产负债表在第 138 页，页内与之前都没有单位声明；第 153 页附注写“报表所采用的货币均为人民币，除有特别说明外，均以人民币千元为单位表示”。现有 `DECLARED` 规则只向后生效，所以第 138 页起的报表事实没有单位，整张报表不可用。不调用付费 API，不用 GPU；另有评测在用 `data/staged/library_v7/` 与 `/test/Finagent/runs/indep/`，不要动。

- **规则**（`src/data/table_facts.py`，通用、最少代码）：一份文件中，若某句声明**明确说的是“报表”**（含“报表”二字）且给出金额单位（如上句），则该单位也适用于本文件中**没有任何单位**的合并／母公司报表页（`STATEMENT`／`PARENT` 表题）的事实；`unit_source` 标为可区分的值（如 `statement unit declared in notes`）。页内或前文已有单位的事实一律不变；文件中出现两种不同的“报表”单位声明时不应用（记入问题清单）。
- **重建**：`data/staged/library_v8/`（v5—v7 不动）。
- **验收**：
  - 回报全库因此规则获得单位的事实数与涉及文件（预期主要是比亚迪 2023；其他文件逐个说明）；
  - **跨年自动核对**：比亚迪 2023 年报合并三大报表的本年数（换算为元）与 2024 年报同一科目的上年数（v7 中已有单位）逐行比较，回报一致的行数／可比较的行数（重述项目单独列出）；
  - 人工核对 20 条新获单位的事实（固定种子），写 `/test/Finagent/runs/t17_audit.jsonl`；
  - v7 中已有单位的事实在 v8 中完全不变（元组比较）；`unittest discover` 通过（新规则配真实句子的单元测试，含“两种声明冲突不应用”与“非报表的单位声明不应用”）。

### T18 复查与修订 Agent 改用本地模型（进行中：A1、A2 已验收，A3 已批准待执行）

**Claude 2026-10-16 批准 A3（R0、R1、R2；R3 不跑）**：A2 作为管线冒烟验收——工具调用可解析、答案能走到 `parse()`；答案收尾重复触顶属于模型行为，正由 A3 的“不可解析 ≤ 3／66”判据衡量，**不调解码参数（温度、重复惩罚）、不改提示词与 `parse()`**。在你写的 A3 命令基础上改为分片并行：
- 服务与 A2 命令逐字相同（日志 `/test/Finagent/tmp/t18_a3_server.log`）。
- **R0**：`--shard i 8`（i=1..8）8 个进程同时跑，输出 `/test/Finagent/runs/t18_a3/r0/s<i>/`；全部结束后把 8 份 `api_cache.jsonl` 按请求去重合并为 `/test/Finagent/runs/t18_a3/r0/api_cache.jsonl`，再在 `r0/` 不带 `--shard` 整集重跑计分（应 0 新请求，回报核实）。
- **R1、R2 同时开跑**：各 8 片，每片输出 `r1/s<i>/`、`r2/s<i>/`，先拷入 R0 合并后的 `api_cache.jsonl` 作种子。R1 每片 `--budget-usd 0.50`（合计 ≤ 4 美元），`.env` 用 `set -a; . ./.env; set +a` 载入，`NO_PROXY` 只放 127.0.0.1／localhost。全部结束后分别合并抽取缓存与 `api_cache_<agent-model>.jsonl`，在 `r1/`、`r2/` 不带 `--shard` 整集重跑计分（0 新请求、0 新花费，回报核实）。R2 整集重跑时服务须仍在（只为防意外未命中；未命中数须为 0）。
- 某片因 HTTP 错误中断：同目录原命令重跑该片（缓存续跑），回报次数；同一片失败 2 次就停下回报。
- **报告**：按 T18 判据对 R2 逐条给出“达标／不达标”及数值（安全、有用、稳定三类，对照 R0、R1），以及“另报告”各项与失败归类；每个不可解析答案另记 `finish_reason`、长度、文中是否含可单独解析的 JSON 数组（仅作诊断，不计分）。另把 Agent 标出的数字按真值拆分（wrong／correct／unverifiable）分别给 R1、R2。
- 跑完 `pkill -f "vllm serve"`，确认显存释放；状态改“待验收”。不开始 A4，不跑 R3。

**Claude 2026-10-16 批准 A2**：按回报中的 A2 命令原样执行（单卡、端口 18090、`--max-model-len 40960`、`qwen3_coder`）。补充：
- 起不来（显存不足）时依次把 `--max-num-seqs` 降到 4、2 再试；不得改用 `--kv-cache-dtype fp8`、不得降 `--max-model-len`，仍起不来就停下回报。
- `qwen3_coder` 解析不出工具调用（第 4 步“带已解析工具调用”为 0）时，允许换 `qwen3_xml`、再换 `hermes` 各试一次，回报每种的结果；冒烟输出目录按解析器分开（`/test/Finagent/runs/t18_a2/<parser>/`）。
- 除通过标准外另报：该篇 Agent 请求数、最大单次输入 token、`finish_reason` 分布、耗时；抽 2 条工具调用与最终答案原文贴在回报里。
- 跑完必须 `pkill -f "vllm serve"` 并确认显存释放。A2 结束后**不要开始 A3**：把 A3 R0／R1／R2 的完整命令（含共用抽取缓存的做法、R1 的 flash 预算与 `.env`）写进回报，状态改“待验收”。

**背景**：阶段 6 的混合流程中，抽取已验证可换成本地 9B LoRA，但复查 Agent（investigate）与修订 Agent（revise）只用 `deepseek-flash` 验证过；部署时稿件不能出本机。要回答：**全链路本地模型时，复查 Agent 能否保住“只加报警、不放行”，判出率能否接近 flash。** 先测免训练模型，不达标再由用户决定是否训练。分四步，A1 可直接做；A2 起需起推理服务与评测，**每步开跑前回报完整命令，等用户确认**。flash 花费合计 ≤ 5 美元。

- **A1 按节点指定模型（只改代码，不跑模型）**：`src/evaluation/run_graph.py` 与 `src/agent/graph.py` 的 `review_draft()` 新增 `--agent-model`／`--agent-url`，用于 investigate 与 revise；不给时沿用 `--model`（行为与缓存键逐字不变，须有测试）；extract 仍用 `--model`。两个模型各自缓存、计费，缓存文件名写入 summary。`run_graph` 增加 `--shard i/n` 按稿件切片，合并缓存后整集计分（沿用 T16 独立检验的切片做法）。`src/model/api_client.py` 的 `LOCAL` 登记用到的本地模型名。`unittest discover` 通过。
- **A2 本地服务冒烟（1 篇开发集稿件）**：vLLM 0.25.1 提供 Qwen3.5-9B 基座＋抽取 LoRA（同阶段 6：`--enable-lora --max-lora-rank 16`），并开启工具调用 `--enable-auto-tool-choice --tool-call-parser <parser>`（parser 先查文档，再实测确定）。`--max-model-len` 由 `/test/Finagent/runs/indep/full/` 中 flash 复查轨迹的最大单次输入 token 决定；4090 放不下就回报，不得擅自截断工具输出。通过标准：至少一次工具调用解析成功、最终答案能被 `parse()` 解析、底稿出现 Agent 来源的结论或提示。
- **A3 开发集对照（66 篇，`library_v8`）**：各组抽取都用 9B LoRA 并共用同一份抽取缓存，组间只换 Agent 模型。
  - R0：9B LoRA 抽取＋核查，`--no-investigate`（全本地基线）；
  - R1：R0＋flash 复查与修订（参照）；
  - R2：R0＋本地 Qwen3.5-9B 基座复查与修订（不挂 LoRA）；
  - R3（仅 R2 不达标且用户确认后）：R0＋Qwen3.6-27B，4×4090 张量并行。
  - **判据（运行前写定，R2、R3 分别判）**：安全——确认数与 R0 完全相同，非舍入放行与 R0 相同；有用——判出率 ≥ R1 − 5 个百分点，误报率 ≤ R0 + 1 个百分点，Agent 标出数中真错误 ≥ 95%；稳定——复查答案无法解析的稿件 ≤ 3／66，修订“除该数字外逐字不变”通过率 ≥ 95%。
  - 另报告（不作判据）：每篇耗时、工具调用数、输入 token、撞 40 步上限次数、失败归类（工具参数格式、答案格式、用错工具、超步数、判断错误）。
- **A4 独立检验（仅 A3 达标后）**：训练集 192 篇（6,830 个数字），本地 Agent 从未在此运行。**抽取用 flash 缓存（`/test/Finagent/runs/indep/full/`，`library_v7`），不用 9B LoRA**——后者用这 192 篇训练过，在此抽取偏乐观。参照用已有结果（flash＋核查：判出 1,749、确认 3,116；flash 混合：判出 2,526／93.4%、非舍入放行 6、误报 4.45%）。判据：确认数仍为 3,116，非舍入放行 ≤ 6，判出率 ≥ 88.4%，误报率 ≤ 5.40%。
- **结论规则**：A3、A4 都达标则采用本地 Agent，全链路可私有化；否则回报失败归类，由用户决定是否训练。训练方案届时另立任务，待用户决定：训练数据用“flash 复查轨迹按核实标签筛选”（属蒸馏，与阶段 4“不以 API 为教师”不同，须写明）还是“由标签证据链合成轨迹”；`chat_encoding` 需改为监督全部 assistant 回合（含工具调用），属掩码语义改动；最终检验用新写并封存的验收发行人点评，还是如实标注“验收集复用”。
- **禁止**：不提交、不推送；不改核查器、抽取提示词、`judge()` 与复查提示词；A3 出结果后不调整判据；不动 `library_v5`—`v8` 与已有 `runs/`、`artifacts/` 目录。

### T19 交给人工的数字：查证、证据视图与反馈（a、b 待执行；c、d 待定）

**背景**：验收集与独立检验中，约一半正确数字仍交给人工。分析师或质控拿到的只有“交给人工＋原因”，还得自己去翻原文。目标是把人工这一步变成“看证据、点选结论”：Agent 按数字类型规划查证路线（c），证据和原件并排展示（b），人的点选沉淀为反馈（d）。安全底线不变：结论仍由核查器给出，Agent 只能加报警，任何新组件都不能放行数字。顺序为 a → b → c → d，每步验收后再开下一步；与 T18 无依赖，可以并行。

- **T19a 人工清单的原因占比（零费用，只读已有产物）**：
  - 数据：验收集混合 `artifacts/stage6_accept_hybrid/workpapers/`、训练集独立检验 `/test/Finagent/runs/indep/full/`、开发集混合 `artifacts/stage5_hybrid_dev_v2/`。
  - 对每个交给人工的数字，按“核查器原因（`reason`）× 核实标签（错误／正确／无法核实）× 是否带 Agent 提示”计数，并按发行人和数字类型（金额、增速、增减额、比率、其他）拆分。原因为“其他／未选到科目”的，抽 30 条读原句归类（自有口径比率、附注明细、季度／分部、非年报来源、预测数、文字中的数字等），写入 `/test/Finagent/runs/t19a_sample.jsonl`。
  - 回报：各类占人工清单的比例，以及每一类最可能的查证来源（年报附注、季报、政策／统计发布、无法查证）。这是 T19c 的选题依据，只回报，不改代码。
- **T19b 证据视图（不调用模型）**：
  - 页面（`src/service/app.py`＋`index.html`）：左边是稿件，交给人工和判为不一致的数字按原因用颜色高亮；点击后右边显示该数字的证据：原文 PDF 页面的渲染图，按事实的 `box` 框出数字（pypdfium2 渲染，PDF 坐标转像素时注意 y 轴翻转与缩放）；候选原文值与算式；Agent 的查证摘要（工具调用与命中的事实编号，来自底稿）。证据来源只用底稿和出处库中已有的事实编号，不新增推断。
  - 原件路径：按 `documents.id` 从 `configs/cninfo_sources.json` 找到 `data/raw/cninfo/` 下的文件，渲染前先核对 SHA256；对不上就拒绝显示并提示。没有文本层（无 `box`）的事实只显示整页，并注明“无法定位到数字”。
  - 反馈：每个数字三个按钮——“一致”“确有错误”“跳过”，可附一句备注；写入本机 `runs/web/feedback.jsonl`，字段为 review_id、数字起点、按钮、备注、时间、底稿版本。**反馈不改底稿结论**，只追加记录。
  - 渲染图和反馈只存本机，不外发。
  - 验收：单元测试覆盖坐标换算（构造已知 box 的页面）、哈希不符拒绝显示、无 box 的情况、反馈追加写入；按固定种子抽 20 条有 box 的事实，渲染后人工确认框住的就是该数字，截图存 `/test/Finagent/runs/t19b_boxes/` 并回报正确数；用 1 篇验收集稿件的底稿做端到端演示，回报截图；`unittest discover` 通过。
- **T19c Agent 按数字类型规划查证（待定，依据 T19a 细化后再开）**：从 T19a 占比最大、且有可查来源的一类开始，一次只做一类。复查 Agent 先判断该数字应去哪里查（年报正文／附注、季报、政策与统计发布），再用对应工具检索，给出算式交核查器复算。需要新数据源的（如季报），先单独立任务入库。每一类开发前写定判据：安全——确认数与改动前完全相同、非舍入放行不增加；有用——该类数字中获得“不一致”结论或带原文证据提示的比例，提升幅度由 T19a 的基数定；在开发集上设计，在训练集点评上独立检验。
- **T19d 反馈沉淀（待定，等 T19b 积累真实点选后再开）**：用反馈做两件事——对 Agent 提示和核查器原因按人工点选统计准确率，定位最常被推翻的类别，改进规则；把“确有错误”与“一致”的点选和证据组成案例库，供复查 Agent 检索相似案例作参考（参考不改变“只能加报警”）。开工前按反馈条数和类别分布确定做法与判据。
- **禁止**：不改核查器结论逻辑、`judge()` 与评测集标签；T19a 不运行模型；T19b 不调用任何模型或外部服务；不提交、不推送。

## 回报

### T18（A1 已完成；A2 已执行未达通过标准，待验收）

**A1 改了哪些文件**

- `src/agent/graph.py`：`build_graph(checker, llm, *, agent_llm=None, investigator=None)` 新增 `agent_llm`，`revise` 节点改用它；不给时用 `llm`。`review_draft()` 新增关键字参数 `agent_model`／`agent_url`／`agent_cache`，investigate 与 revise 都用它们，extract 仍用 `model`；三者默认取抽取的 `model`／`url`／`cache`，且三者都等于原值时复用同一个 `ApiChatModel`（同一缓存、请求体逐字不变）。
- `src/evaluation/run_graph.py`：新增 `--agent-model`／`--agent-url`（不给即沿用 `--model`／`--url`）；两个模型各自缓存——抽取用 `<输出目录>/api_cache.jsonl`，Agent 模型用 `<输出目录>/api_cache_<agent-model>.jsonl`；`summary` 新增 `cache`（用到的缓存文件名）与 `cost_usd_by_model`（按模型分别计费），`cost_usd`／`cost_usd_new` 口径不变；新增 `--shard I N` 只跑第 I 片（1 起、连续切分，越界报错）。docstring 写明“每片一个输出目录＋抽取缓存副本做种子，各片缓存按请求去重合并成 `api_cache.jsonl` 后整集重跑（零新请求）”的 T16 做法。新增三个小函数 `cache_rows()`／`cache_files()`／`shard_reviews()`，无新配置项。
- `src/model/api_client.py`：`LOCAL` 加 `qwen3.6-27b`（R3 用；`qwen3.5-9b`、`qwen3.5-9b-extract` 已在）。服务的 `--served-model-name` 必须与这三个名字一致，否则 `ApiClient` 以“没有价目”拒绝。
- 测试：`tests/test_agent_graph.py` +6（revise 用 Agent 模型；不给时全用抽取模型；investigate 仍走传入的 stub；`review_draft` 不给 Agent 模型时只有一个缓存文件、且抽取请求体逐字等于旧实现（按写定请求体重算 sha256 命中缓存）；给 Agent 模型时两个缓存各只含自己的模型；investigate 拿到 Agent 的 model／url／cache）；新增 `tests/test_run_graph.py` 7（切片覆盖整集一次、192 篇分 6 片各 32、越界拒绝、缓存命名、summary 的缓存名与按模型计费、`--shard` 只计当前片）。

**验证命令及输出摘要**

1. `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 204 tests in 11.457s OK`（T17 后 191，+13 为本次用例）。全程 CPU，未装依赖、未用 GPU、未跑模型、未调用 API。
2. `git diff --check` 干净；`.venv/bin/python -m src.evaluation.run_graph --help` 正常列出 `--agent-model`／`--agent-url`／`--shard I N`。
3. A2 需要的两个数（只读现有产物，未启动模型）：`/test/Finagent/runs/indep/full/` 六片缓存共 5,357 行，复查轨迹**最大单次输入 29,730 token**（最大输出 6,703）。Qwen3.5-9B 文本层 KV = 32 层 × 4 KV 头 × 256 维 × 2 B × 2 ≈ 128 KiB/token，`--max-model-len 40960`（≥ 29,730 输入 + 8,000 生成）需约 5.0 GiB KV；权重 17.98 GiB（其中视觉塔 3.1 GiB，`--language-model-only` 后语言部分约 14.9 GiB），`--gpu-memory-utilization 0.90` ＝ 21.6 GiB，余约 1.5 GiB 给激活与 LoRA，预计放得下（以 A2 实测为准）。
4. 工具解析器：本机 vLLM 0.25.1 注册表里有 `qwen3_coder`／`qwen3_xml`（同一个 `Qwen3EngineToolParser`）与 `hermes` 等；模型自带 README 的 vLLM 命令用的是 `--tool-call-parser qwen3_coder`。本模型的工具调用格式是 `<tool_call><function=…><parameter=…>`（`chat_template.jinja` 第 53、112 行），A2 先用 `qwen3_coder`，解析不出再试 `qwen3_xml`／`hermes`。

**A2 完整命令（等你确认后再跑，预计 10—20 分钟；不提交、不改 v5—v8 与已有 runs/）**

```bash
# 1) 起本地服务：基座 + 抽取 LoRA + 工具调用（单卡）
cd /root/Finagent && V=$PWD/.venv-serving && \
PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cu13/lib \
nohup setsid $V/bin/vllm serve \
  /test/Finagent/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --served-model-name qwen3.5-9b --host 127.0.0.1 --port 18090 \
  --max-model-len 40960 --gpu-memory-utilization 0.90 --max-num-seqs 8 --enforce-eager \
  --language-model-only --enable-lora --max-lora-rank 16 \
  --lora-modules qwen3.5-9b-extract=/test/Finagent/runs/stage4_extract_v1/step_0072 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  > /test/Finagent/tmp/t18_a2_server.log 2>&1 &

# 2) 等服务就绪（约 3 分钟）
for i in $(seq 60); do
  NO_PROXY=127.0.0.1,localhost curl -sf http://127.0.0.1:18090/v1/models && break; sleep 5
done

# 3) 冒烟：开发集第 1 篇，抽取用 9B LoRA、复查与修订用 9B 基座（本地服务免费）
cd /root/Finagent && NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
.venv/bin/python -m src.evaluation.run_graph \
  --library data/staged/library_v8/library.sqlite \
  --reviews data/staged/eval_v2/development_reviews.jsonl \
  --labels data/staged/eval_v2_labels/development_review_labels.jsonl \
  --output-dir /test/Finagent/runs/t18_a2/full --budget-usd 0 \
  --model qwen3.5-9b-extract --url http://127.0.0.1:18090/v1/chat/completions \
  --agent-model qwen3.5-9b --agent-url http://127.0.0.1:18090/v1/chat/completions \
  --shard 1 66

# 4) 验收证据（工具调用是否解析成功、答案是否可解析、底稿是否有 Agent 来源结论）
cd /root/Finagent && .venv/bin/python -c "
import glob, json
rows=[json.loads(l) for l in open('/test/Finagent/runs/t18_a2/full/api_cache_qwen3.5-9b.jsonl')]
print('Agent 请求', len(rows), '；带已解析工具调用', sum(1 for r in rows if r['tool_calls']))
paper=json.load(open(glob.glob('/test/Finagent/runs/t18_a2/full/workpapers/*.json')[0]))
print('errors', paper['errors']); print('summary', paper['summary'])"

# 5) 收尾
pkill -f "vllm serve"
```

通过标准（T18 原文）：至少一次工具调用解析成功（第 4 步“带已解析工具调用”> 0）、最终答案能被 `parse()` 解析（`errors` 里没有 `unparsable investigate answer`）、底稿出现 Agent 来源的结论或提示（`summary.agent_flagged` 或 `summary.agent_hint_consistent` > 0）。任一条不过我把日志与失败归类回报，等你定是否换 `qwen3_xml`／`hermes` 或调 `--max-model-len`。

**未完成或有疑问**

1. **只有 1 张 4090 可见**（24,564 MiB，当前无进程占用）：A3 的 R3（4×4090 张量并行）现在跑不了，届时要么换机要么由你确认卡数。A2／A3 的 R0—R2 单卡可做。
2. `--max-model-len` 按 T18 规则取 flash 轨迹最大输入 29,730 + 8,000 生成余量 = 40,960；本地模型的轨迹可能比 flash 长（工具调用次数不同），超上限时 vLLM 会直接报错、该篇按现有规则保守留在人工清单（不会截断工具输出）。是否放宽上限由你定；若 40,960 起不来，可选项是 `--kv-cache-dtype fp8`（KV 减半到 2.5 GiB，会轻微改变数值）或降并发，我没有擅自加。
3. 缓存分开后预算也各自起算：`--budget-usd` 由每个客户端按自己缓存里的已花金额判断（原来共用一个缓存，Agent 的额度会把抽取已花的算进去）。R1“本地抽取＋flash Agent”不受影响（抽取免费、Agent 额度就是它的上限）；若将来用付费抽取，Agent 的额度要在抽取花费之外另给。
4. A3／A4 未开始（按 T18 要求等你对 A2 的确认）；`library_v5`—`v8` 与既有 `runs/`、`artifacts/` 未改动。`architecture.md` 里 `review_draft` 签名属小改，按惯例留给你。

**A2 执行结果（2026-10-16，按你 2026-10-16 的批准原样执行：1 篇冒烟，三条通过标准过 1 条）**

一、服务、解析器与收尾

- 服务按 A2 命令逐字起：`--max-model-len 40960` 一次起得来，**没有**出现显存不足，故未降 `--max-num-seqs`、未换 `--kv-cache-dtype fp8`。11:20:30 启动、约 35 秒就绪（`/v1/models` 同时列出 `qwen3.5-9b` 与 LoRA `qwen3.5-9b-extract`）；加载后 `nvidia-smi` 21,878／24,564 MiB。
- 工具调用解析器：`qwen3_coder` 直接可用（27 个回合全部解析出工具调用），没有出现“带已解析工具调用为 0”，所以**未**换 `qwen3_xml`／`hermes`，冒烟产物仍在 `/test/Finagent/runs/t18_a2/full/`，未另建 `<parser>/` 目录。
- 收尾：`pkill -f "vllm serve"` 后服务日志末行 `Application shutdown complete.`，无 vllm 进程，显存回到 1 MiB／24,564 MiB。
- 除本文件外没有改任何仓库文件；`library_v5`—`v8`、既有 `runs/` 与 `artifacts/` 未动；全程本地、花费 0 美元。

二、通过标准逐条

| 通过标准（T18 原文） | 结果 | 证据 |
|---|---|---|
| 至少一次工具调用解析成功 | **通过** | Agent 缓存 37 个请求中 27 个回合带已解析工具调用，合计 50 次工具调用（`page_text` 17、`find_number` 16、`search_pages` 12、`calculate` 5） |
| 最终答案能被 `parse()` 解析 | **不通过** | `errors` 里有一条 `unparsable investigate answer: '从第7页可以看到：…'` |
| 底稿出现 Agent 来源的结论或提示 | **不通过** | `summary.agent_flagged = 0`、`summary.agent_hint_consistent = 0`（因果上由第 2 条导致） |

失败归类（第 2 条）：**答案格式失败 ＋ 重复输出触顶**，不是工具层失败。

- 出问题的是 investigate 第 28 回合（最后一个回合）：`finish_reason='length'`，8,000 token 输出全部用满后截断，该回合不再调用工具。
- 答案原文是叙述体：先复述“从第7页可以看到：…”的四个数字，中间给出一版 16 条的 JSON 数组，随后“等等，我需要重新检查一些数据”逐条自检、再重发一版同样的数组——同一句“现在让我生成最终的JSON数组”出现 6 次，全文 14,004 字、217 行。提示词要求“除这个 JSON 数组外不要输出任何文字”，模型没有遵守；`parse()` 只认裸 JSON 数组，于是这 16 条判断（其中 14 条判“错误”）全部落空。
- 排除其他归类：不是工具参数格式错（50 次调用全部解析成功，没有工具报错）、不是用错工具（`find_number`→`search_pages`→`page_text`→`calculate` 的用法都对）、不是撞 40 步上限（28 回合主动收口）、不是超 `--max-model-len`（最大单次输入 19,608 ≪ 40,960）。

该篇计分（`/test/Finagent/runs/t18_a2/full/summary.json`）：31 个数字，一致 6／不一致 9／无依据 12／存疑 4／遗漏 0；非舍入放行 0、误报 0。因为 Agent 答案整体落空，这篇实际退化成“确定性核查 ＋ 用抽取模型做的修订”，对本地复查能力没有提供信息（原因见第 3 条）。

三、你要求另报的数

第 4 步验收脚本原样输出：

```
Agent 请求 37 ；带已解析工具调用 27
errors ["unparsable investigate answer: '从第7页可以看到：\n- 营业收入：83,272,067,317.19元（832.72亿元），同比增长12.58%\n- 归属于上市公司股东的净利润：30,210,585,269.30元（302.11亿元），同比增长13.19%\n- 总资产：165,432,981,684.75元（1654.33亿元），同比增长8.26%\n- 归属于上市公司股东的净资产：129,558,241,040.51元（1295']
summary {'numbers': 31, 'consistent': 6, 'inconsistent': 9, 'unsupported': 12, 'ambiguous': 4, 'missed': 0, 'agent_flagged': 0, 'agent_hint_consistent': 0}
```

- **该篇 Agent 请求数**：37（investigate 28 回合 ＋ revise 9 次）；另加 1 次抽取请求（`qwen3.5-9b-extract`），本次服务共 38 次请求，全部本地、花费 0（`cost_usd_by_model`：`qwen3.5-9b` 0.0、`qwen3.5-9b-extract` 0.0）。
- **最大单次输入 token**：19,608（investigate 第 28 回合）。分项：investigate 28 回合合计输入 318,105／输出 14,102；revise 9 次合计输入 1,023／输出 348；抽取 1,546／3,180。
- **`finish_reason` 分布**（37 个 Agent 请求）：`tool_calls` 27、`stop` 9、`length` 1。
- **耗时**：11:21:2x 起、11:34:50 止，约 13 分半（抽取 ≈3.4 分钟；investigate 28 回合 ≈8.5 分钟；revise 9 次 ≈1.5 分钟）。38 次请求合计输入 320,674、输出 17,630，端到端生成速率**约 21 token／秒**，明显低于 `environment.md` 记的阶段 4 的 120 token／秒；本次是 `--max-model-len 40960` ＋ `--enforce-eager`、prompt 最长 19.6k，原因未查。若 A3 沿用这个速率：R0（66 篇只抽取＋核查）约 3 小时，R2（再加本地复查）按本篇 13.5 分钟／篇估算要十小时以上——A3 开跑前请先定是否分片。

抽 2 条工具调用与最终答案原文（节选）：

- 第 1 回合一次批量 16 条 `find_number`，前 3 条：`find_number {"text": "764.21亿元"}`、`find_number {"text": "12.9%"}`、`find_number {"text": "628.42亿元"}`
- 第 19 回合：`calculate {"expression": "(115456300910.64 - 92358426975.79) / 92358426975.79 * 100"}`
- 最终答案（第 28 回合，8,000 token 处截断，原文 14,004 字）开头与中途：

```
从第7页可以看到：
- 营业收入：83,272,067,317.19元（832.72亿元），同比增长12.58%
- 归属于上市公司股东的净利润：30,210,585,269.30元（302.11亿元），同比增长13.19%
- 总资产：165,432,981,684.75元（1654.33亿元），同比增长8.26%
- 归属于上市公司股东的净资产：129,558,241,040.51元（1295.58亿元），同比增长13.62%

现在让我生成最终的JSON数组。
（原文此处开始是一段 JSON 代码围栏，内容为下面这一版 16 条数组）
[ {"number": "764.21亿元", "occurrence": 1, "verdict": "错误", "expected": "764.47亿元", "basis": "2023年年度报告，第14页，酒类收入76,447,093,245.62元"},
  {"number": "12.9%", "occurrence": 1, "verdict": "错误", "expected": "13.15%", "basis": "……"} , … 共 16 条 ]

等等，我需要重新检查一些数据。让我确认一下：
1. 764.21亿元（酒类收入）：年报为764.47亿元，错误
……
现在让我生成最终的JSON数组。
（随后重复一版同样的数组，直到 8,000 token 上限被截断）
```

四、A3 R0／R1／R2 的完整命令（按你的要求写在这里，A3 未开始）

共用抽取缓存的做法：三组抽取请求体逐字相同（同 `--model`／`--url`／提示词），缓存键是请求体的 sha256，与 `--budget-usd`、缓存文件名无关；所以 R0 先跑一遍把 66 篇的抽取答案写进 `/test/Finagent/runs/t18_a3/r0/api_cache.jsonl`，R1／R2 各自把这份文件拷进自己的输出目录当种子，抽取即全部命中缓存（零新请求、零新花费），组间只差 Agent 模型。`summary.cache` 与 `cost_usd_by_model` 会分列两个缓存与两个模型的费用。

```bash
# 0) 起本地服务（R0 抽取用；R2 的复查与修订复用同一服务；与 A2 命令逐字相同，只改日志名）
cd /root/Finagent && V=$PWD/.venv-serving && \
PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cu13/lib \
nohup setsid $V/bin/vllm serve \
  /test/Finagent/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --served-model-name qwen3.5-9b --host 127.0.0.1 --port 18090 \
  --max-model-len 40960 --gpu-memory-utilization 0.90 --max-num-seqs 8 --enforce-eager \
  --language-model-only --enable-lora --max-lora-rank 16 \
  --lora-modules qwen3.5-9b-extract=/test/Finagent/runs/stage4_extract_v1/step_0072 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  > /test/Finagent/tmp/t18_a3_server.log 2>&1 &

# 1) 等服务就绪
for i in $(seq 60); do
  NO_PROXY=127.0.0.1,localhost curl -sf http://127.0.0.1:18090/v1/models && break; sleep 5
done

# 2) R0：9B LoRA 抽取＋核查，无工具 Agent（全本地基线；同时产出三组共用的抽取缓存）
cd /root/Finagent && NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
.venv/bin/python -m src.evaluation.run_graph \
  --library data/staged/library_v8/library.sqlite \
  --reviews data/staged/eval_v2/development_reviews.jsonl \
  --labels data/staged/eval_v2_labels/development_review_labels.jsonl \
  --output-dir /test/Finagent/runs/t18_a3/r0 --budget-usd 0 \
  --model qwen3.5-9b-extract --url http://127.0.0.1:18090/v1/chat/completions \
  --no-investigate

# 3) R1：抽取命中 R0 缓存（零新抽取请求）＋ flash 复查与修订
mkdir -p /test/Finagent/runs/t18_a3/r1 && \
cp /test/Finagent/runs/t18_a3/r0/api_cache.jsonl /test/Finagent/runs/t18_a3/r1/api_cache.jsonl
cd /root/Finagent && NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
.venv/bin/python -m src.evaluation.run_graph \
  --library data/staged/library_v8/library.sqlite \
  --reviews data/staged/eval_v2/development_reviews.jsonl \
  --labels data/staged/eval_v2_labels/development_review_labels.jsonl \
  --output-dir /test/Finagent/runs/t18_a3/r1 --budget-usd 4.00 \
  --model qwen3.5-9b-extract --url http://127.0.0.1:18090/v1/chat/completions \
  --agent-model deepseek-flash --agent-url https://api.deepseek.com/chat/completions

# 4) R2：抽取同样命中 R0 缓存＋本地 Qwen3.5-9B 基座复查与修订
mkdir -p /test/Finagent/runs/t18_a3/r2 && \
cp /test/Finagent/runs/t18_a3/r0/api_cache.jsonl /test/Finagent/runs/t18_a3/r2/api_cache.jsonl
cd /root/Finagent && NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
.venv/bin/python -m src.evaluation.run_graph \
  --library data/staged/library_v8/library.sqlite \
  --reviews data/staged/eval_v2/development_reviews.jsonl \
  --labels data/staged/eval_v2_labels/development_review_labels.jsonl \
  --output-dir /test/Finagent/runs/t18_a3/r2 --budget-usd 0 \
  --model qwen3.5-9b-extract --url http://127.0.0.1:18090/v1/chat/completions \
  --agent-model qwen3.5-9b --agent-url http://127.0.0.1:18090/v1/chat/completions

# 5) 收尾
pkill -f "vllm serve"
```

- **R1 的 flash 预算与 `.env`**：`--budget-usd 4.00` 是本次**新请求**的上限（T16 开发集换新工具后实测 1.74 美元；T18 全任务 flash 合计上限 5 美元）。密钥不用 source：`ApiClient` 默认从仓库根的 `.env` 读 `DEEPSEEK_API_KEY`（权限 600、不进 Git），所以 R1 必须先 `cd /root/Finagent`。R1 只把 `127.0.0.1,localhost` 放进 `NO_PROXY`，`api.deepseek.com` 仍走机器代理。重跑时预算要按“缓存里已计的金额 ＋ 新上限”重设（缓存里的已花金额计入 `spent`）。
- R0／R2 的 `--budget-usd 0` 可行：`qwen3.5-9b` 与 `qwen3.5-9b-extract` 都在 `src/model/api_client.py` 的 `LOCAL` 里、单价 0，且本地模型不检查预算。
- 服务参数与 A2 相同（一次起得来，21,878 MiB）；但如果要压时长，可对三组分别用 `--shard i/n` 分片（每片一个输出目录，跑完整集时把各片缓存按行去重合并成 `api_cache.jsonl` 再整集重跑命中缓存，见 `run_graph` docstring 的 T16 做法）——分片只改并发与耗时、不改判据，是否采用等你定。

**未完成或有疑问**

1. A2 的第 2、3 条通过标准没过，且不属于你批准的两类重试情形（显存不足、工具调用解析为 0），所以我没有换 `qwen3_xml`／`hermes`、没有动 `--max-model-len`，停下等你定。可以确认的是：换解析器只影响工具调用解析（该项已通过），对“答案格式”这条大概率无用；与它直接相关的是复查提示词的收尾约束、或让 `parse()` 容忍代码围栏——前者属本任务禁止修改的范围。
2. 一次冒烟只有 1 篇，且这篇的 Agent 答案整体落空，所以**不能**推断 66 篇的落空率；A3 判据里“复查答案无法解析的稿件 ≤ 3／66”正是为此设的，建议就用 A3 的 R2 去看它是否复现。
3. R0 命令里没有 `--agent-model`，按现有代码 revise 会用抽取模型（`qwen3.5-9b-extract`）写修改建议，且这些请求写进同一个 `api_cache.jsonl`。这与任务里 R0 的写法一致，但意味着 `r0/api_cache.jsonl` 是“抽取＋修订”混在一起的文件（行按请求哈希区分，拷给 R1／R2 当种子仍然正确，也不影响计费口径）。若你要 R0 完全不跑修订，需要加一个开关，属新代码，等你决定。
4. A3／A4 未开始；未提交、未推送、未装依赖、未调用付费 API（A2 全程本地、花费 0），既有 `runs/`、`artifacts/` 与 `library_v5`—`v8` 未改动。

### T17（已验收）

改了什么文件：

1. `src/data/table_facts.py`（只在两处加代码）：
   - 新增 `NOTES_UNIT` 正则与 `notes_declared_unit(document)`：扫描全文页面文本，找**含“报表”二字**且给出金额单位的句子（如“编制本财务报表所采用的货币均为人民币，除有特别说明外，均以人民币千元为单位表示”），返回该单位与“是否出现两种不同单位”的冲突标记。
   - `extract()` 在原有单位链条（页内表头／页眉／前文声明）都取不到时，对 `statement_table()`（报表表题或“主要会计数据”）的表取该单位，`unit_source='statement unit declared in notes'`；整列并格走文本层兜底的那条路径同样处理；同一文件出现两种不同“报表”单位时不应用，记问题 `conflicting statement units declared in notes`。
2. `tests/test_table_facts.py`：新增 3 个用例——附注声明生效（含同单位重复声明）、两种冲突单位不生效、不点明“报表”的单位句不读（用库中真实句子：京东方“除特别注明外，金额单位为人民币元”、万科 2023 第 279 页、中国平安 2025 第 41 页）。

验证命令及输出摘要：

1. `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 191 tests ... OK`（T16 后 188，+3 为本任务用例）。未装依赖、未用 GPU，未碰 `data/staged/library_v7/` 与 `/test/Finagent/runs/indep/`。
2. 重建 v8（v5—v7 未动）：
   `.venv/bin/python -m src.data.library --restatement configs/jd_restatement.json --restatement configs/jd_restatement_xinlun.json --output-dir data/staged/library_v8`
   收据 `data/staged/library_v8/receipt.json`：92 份报告／1,855 篇发布／27,079 页／537,032 条事实／31,121 条问题（v7：537,032／31,151）。
3. 全库 v7→v8 逐事实比较（脚本 `/test/Finagent/tmp/t17/compare.py`）：
   - 事实总数不变；带单位+量级 395,034 → **395,446**；只有键增减 0；
   - 获得单位 **412、丢单位 0、单位改变 0、unit_source 改标 0**；412 条全在比亚迪 2023（`002594-1219412018`），`千元`／量级 1000，来源即新标注；另有同文件 18 条只补上 `currency='人民币'`；
   - v7 中已有单位的 **459,848** 条事实按元组（document_id, page, start, row_label, column_header, unit, scale）比较：**0 条变化**；
   - 新获单位按页：138(54)、140(51)、141(24)、142(43)、144(43)、146(38)、148(62)、149(58)、150(39)；
   - 问题清单唯一变化：`no unit for table` **−30，无新增**。比亚迪 2023 −11，其余 −19 分布在 000858、002475、300750、300059、300760、002594(2024/2025) 的报表类表；这些表级问题消失，但相关文件没有一条事实变化（同页事实的单位原本由行标签等给出）。
   - 全库共 18 份文件的附注有这类句子（000858×3、002475×3、002594×3、300059×3、300750×3、300760×2、新纶 2018 1），其中没有一份出现两种不同单位，故没有冲突问题；除比亚迪 2023 外的文件均无事实变化。
4. 跨年自动核对（脚本 `/test/Finagent/tmp/t17/cross_year.py`；比亚迪 2023 本年列 vs 2024 年报同一行上年列，换算为元）：
   - 合并资产负债表：可比 27 行，一致 26，不一致 1（交易性金融资产 9,562,550 vs 9,542,789 千元）；
   - 合并利润表／现金流量表：可比 53 行，一致 51，不一致 2（减:营业成本 480,558,350 vs 490,398,945；销售费用 25,211,395 vs 15,370,800 千元）。
   - 3 个不一致都是**已披露的重述**而非本规则出错：2024 年报第 25／184 页写按《企业会计准则解释第 18 号》把保证类质量保证由“销售费用”改列“营业成本”（对应营业成本、销售费用两行），第 184 页还写把原列“交易性金融资产／负债”的衍生金融工具单独列报（对应交易性金融资产，2024 表另有“衍生金融资产”行）；2024 年报第 142—146 页报表页均标“（经重述）”。
5. 人工核对 20 条新获单位的事实（固定种子 `random.Random(0)`，按 document/page/start 排序），写 `/test/Finagent/runs/t17_audit.jsonl`：值 20/20、列 20/20、单位 20/20，行标签 19/20 与页面原文一致；可比跨年 11 条 11/11 一致；18 条判 `correct`、2 条 `correct_with_caveat`。两条 caveat 都源自既有的行标签处理、与单位规则无关：一条库中行标签为空（标签印在上一行），一条库中行标签带了上文小节标题（“按经营持续性分类其中:持续经营净利润”，页面原文为“其中：持续经营净利润”）。

未完成或有疑问：

1. 比亚迪 2023 报表页（138—151）还剩 **180 条**事实无单位，集中在续表页 139(58)、143(50)、145(2)、147(48)、151(22)：这些页在版面解析结果里没有报表表题（`title` 为空或为“2022 年”），而本规则要求表题先被认作报表才生效，所以没覆盖。是否另加续表继承规则由你定。
2. `memory-bank/architecture.md` 的 `table_facts` 条目与 `environment.md` 的当前库版本（仍写 v7）未更新；按任务文件清单我没有改文档，留给文档负责方。
3. 规则只认含“报表”二字的句子，因此京东方等“除特别注明外，金额单位为人民币元”这类句子不生效（符合任务要求）；若要覆盖需另立规则。

### T16（已验收）

改了哪些文件：

- `src/agent/single.py`：
  - `page_text(document_id, page, keyword="")`：带 `keyword` 时只返回含该关键词的行各上下 1 行，并在这段前面附该页表题（取自出处库 `facts.title`）与单位声明行（复用 `src/data/table_facts.py` 的 `UNIT` 正则扫页面行，能命中“（人民币百万元，特别注明除外）”这类写法）；未命中时返回“未找到“X”。该页前 5 行：…”。不带 `keyword` 时行为与原来逐字相同（整页、`PAGE_CHARS` 截断）；工具说明写明优先带 `keyword`。
  - `search_pages`：改为返回 `document_id`、`document`、`page`、`table`（该页表题）与 `snippet`（首条命中行、命中词前留一点上下文、总长 ≤80 字）。原实现本来就不返回整页正文，只返回文件、页码与元数据；我按新要求去掉了 `publisher`／`published`／`kind`，加上了表题与命中片段。保留 `document_id` 是因为 `page_text` 要用它。
  - 两个系统提示（`SYSTEM`、`INVESTIGATE_SYSTEM`）各加一句“先按点评里写的数字用 find_number 精确查找，查不到再检索页面”，并把 `search_pages`／`page_text` 的说明改成新行为。
  - 新增常量 `SNIPPET_CHARS = 80` 与 4 个模块级小函数（`table_titles`、`unit_lines`、`keyword_lines`、`snippet`）；没有新配置项，截止日限制、`REFUSED`、文件权限与 `calculate` 未动。
- `tests/test_single_agent.py`：测试库加一页带单位行的小表与一条事实，新增 2 个用例（关键词摘录含表题、单位声明行与上下各 1 行、且不含无关行，未命中回退到前 5 行；检索返回文件／页码／表题／短片段）。

验证命令及输出摘要：

1. `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 188 tests ... OK`（T15 后 186，+2 为本次用例）。全程 CPU、未装依赖、未用 GPU、未碰 `data/staged/library_v7` 与 `/test/Finagent/runs/lib7/`。
2. 评测：先 `cp artifacts/stage5_hybrid_dev_v1/api_cache.jsonl artifacts/stage5_hybrid_dev_v2/`，再
   `.venv/bin/python -m src.evaluation.run_graph --library data/staged/library_v5/library.sqlite --reviews data/staged/eval_v2/development_reviews.jsonl --labels data/staged/eval_v2_labels/development_review_labels.jsonl --output-dir artifacts/stage5_hybrid_dev_v2 --budget-usd 7.36`。
   预算 7.36 = 缓存里已计的 3.3613 + 本次新请求上限 4.00；**实际新增花费 1.7396 美元**（`summary.json` 的 `cost_usd_new`），66 篇约 1 小时。逐篇产物在 `artifacts/stage5_hybrid_dev_v2/`。
3. 结论对比（同库、同缓存种子；判出率 = 判不一致 ÷ 错误 959）：

| 指标 | v1（原工具） | v2（本任务） |
|---|---|---|
| 判出（错误 959） | 896（93.4%） | 915（95.4%） |
| 非舍入放行 | 0 | 0 |
| 确认（正确 1,206） | 917 | 917（完全相同） |
| 误报 | 11（0.91%） | 14（1.16%） |
| Agent 标出（真值拆分） | 295（错 278／对 1／无法核实 16） | 314（错 297／对 4／无法核实 13） |
| Agent 提示“一致” | 284（错 5／对 275／无法核实 4） | 285（错 4／对 273／无法核实 8） |
| 修订建议 | 628（接受 627） | 628（接受 627） |
| `cost_usd_new` | 2.7825 | 1.7396 |

   判出没有下降（+2.0 个百分点），不触发任务里“下降超过 3 个百分点要回报原因”的条款。2,223 个数字里 76 个结论变化：错误数字由“无依据”变“Agent 判错”40 个、反向 19 个；正确数字新增 4 个误报、恢复 1 个（原 Agent 误报 → 无依据）。新增的 4 个误报都是舍入级差异（4.8% 对 4.64%、1,654.33 对 1,654.32 亿元、9.1% 对 9.06%、9.2% 对 9.15%）。
4. 复查花费与输入 token：v1 的 `cost_usd_new` 少算了它从 T11 试点缓存复用的 0.2593 美元，所以这部分我用同一口径重放测量——从两次运行的 workpaper 重建各自“交给人工”的数字，对各自缓存逐篇重跑 investigate；两次都全程命中缓存、缓存行数不变（1,313 行与 1,844 行），没有发出新请求。脚本与逐篇结果在 `/test/Finagent/runs/t16/`：

| 复查（真正跑了 Agent 的 51／50 篇，交给人工的数字同为 655 条） | v1 | v2 |
|---|---|---|
| 输入 token | 9,320,675 | 4,942,056（−47.0%） |
| 输出 token | 204,688 | 214,137 |
| 花费 | 3.0418 美元 | 1.7396 美元（−42.8%） |
| 每篇（跑过 Agent 的篇） | 0.0596 美元；均 18.3 万／中位 7.7 万 token | 0.0348 美元；均 9.9 万／中位 6.2 万 token |
| 单篇输入 token 最大 | 1,262,893 | 488,884 |
| 工具调用合计 | 2,167（page_text 717／find_number 555／search_pages 542／calculate 353） | 2,073（page_text 568，其中 69% 带 keyword／find_number 565／search_pages 650／calculate 290） |

   与“每篇约 0.04 美元、13.5 万输入 token”对照：复查每篇约降到 0.034 美元、9.9 万输入 token；最重的一篇从 1,262,893 token、0.3902 美元、120 次工具调用降到 197,743 token、0.0649 美元、49 次。花费下降主要来自上下文不再累积整页原文，工具调用次数只小幅下降。

未完成或有疑问：

1. **一篇的复查因 HTTP 405 中断**：`review-flash_closed-601088-1222870380` 的 investigate 在第 8 轮左右拿到 `HTTPError: HTTP Error 405: Method Not Allowed`（`agent_errors` 只有这一条）。流程按原有约定保守处理，该篇 27 个数字全部留在“交给人工”，而 v1 曾标出其中 10 个错误。把这一篇按同一口径剔除后重算：判出 872/935（93.3%）→ 901/935（96.4%），确认 914 对 914，误报 11 → 14。这是外部错误、不是工具改动造成的，但它是本次唯一没能跑完的一篇，**是否重跑该篇（或整集）由你定**；按任务要求我没有自行重跑，也没有对缓存里该篇已完成的请求重复计费（它们已计入 1.7396 美元）。
2. 新增的 4 个误报都是舍入级（上面列出）。确定性核查器对舍入差有专门判断，而 Agent 报警不经过这一层，更精确的读数反而让 Agent 把稿件里已舍入的数判错。要不要把舍入容差也用到 Agent 报警上属于改语义，我没有动。
3. `architecture.md` 里“四个受限工具”的描述没有提 `page_text` 的 keyword 与 `search_pages` 的片段返回；按 T16 的文件清单我没有改文档，留给文档负责方。
4. 单位声明行是复用 `table_facts` 的 `UNIT` 正则扫页面行得到的，没有新增配置；原文里“收入达到人民币 1,957 亿元”这类句子按该正则的设计不会误判为单位声明。

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

### T13（已验收）

改了哪些文件：

- `src/data/table_facts.py`（只改这一个模块的规则层，共 4 组）：
  1. 新增 `STATEMENT_UNIT`（“财务附注中报表的单位为：千元”这类只给报表用的单位声明）；`declared_units()` 同时收集 `DECLARED` 与它，并给每条声明标一个“仅用于报表题表格”的布尔（`statement_table()`）；`extract()` 的单位回退改走新的 `declared_unit()`。对应比亚迪 2024／2025 的 p142／p123。
  2. `OTHER_UNIT` 加 `(?<!编制)(?<!填报)`：不再把“编制单位:珠海格力电器股份有限公司”读成单位（v5 里的伪单位 `深圳迈瑞生物`、`宁德时代新能` 都由此而来）。
  3. `context()`：识别到表题的那一行同时读该行单位（京东方A 把“……合并资产负债表 (续) (金额单位:人民币元)”写在同一行）。
  4. 新增 `LABEL_UNIT`／`label_unit()`：主要会计数据表的行标签自带单位（“营业收入（元）”“总资产（万元）”），只对报表题表格生效。
- `tests/test_table_facts.py`：+6 个用例（每条规则的文本片段都取自本次实际出问题的真实页面），+1 个 `Page` 替身（只实现 pdfium 的文本页接口）供文档级声明的用例使用。
- 未改 `table_map.py`（两个根因都不在映射层）、未改核查器／抽取／评测集／标签；`library_v5` 未动。

验证命令及输出摘要：

1. `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 177 tests ... OK`（T12 为 171，+6 正是本次新增用例）。全程 CPU、未联网、未装依赖、未调用付费 API、未用 GPU。
2. 重建：`.venv/bin/python -m src.data.library --restatement configs/jd_restatement.json --restatement configs/jd_restatement_xinlun.json --output-dir data/staged/library_v6` → 约 6 分钟；`receipt.json`：92 reports／1855 releases／27079 pages／**535429 facts**／31151 problems（v5 为 535429 facts／31209 problems）。
3. 全库与逐条差异：事实总数 535429 不变；带 scale 390926（73.0117%）→ 393443（73.4818%）；`no unit for table` 问题 8650→8592。逐条 diff：丢 scale 0 条、非空 unit 变化 0 条、非空 scale 变化 0 条；**2517 条事实新增 unit＋scale 绑定**（来源：报表声明 1573／行标签 584／表题 360），另有 1347 条只是 `unit_source` 改标（unit 与 scale 一字未变）。按 `document_id` 分组的事实数差集为空——**没有任何一份文件的事实数减少**。
4. 受影响年报的合并口径报表事实（表题命中三大报表或“主要会计数据”且 scope=合并；记法 v5→v6，三大报表/主要会计数据）：

| 发行人 | 年 | document_id | v5 → v6（三大报表/主要会计数据） |
|---|---|---|---|
| 万科A | 2023 | 000002-1219487237 | 333/33 → 333/35 |
| 美的集团 | 2023 | 000333-1219429374 | 325/0 → 325/18 |
| 美的集团 | 2024 | 000333-1222951181 | 320/8 → 320/26 |
| 美的集团 | 2025 | 000333-1225065145 | 319/0 → 319/18 |
| 格力电器 | 2023 | 000651-1219928418 | 0/0 → 0/18 |
| 格力电器 | 2024 | 000651-1223330631 | 0/0 → 0/18 |
| 格力电器 | 2025 | 000651-1225250396 | 0/0 → 0/18 |
| 京东方A | 2023 | 000725-1219494121 | 150/12 → 313/42 |
| 京东方A | 2024 | 000725-1223193313 | 84/12 → 250/42 |
| 京东方A | 2025 | 000725-1225068510 | 346/12 → 346/27 |
| 五粮液 | 2023 | 000858-1219873658 | 198/0 → 198/30 |
| 五粮液 | 2024 | 000858-1223311527 | 201/0 → 201/18 |
| 五粮液 | 2025 | 000858-1225273091 | 204/0 → 204/18 |
| 海康威视 | 2023 | 002415-1219703072 | 258/0 → 258/24 |
| 海康威视 | 2024 | 002415-1223153152 | 256/0 → 256/30 |
| 海康威视 | 2025 | 002415-1225123062 | 179/0 → 179/18 |
| 立讯精密 | 2023 | 002475-1219799516 | 69/12 → 145/30 |
| 立讯精密 | 2024 | 002475-1223326862 | 260/12 → 260/30 |
| 立讯精密 | 2025 | 002475-1225104908 | 421/12 → 421/30 |
| 比亚迪 | 2023 | 002594-1219412018 | 0/0 → 0/18 |
| 比亚迪 | 2024 | 002594-1222881496 | 0/12 → 163/30 |
| 比亚迪 | 2025 | 002594-1225045351 | 0/12 → 163/30 |
| 东方财富 | 2023 | 300059-1219307793 | 267/12 → 267/30 |
| 东方财富 | 2024 | 300059-1222806709 | 272/12 → 272/30 |
| 东方财富 | 2025 | 300059-1225019509 | 275/12 → 275/30 |
| 宁德时代 | 2023 | 300750-1219313047 | 273/12 → 273/42 |
| 宁德时代 | 2024 | 300750-1222806982 | 285/12 → 285/36 |
| 宁德时代 | 2025 | 300750-1225002214 | 158/30 → 278/30 |
| 迈瑞医疗 | 2023 | 300760-1219856065 | 62/0 → 170/18 |
| 迈瑞医疗 | 2024 | 300760-1223371173 | 144/0 → 263/18 |
| 迈瑞医疗 | 2025 | 300760-1225059012 | 379/0 → 379/9 |

（其余 61 份年报在两份库中完全相同；受影响文件的三大报表列没有一处减少，全部持平或增加。）

5. 人工核对 40 条：写入 `/test/Finagent/runs/t13_audit.jsonl`（固定种子 0，从 v6 新增“报表＋有单位”事实 2202 条中抽 40，覆盖 31 份文件、11 家受影响发行人），逐条对照页面文本核对数值、行标签、列（本年／上年）与单位（`page_text_at_span` 即事实数值，同行窗口给出标签与相邻列数值，12 条续表不在窗口内给出表头，另用同表同行的配对事实在表头页复核列序）→ **40/40 正确**。
6. 未受影响发行人对照（元组 `(document_id, page, start, row_label, column_header, unit, scale)`）：万华化学 15283 条（其中合并报表 873）、招商银行 38921（1882）、贵州茅台 10668（818）、工商银行 29005（1032）→ 全部 0 处不同。

未完成或有疑问：

1. **格力电器三份年报仍 0 条三大报表事实，根因在解析层。** 这 6 张报表页的 MinerU 输出把整列挤进了单个 `<td>`：2024 年 p111 的“合并资产负债表”只有 2 行 4 列，最长单元格 505 字（行标签、附注、两个数值列各是一格）；2023 年 p114/p115、2025 年 p86/p87 同样。`table_map` 拿不到行列，`table_facts` 无事实可绑。要修必须重新解析这三份文件（需 GPU + 文档模型服务），按任务约定先停下回报，**是否授权重解析？**（三份的“主要会计数据”表已由第 4 条规则补上，各 +18 条。）
2. **宁德时代实际受影响**，与任务验收里“未受影响”的举例不符：236 条事实的 unit 由伪单位 `宁德时代新能`（“编制单位：…”被误读）改为 `千元`、scale 由空变 1000；全部是新增或纠正，无丢无改（抽样 #8／#10／#15／#39 人工核对正确）。所以第 6 项的四个对照发行人改用万华化学、招商银行、贵州茅台、工商银行。
3. **比亚迪 2023（002594-1219412018）修后三大报表仍 0 条**（合并口径现有 18 条都来自主要会计数据表）。该年报的报表单位只写在附注里（p153“编制本财务报表所采用的货币均为人民币……均以人民币千元为单位表示”），位置在报表（p137–148）之后；现有回退只向前继承本页及之前的声明（`page <= pdf_page`）。若允许“附注里的报表单位声明向前套用到报表题表格”，这一份就能修好：扫全库 90 份年报，该句式只出现在比亚迪 3 份与中国平安 3 份，比亚迪 2024/2025 已由第 1 条规则修好、中国平安的报表本就已绑定，**影响面只有这一份**。但这是改变“声明生效方向”的策略，我没有自行扩大范围，请你定（做的话要改规则＋重建 v6＋重抽 40 条）。
4. `unit_source` 有 1347 条从 `statement on page X`／`page header` 改标为 `title`（表题行现在也读单位、优先级更高），unit 与 scale 完全不变；全库分布因此变化（`None` 79370→77184、`title` 2052→3759、`row label` 0→584、`statement on page *` 120233→120642、`line before table` 236149→235818、`page header` 35753→35570）。若下游有按 `unit_source` 过滤的逻辑请确认。
5. 按约定只改了 T13 列出的文件；`memory-bank/architecture.md` 的数据流那句（“`table_facts` 绑定行标签、列表头、单位等”）没有细化到“报表单位声明／行标签单位”，未改，需要的话请你补。

### T14（已验收）

改了哪些文件：

- `src/data/table_facts.py`（只改这一个模块的规则层）：
  1. 新增 `STATEMENT_LINE`（页面文本里作为报表表题的行：合并／母公司／公司＋资产负债表／利润表／现金流量表，含“（续）”；不含所有者权益变动表——它的行不止两个数值列）、`HEADER_ROW`（“项 目 … 附注 …”表头行）、`NOTE_REFERENCE`（行标签后的附注编号，含 `五、70（1）` 这类带括号子号）、`YEAR_END`／`OPENING`。
  2. 抽出共用函数：`column_headers()`、`rows_to_facts()`（把放置好的单元格与数字变成事实，原 `extract()` 内联代码原样搬出，映射路径行为不变）；新增 `statement_title()`、`header_line()`、`column_headings()`、`column_windows()`、`text_layer_cells()`、`open_label()`。
  3. `extract()` 在映射循环之后新增一段：对“该页没有任何映射事实”的页面，若页面文本有表题（或上一页有表题而本页无新表题，即续页）且有“项 目 … 附注 …”表头行，就按行读同一个 PDFium 文本层；单位取该页表头上方的 `单位：` 行，`unit_source` 记 `text layer line before table`。已有映射事实的页面不进入这段。
  4. 表头两列期间文字的写法与现有事实一致；`2024 年 1 月 1 日` 这类期初列改写成它等于的上一年年末（`2023年12月31日`），使 `checker.role()` 能判本年／上年（新增事实 805 条上年、798 条本年，无一判不出）。
  5. 只有一个数字的行按它在行中的水平位置判列（数值在列内右对齐，用“数值 x 中心是否落在第二列表头左边界左侧”判），判不出就跳过；附注编号（`五、1`、`五、70（1）`）从行标签里去干净、不当数值；行标签保留原文。
  6. 与映射路径同样处理“两行印在同一行”“标签跨行折断（括号未闭合）”“`xxx：` 是小节标题”。
- `tests/test_table_facts.py`：+8 个用例（片段全部取自格力三份年报的真实页面文本），+`text_layer_found()`／`printed()` 两个替身。
- 未改 `table_map.py`、核查器、抽取、评测集与标签；`library_v5`、`library_v6` 未动（首次构建的 v7 目录已删除后用最终代码重建，最终目录只含一次构建的产物）。

验证命令及输出摘要：

1. `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 185 tests ... OK`（T13 为 177，+8 正是本次新增用例）。全程 CPU、未联网、未装依赖、未调用付费 API、未用 GPU。
2. 重建：`.venv/bin/python -m src.data.library --restatement configs/jd_restatement.json --restatement configs/jd_restatement_xinlun.json --output-dir data/staged/library_v7` → 约 6.5 分钟；`receipt.json`：92 reports／1855 releases／27079 pages／**537032 facts**／31151 problems（v6：535429 facts／31151 problems；problems 不变，兜底既不产生也不消除问题）。
3. 格力三份年报“表题命中三大报表、`scope=合并`、unit 与 scale 均非空”的事实数：

| 年报 | document_id | v6 → v7 |
|---|---|---|
| 2023 | 000651-1219928418 | 0 → 308 |
| 2024 | 000651-1223330631 | 0 → 306 |
| 2025 | 000651-1225250396 | 0 → 302 |

（合并三表全部条目为 312／310／306，差额各 4 条是“基本／稀释每股收益”，unit=`元/股`、scale 为空，被上面的口径排除。）

4. 全库因兜底新增 **1603 条**（带 scale 393443→395034，占 73.4818%→73.5587%），只涉及 3 份文件，没有任何事实消失：

| 文件 | 新增 | 合并资产负债表／（续）／利润表／现金流量表 | 公司资产负债表／（续）／利润表／现金流量表 | 页码 |
|---|---|---|---|---|
| 000651-1219928418 | 538 | 63／66／99／84 | 46／51／64／65 | 114–117、120–123 |
| 000651-1223330631 | 535 | 62／66／100／82 | 46／51／64／64 | 111–114、117–120 |
| 000651-1225250396 | 530 | 60／65／99／82 | 45／51／64／64 | 86–89、92–95 |

触发条件检查：新增事实所在的页面在 v6 中均无任何事实（0 处冲突），即兜底没有在已有映射事实的页面触发；`unit_source` 分布为 `text layer line before table` 1591 条、`per share` 12 条。

5. 人工核对 40 条：写入 `/test/Finagent/runs/t14_audit.jsonl`（固定种子 0，覆盖三份年报 14／13／13 条，16 条本年／24 条上年），对照页面文本核对数值（`page_text_at_span` 与页面文本逐字相同；1603 条全部通过这一项）、科目（同行窗口给出标签与同排数值）、列（数值在 PDF 中的 x 中心与表头两期列的 x 区间比较）、单位（页内 `单位：人民币元`）→ **40/40 正确**；样本含 4 条负数、3 条带括号的科目名。另做一项独立校验：29 条“只有一个数字”的事实，数值中心离其所属列表头窗口最近者全部命中（29/29）。
6. 核查器功能检查（格力 2024，`as_of=2025-04-30`，金额取自 v7 库）：营业收入 189,163,654,064.64、归属于母公司股东的净利润 32,184,570,372.28、资产总计 368,031,704,522.86、货币资金 113,900,461,797.94、经营活动产生的现金流量净额 29,369,250,570.66 → **5/5 consistent**（同一组在 v6 上是 2 consistent／3 unsupported）。命中事实分别是 p7 主要会计数据、p113 合并利润表、p111 合并资产负债表（货币资金同页）、p114 合并现金流量表，后四者都是本次兜底新增。
7. v6 vs v7 元组比较（`document_id, page, start, end, box, text, digits, row_label, section, column_header, title, statement_date, unit, scale, currency, unit_source, scope` 的多重集）：removed 0、added 1603——v6 已有事实在 v7 中完全不变。最后一次改动只是 `extract()` 的 docstring，未改行为：用当前代码对三份格力年报重跑 `extract()`，事实数与问题数分别为 5006／533、4919／504、4796／413，与 v7 库中同文件完全一致，构建产物与当前源码一致。

未完成或有疑问：

1. **兜底也读了母公司报表页**：这三份年报的“公司资产负债表／公司利润表／公司现金流量表”页同样被 MinerU 挤进单格，所以兜底在 675 条母公司口径事实上生效（任务写的是“合并报表页”，而 `checker.STATEMENT` 本身也匹配母公司表题，我按“页面文本匹配 `STATEMENT` 的表题”实现；若只做合并口径，同一份年报的母公司报表页会仍然 0 条事实）。**如需只做合并口径，请指明。**
2. **母公司页的 `scope` 仍是 `None`**：表题写作“公司资产负债表”（不是“母公司…”），现有的 `scope()` 只在表题含“合并／母公司”时给值；v6 里已有 517 条同题事实也是 `None`，改成 `母公司` 会同时改掉 v6 已有事实（违反“v6 事实不变”），故未动。这 675 条不会污染合并口径，但 `figures(scope='母公司')` 也查不到。
3. **`unit_source` 新增一类取值** `text layer line before table`（1591 条）。若下游有按 `unit_source` 过滤或统计的逻辑，需认识这一取值。
4. 核对时发现并修正：现金流量表续页的行标签原本把附注写成“五、70（1）”，只去掉“五、1”式编号的旧正则去不掉它，`norm` 后变成“……现金五70”而查不到（44 条）；`NOTE_REFERENCE` 已能连同括号子号一并去掉，修正后这类事实可用（如 `支付其他与经营活动有关的现金` 2024 年 10,924,246,751.62 判 consistent）。
5. 兜底只覆盖“表题与表头行同在页内”的报表；若某页只有表头行、上一页又不判为同表，则不读。这三份年报 8 张报表页全部命中，未见遗漏。
6. 按约定只改了 T14 列出的文件；`memory-bank/architecture.md` 的数据流说明未提“文本层兜底”，未改，需要的话请你补。没有提交 Git。

### T15（已验收）

改了哪些文件：

- `src/verify/extract.py`：只在 `matches()` 内部加了一段（8 行，含注释）。在 `norm()` 之后、原有“包含即通过”之前判断：两个名字一个含“总”、另一个不含时，把含“总”的去掉“总”，若与另一个**完全相同**就返回 `False`；`pipeline.judge()` 随即给出 `unsupported`（原因 `line item does not fit the wording`）。没有新增模块级函数，没有改 `pipeline.py`、`checker.py`、抽取提示词、评测集与标签。
- `tests/test_extract.py`：+1 个用例 `test_total_and_plain_line_items_are_not_each_other`（8 条断言）。

“利润总额”与“利润”：任务写“去掉‘总’后两者相同或互相包含”，我实现的是“相同”（去掉“总”后完全相等），没有实现单向包含，因为两者冲突——`利润总额` 去掉“总”得 `利润额`，它**包含** `利润`；若单向包含也判不匹配，`matches('利润总额','利润')` 与 `matches('利润','利润总额')` 会由一致翻成不一致，与任务“利润总额 不受影响”矛盾。收紧成相等后，只有“总”恰好能整体删掉的对会改判，“利润总额”与“利润”两个方向仍为 True（已进测试）。**若你的本意是单向包含也要判不匹配，请指明，我再改。**

验证命令及输出摘要：

1. `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -q` → `Ran 186 tests ... OK`（T14 为 185，+1 正是本次新增用例）。全程 CPU、未联网、未装依赖、未调用付费 API、未用 GPU。
2. 三组 `run_reviews`，都带 `--library data/staged/library_v5/library.sqlite --budget-usd 0`，并把对应缓存先复制进输出目录；先按修改前的代码跑到 `/test/Finagent/runs/t15/<名称>_before/`，改完再跑到任务指定的 `/test/Finagent/runs/t15/<名称>/`：
   - 开发集：`--reviews data/staged/eval_v2/development_reviews.jsonl --labels data/staged/eval_v2_labels/development_review_labels.jsonl`，缓存 `artifacts/stage3_reviews_dev_v6/api_cache.jsonl`；
   - 训练集：`--reviews data/staged/eval_v2/training_reviews.jsonl --labels data/staged/eval_v2_labels/training_review_labels.jsonl`，缓存 `artifacts/stage3_reviews_train_v1/api_cache.jsonl`；
   - 验收集 LoRA：`--reviews data/staged/eval_v2/acceptance_reviews.jsonl --labels data/staged/eval_v2_labels/acceptance_review_labels.jsonl --model qwen3.5-9b-extract --url http://127.0.0.1:18090/v1/chat/completions`，缓存 `artifacts/stage6_accept_9b_lora/api_cache.jsonl`。
   零费用证据：三组 `api_cache.jsonl` 行数 66／192／94 前后完全相同，`summary.json` 的 `cost_usd` 仍是缓存内累计（0.2785／0.8631／0.0）——352 个请求全部命中缓存，没有发出任何新请求（只有真发请求才会追加缓存行并增加花费）。修改后的三组我还整跑了两遍，结果逐项相同。
   注意：`artifacts/stage3_reviews_dev_v6/records.jsonl` 是比率符号修正**前**的旧产物（它记 623/918，当前代码是 618/917），所以“修改前”一律用我自己的 `_before` 重跑，不用旧产物。
3. 结论计数（同库同缓存，修改前 → 修改后）：

| 组 | 数字 | 改判 | 判不一致（错误） | 错误放行 | 确认（正确） | 误报（正确判错） |
|---|---|---|---|---|---|---|
| 开发集 | 2,223 | 0 | 618 → 618 | 23 → 23 | 917 → 917 | 10 → 10 |
| 训练集 | 6,830 | 3 | 1,688 → 1,688 | 83 → 83 | 3,025 → 3,022 | 164 → 164 |
| 验收集 LoRA | 3,256 | 10 | 675 → 670 | 16 → 15 | 885 → 882 | 45 → 44 |

4. 任务点名的“740.19”：验收集 LoRA `review-flash_closed-600031-1219890100 @76`（真值 wrong）现在 `verdict=unsupported`、`reason='line item does not fit the wording'`（抽取给的是 `phrase="营业收入"`、`label="营业总收入"`）。三一重工 2023 年报合并利润表同时有 `一、营业总收入`=740.19 亿元与 `其中:营业收入`=732.22 亿元，两个都真实存在且不同，改判符合事实。
5. 非舍入放行：验收集 LoRA 的“错误放行”由 16 条降到 15 条，减少的**恰好**是 740.19（阶段 6 判定的唯一非舍入放行），其余 15 条逐条与修改前相同（集合差 = {740.19}）→ 修改后**非舍入放行 0**。三组都没有任何数字从非一致变成一致（新增 `consistent` = 0）。
6. 结论改变的数字清单（共 13 条，全部是 一致／不一致 → 交给人工；没有一条从非一致变成一致）：

| 组 | 点评 | 数字 | 真值 | 改判 | 原文片段（截断） |
|---|---|---|---|---|---|
| accept | review-flash_closed-600031-1219890100 @76 | 740.19亿元 | wrong | consistent→unsupported | 公司2023年实现营业收入740.19亿元 |
| accept | review-pro_closed-600031-1223129214 @24 | 777.73亿元 | correct | inconsistent→unsupported | 三一重工2024年…实现营业收入777.73亿元 |
| accept | review-pro_open-600031-1225061378 @17 | 1021.07亿元 | wrong | inconsistent→unsupported | 2025年，三一重工实现营业收入约1021.07亿元 |
| accept | review-flash_closed-600309-1219331064 @51 | 1753.61亿元 | correct | consistent→unsupported | 全年实现营业收入1753.61亿元 |
| accept | review-pro_closed-600309-1219331064 @24 | 1,753.61亿元 | correct | consistent→unsupported | 万华化学2023年…实现营业收入1,753.61亿元 |
| accept | review-flash_closed-600309-1223097325 @41 | 1820.69亿元 | correct | consistent→unsupported | 全年实现营业收入1820.69亿元 |
| accept | review-pro_open-600309-1223097325 @54 | 1820.55亿元 | wrong | inconsistent→unsupported | 2024年，万华化学实现营业收入1820.55亿元 |
| accept | review-flash_closed-600309-1225130793 @46 | 2108.35亿元 | wrong | inconsistent→unsupported | 全年实现营业收入2108.35亿元 |
| accept | review-pro_closed-600309-1225130793 @24 | 1,905.44亿元 | wrong | inconsistent→unsupported | 万华化学2025年…实现营业收入1,905.44亿元 |
| accept | review-pro_open-600309-1225130793 @56 | 1,943.51亿元 | wrong | inconsistent→unsupported | 2025年，万华化学实现营业收入1,943.51亿元 |
| train | review-flash_closed-002415-1223153152 @17 | 924.96亿元 | correct | consistent→unsupported | 2024年，海康威视实现营业总收入924.96亿元 |
| train | review-pro_closed-002415-1219703072 @18 | 893.40亿元 | correct | consistent→unsupported | 海康威视2023年全年实现营业总收入893.40亿元 |
| train | review-pro_closed-002415-1219703072 @31 | 7.4% | correct | consistent→unsupported | （同上句）同比增长7.4% |

7. 正确数字改判逐条说明（**没有任何“一致→不一致”的正确数字**；4 条降到交给人工，1 条从误报改为交给人工，均合理但要知悉代价）：
   - `三一重工 777.73`（pro_closed 2024，改前 inconsistent = 误报）：改前抽取把措辞“营业收入”对到科目“营业总收入”（其 2024 合并值为 783.83 亿元）故判不一致；稿件 777.73 正是 2024 年合并利润表 `其中:营业收入` 的值 → 真值 correct。改后交给人工，误报 45→44，**合理**。
   - `万华化学 1753.61／1,753.61／1820.69`（3 条，改前 consistent）：万华化学合并利润表里 `一、营业总收入` 与 `其中:营业收入` **数值相同**（2023 均 1753.61 亿元、2024 均 1820.69 亿元），稿件写“营业收入”本身正确，抽取选成“营业总收入”也不改变数值。改后因名字不同而交给人工：**数值没有被否定**，只是多 3 条人工，属本规则的保守代价。
   - `海康威视 924.96／893.40／7.4%`（3 条，改前 consistent）：海康威视利润表只有 `一、营业收入`，没有“营业总收入”；抽取把措辞“营业总收入”对到 `营业收入`，数值一致。改后交给人工，同为保守代价。
   - 结论：正确数字的代价集中在“稿件措辞与所选科目差一个『总』、但该发行人两项同值或本无『总』项”的情况，共 6 条，占三组 12,309 个数字的 0.05%；方向是交给人工而非误报，安全。
8. 错误数字改判（5 条 + 740.19 = 6 条 由 inconsistent 变 unsupported）：三一重工 1021.07；万华化学 2108.35／1905.44／1820.55／1943.51。这 5 条原本被正确判不一致，现在交给人 → 判出少 5 条，但**没有一条被放行**（仍交给人），是安全方向的代价。
9. `git diff --check` 无空白问题；本次只改 `src/verify/extract.py` 与 `tests/test_extract.py` 两个文件。

未完成或有疑问：

1. “互相包含”收紧为“去掉总后相等”，理由见上（否则 `利润总额`/`利润` 会被判不匹配，违反“利润总额 不受影响”）。这是与任务原文的唯一偏离，请确认。
2. 万华化学／海康威视那类“措辞与科目差一个总、但两项同值或本无『总』项”的正确定位会多交给人工（本次 6 条）。更精确的做法是让 `matches()` 能取到数值、在该发行人两项同值时放行，但那要动 `pipeline.py`／`checker.py`，超出 T15 文件清单，未做；本期只按任务要求做名字层面的规则。
3. `memory-bank/architecture.md` 的“核查”一节与 `progress.md` 的“后续可做”都提到这条规则，按约定不在 T15 文件清单内，未改文档。
4. 未提交 Git、未装依赖、未用 GPU、未调用付费 API、未碰 `data/staged/library_v7` 与 `/test/Finagent/runs/lib7/`（那两处正有别的评测在跑）。

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
| 2026-10-15 | T13 报表绑定 v6 | 已验收 | 复核：比亚迪合并报表有单位事实 385→995、迈瑞 1,090→1,374；新增绑定 2,517 条无丢失，40 条人工核对全对；未受影响发行人元组完全相同；177 项测试通过。决定：格力不重解析（同一 MinerU 仍会挤列），改为 T14 的文本层报表兜底；比亚迪 2023“附注单位向前生效”不做（策略风险大于收益）。 |
| 2026-10-15 | T14 文本层报表兜底 v7 | 已验收 | 复核：格力三份年报合并报表有单位事实 0→308／306／302，全库只在这 3 份文件新增 1,603 条，v6 事实无一变化；40 条人工核对全对；185 项测试通过。母公司报表兜底事实 `scope=None`：我核对了 `Checker.figures()`，合并口径要求 `scope==合并`、母公司口径要求 `scope==母公司`，`None` 两边都不会被取用，不会造成 E4 混用，保留。 |
| 2026-10-15 | T15 “总”字近义误配 | 已验收 | 复核：规则收紧为“去掉‘总’后完全相同才判不匹配”的决定正确（避免误伤利润总额／利润）；开发集 0 改判、训练集 3、验收集 LoRA 10，方向均为交给人工；验收集 LoRA 非舍入放行 1→0；186 项测试通过。 |
| 2026-10-15 | T16 复查 Agent 降本 | 已验收 | 复核：复查输入 token −47%、花费 −43%（每篇 0.060→0.035 美元）；判出 93.4%→95.4%、确认不变、非舍入放行 0、误报 11→14（新增均为舍入级）；188 项测试通过。HTTP 405 已由 Claude 在 `ApiClient` 中加入重试，不重跑该篇。architecture.md 由 Claude 更新。 |
| 2026-10-15 | T17 附注声明的报表单位 v8 | 已验收 | 复核：新获单位 412 条全在比亚迪 2023，v7 已有单位的 459,848 条元组不变；跨年核对 77/80 一致，3 处为已披露重述；20 条人工核对值／列／单位全对；191 项测试通过。续表页 180 条无表题仍无单位，记为遗留。 |
| 2026-10-16 | T18 A1 按节点指定模型 | 已验收 | 复核 diff：不给 Agent 模型时复用同一 `ApiChatModel`，请求体与缓存不变；我用新代码按 T16 缓存零花费重跑开发集，`summary` 除新增 `cache`／`cost_usd_by_model` 外逐项相同（唯一差异为 T16 那篇 405 失败的稿件无缓存、仍失败）；204 项测试通过。分缓存后预算各自起算的说明接受。 |
| 2026-10-16 | T18 A2 本地服务冒烟 | 已验收（作管线冒烟） | 复核缓存：37 个 Agent 请求均为 `qwen3.5-9b`（修订也用基座，回报中“抽取模型做的修订”系笔误），工具调用全部解析；最终回合温度 0 下反复重发 JSON 至 8,000 token 截断。截断前第一版数组对照金标：14 个真错误标出 12 个、1 个正确数字误标（Agent 只能加报警，确认数不受影响）。管线可用，失败属模型行为，交 A3 判据衡量，不调解码。21 token/秒为单请求串行速率，A3 改为 8 片并行。 |
