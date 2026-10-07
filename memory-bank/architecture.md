# 当前架构

## M0 起点：已有推理探针（2026-10-06）

当前只有单进程文本推理可行性入口；方案中的数据接入、训练、Agent、Harness 和服务尚未实现。

| 文件或目录 | 作用 |
| --- | --- |
| `AGENTS.md` / `CLAUDE.md` | 协作规则及其引用入口。 |
| `Qwen3.5金融研究Agent与Harness方案.md` | 完整设计依据；拟定能力不代表已交付。 |
| `promote.md` | 人工协作提示词，不是执行入口。 |
| `requirements.txt` | 当前 Python 依赖版本；虚拟环境复用系统 Torch。 |
| `configs/m0_inference.yaml` | 模型 revision、资产路径、三种精度、生成和重复预算。 |
| `configs/m0_smoke_tasks.jsonl` | 12 个固定文本/工具调用探针及预期；并非金融基准数据集。 |
| `configs/m0_smoke_tools.json` | 搜索、计算、澄清的参数声明；本阶段不执行业务工具。 |
| `src/evaluation/m0_inference_check.py` | 加载一个精度变体，套用本地模板，生成、解析、评分及测量资源。 |
| `memory-bank/plan.md` | M0—M6 全生命周期及进入各步骤的条件。 |
| `memory-bank/progress.md` | 当前状态与阻塞项。 |
| `memory-bank/result.md` | 用户指定的实验记录文件；后续追加，不覆盖历史。 |
| `.venv/` | 本地运行环境，不是可提交源码。 |
| `Qwen3.5-9B/` | 未完成的另一份下载；含 `.incomplete` 权重，不是配置使用的模型。 |

数据流：YAML 配置 + JSONL 任务 + JSON 工具声明 → 本地 tokenizer/chat template → GPU 模型生成 → 工具参数解析和任务评分 → `records.jsonl`、`summary.json`。

正式模型位于 `/test/Finagent/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a`；历史结果位于 `/test/Finagent/artifacts/m0_step1/<variant>/<UTC时间>/`。这些是工作目录外资产；模型和旧结果均保留原位。每种精度独立进程、单卡串行运行；当前没有训练 checkpoint 或任务恢复服务。

注意：文本任务仍加载包含视觉部分的完整多模态模型。工具回传用例输入的是固定示例观察，不能作为真实工具闭环验收。当前探针按生成结果是否提前结束计量，不保证满上下文加满输出预算的峰值资源。

## M0 第 1 步：验收边界补齐（2026-10-06）

保留上述架构起点，本次新增/修改职责如下：

- `src/evaluation/m0_inference_check.py`：拒绝重复/残缺参数、非数值或非有限数值数组及多余工具调用；任务通过要求解析无错误且正常结束。`repeats_identical()` 要求每题的所有重复均存在且一致。新增 `summary.passed` 控制退出码，并记录源码、任务和工具声明哈希。
- `tests/test_m0_inference_check.py`：12 项 CPU 回归测试，覆盖解析、重复完整性，以及入口对错误答案/解析失败/截断输出的拒绝；模拟生成仅验证执行逻辑，真实模型证据单列在结果记录。
- 停止集合沿用 tokenizer 的 EOS `248046`（`<|im_end|>`）与 PAD `248044`（`<|endoftext|>`）；模型配置 EOS 为后者。二者不同不等于不兼容，不能以强行统一 ID 的方式“修复”。嵌入与输出词表均为 `248320 × 4096`，覆盖 tokenizer 的 248077 个 token。

本次不新增生产工具、训练或 Agent 入口。文本答案仍采用现有字符串包含式探针评分，只证明这些短任务的可行性，不替代严格金融基准评分。

## 版本管理边界（2026-10-06）

`.gitignore` 排除虚拟环境、Python 缓存、模型下载、运行产物、checkpoint、数据目录和本地环境变量文件。Git 管理源码、固定配置、测试与项目文档；实际模型和实验产物继续保存在原路径。

## M0 第 2 步：来源关联审计（2026-10-06）

- `configs/tatqa_sources.json`：固定两个数据版本、下载地址、许可入口和 JSON 的 SHA256；TAT-DQA 四个 JSON 已与官方 Drive 文件逐字节核对一致。
- `src/data/tatqa_source_audit.py`：读取 train/dev/公开 test，验证输入哈希，再按整组问题（仅统一大小写和空白、忽略顺序）精确关联 `doc.source`；保留歧义、未匹配项及跨官方划分报告。不使用答案，不建立检索索引或分配训练划分。
- `tests/test_tatqa_source_audit.py`：覆盖单题误匹配、来源歧义、重复 ID、跨划分重叠和答案独立性等 7 项回归。
- `data/raw/{tatqa,tatdqa}/`：原始 JSON，包含评分标签，禁止作为检索材料整体导入。`data/audits/tatqa_sources_v3/` 保存逐上下文映射和审计汇总，均为本地资产。

关联链为 TAT-QA 上下文 → 完整问题组 → TAT-DQA 文档 ID/来源文件名。`doc.page` 是 TAT-DQA 文档相关起始页字段，尚未核验原财报 PDF 页码；不将其直接宣称为原财报页码。恢复数据集来源文件名也不等于已经取得原财报 URL 或披露时间。

### 数据接入初稿

`src/data/build_manifest.py` 读取 FinQA、TAT-QA 和 ConvFinQA 报告元数据。`filename` 按公司代码/报告年归组，联合两套同源数据以 test > dev > train 确定候选报告归属；保留原官方划分并标记排除原因。2019 年 FinQA 公司代码尚未与 TAT 公司全名对齐，相关样本暂隔离。

输出 `data/staged/m0_v1/{evidence,questions,gold,manifest}.jsonl`：前者仅含表格和原文白名单；问题只引用证据 ID；评分标签独立存储；manifest 保留源报告、原划分和候选去向。当前所有文件是离线接入资产，没有模型检索入口，不能将整个目录作为检索资料。`tests/test_build_manifest.py` 覆盖报告归组、非法来源、留出优先级和 gold 隔离。

## M0 验收后的运行入口

- `src/data/leakage.py`：去除无关 ID 后生成可见文本，采用 5-token shingles、Jaccard ≥ 0.9 的精确前缀连接寻找近重复；跨报告命中的报告整体隔离。
- `configs/m0_data_lock.json` / `src/data/fetch_sources.py`：固定各原始文件哈希和 revision，按需下载并验证，已有文件哈希不符时报错。
- `src/data/build_manifest.py`：冻结 `official_filtered_v1` 与 `report_group_v1` 两个协议；前者是保留官方归属的去泄漏子集，不冒称完整官方成绩，后者将源报告整体分配到最高留出优先级。严格按可见上下文＋问题关联 TAT 测试标签，无法关联者不可评分。
- `src/data/validate_manifest.py`：独立核对全部记录、引用、证据字段白名单及保留集的来源/近重复交集；写入绑定文件哈希的 `validation.json`。
- `configs/m0_training.json` / `src/training/m0_lora_check.py`：固定 100 条完整训练样本，最长 1024；仅语言模块 LoRA 可训练。以官方模板的字符偏移建立 assistant loss mask。数据必须具备有效验收回执且哈希一致。
- `data/staged/m0_frozen_v1/` 是验收后的本地数据快照；`artifacts/m0_lora_v1/` 保存训练轨迹、模块枚举、完整状态 checkpoint 与重载结果。

训练 effective batch = 1 × 8 × 1 = 8。基础权重 BF16，LoRA 与 AdamW 状态 FP32，rank 8、alpha 16、dropout 0，学习率 1e-4 线性降至零。checkpoint 包含适配器、优化器、调度器、步数、三类 RNG、确定性样本游标、配置和输入指纹；`latest` 指向完成保存的目录。本轮保持第 10 步和第 20 步两份有效 checkpoint。

## M1：证据、计算与固定流程

保留前述 M0 演进记录；当前新增链路为：可见证据 → SQLite FTS5/BM25 → 数值跨度抽取 → 模型选择事实及运算 → 受控执行 → 带引用的结果记录。

- `src/evidence/models.py`：保存 Evidence、Fact 及原文位置；业务事实必须具备已核验的指标、单位、期间和口径，数值须与引用原文一致。候选事实不能直接用于业务计算。
- `src/tools/calculator.py`：28 位 Decimal、HALF_EVEN，按白名单执行业务计算；转换数值尺度，拒绝不兼容口径及零分母，负基数增长改报差额并解释。
- `src/tools/benchmark_tools.py`：为表格单元格/文本数值登记事实编号和字符位置，仅允许引用事实、先前结果和有限常量。FinQA 基准语义单独处理，不套用业务增长规则。
- `src/evaluation/vendor/finqa_execution.py` / `FinQA_LICENSE`：固定 revision 的官方执行评分函数及 MIT 许可；通过上述预算受限包装调用，未引入任意代码执行。
- `src/retrieval/bm25.py`：仅接受可见证据字段，拒绝 gold 混入；FTS5 索引提供 CPU BM25 查询。
- `src/model/financial.py`：共享原生 `calculate` 工具声明、证据提示和推理解析。完整输入超预算、截断、非法参数均计失败，不默默裁剪。
- `src/evaluation/m1_baseline.py`：冻结 40 个不同开发报告任务，记录初版 JSON 接口对照及独立检索指标。
- `src/evaluation/financial_run.py`：在相同任务上比较直接回答、给定证据工具计算、检索后工具计算；保存逐题输出、耗时、显存和来源选择。
- `src/evaluation/validate_financial_run.py`：独立核对任务完整性、评分与 gold 一致性，并逐条解析计算引用到原始字符跨度。
- `tests/test_calculator.py`、`test_benchmark_tools.py`、`test_financial_model.py`：分别覆盖业务口径、基准运算/引用和原生工具边界。

两套计算契约不可混用：数据集抽取的数值跨度不是已经核验单位和口径的业务 Fact。引用可回查只证明原文数值存在，不自动证明模型选对了指标/期间。当前固定流程尚不具备动态 Agent、持久化任务恢复或生产服务能力。

## M2：配对监督、恢复与采用门槛

- `src/training/prepare_sft.py`：最多转换 500 条允许训练的参考程序；拒绝不唯一数值绑定、未支持操作和完整轨迹超预算。保存动作、实际工具观察、两种 assistant mask 和全部排除原因。
- `configs/m2_{answer,action}_training.json`：固定同一组 94 个问题、40 次更新及 2048 上限，绑定准备样本哈希。`src/training/m0_lora_check.py` 复用已有训练/恢复路径，并验证准备样本只含允许训练 ID；原 M0 分支保持原行为。
- `src/training/loss.py`：保留完整上下文和 BF16 输出投影，只对未屏蔽的因果目标计算 FP32 交叉熵；不截断证据、不删除前缀梯度路径。`verify_loss.py` 对照实际模型损失及输出层传入解码器的梯度，并单列原实现自身的运行波动。
- `src/evaluation/regression.py` / `configs/m2_visual_regression.json`：冻结通用工具与三个合成页面探针；检查适配器回归，不能作为真实财报视觉质量证明。
- `src/evaluation/select_adapter.py` / `configs/m2_selection.json`：统一应用配对质量、能力回归和时延门槛；未通过时保留基座。选择是小规模开发决策，不是完整基准或生产质量保证。
- `src/evidence/identity.py` / `src/evaluation/audit_citations.py`：仅在报告及完整结构化原文均相同时承认证据 ID 别名；新审计不覆盖历史生成或数值成绩。
- `tests/test_prepare_sft.py`、`test_training_loss.py`、`test_evidence_identity.py`：验证程序拒绝条件、loss/梯度等价和别名边界。
- `data/staged/m2_sft_v1/` 保存本地监督数据；`artifacts/m2_{action_v2,answer_v1}/` 保存 checkpoint；`artifacts/m2_provenance/` 保存源码快照、实际训练 token 暴露量及选择依据。失败动作试跑和未通过的等价性记录继续保留。

M2 采用决定入口为 `configs/model_choice.json`：后续使用动作 LoRA，基座权重和其他适配器不覆盖。独立来源别名审计为引用支持的有效口径；旧 `correct_evidence` 字段仅表示 ID 精确匹配，不再解释为来源真实性。

## M3：单 Agent 与持久化执行

- `src/harness/store.py`：SQLite 原子保存请求摘要、任务修订、事件、报告和 TTL 缓存；同一请求重试不重复提交，输入改变必须使用新请求。公司、期间、截止日或快照变化时清空旧事实与历史。
- `src/data/corpus.py`：按公司、报告期间和快照限定资料；同报告且结构化原文完全相同的证据仅索引一次。当前基准资料缺乏核实的披露日期，拒绝严格 PIT 查询。
- `src/agent/runner.py`：动态检索、读取、计算、澄清和提交循环；先持久化预算，再调用模型，取消/失败也计调用成本。澄清真实挂起，报告只允许引用当前范围内已执行的计算。
- `src/agent/fixed_runner.py`：默认固定取证计算流程，共用持久化状态、范围过滤和取消；每轮一次模型生成，保存生成结果后可恢复确定性提交。`src/model/financial.decode_prediction()` 复用原有解析与受控执行，不重复调用模型。
- `src/model/agent.py`：模型生命周期与跨工作目录的单卡进程互斥（`/tmp/finagent-cuda0.lock`），生成取消传至停止条件；OOM 后须健康检查，关闭时释放资源。`m0_inference_check.generate()` 增加可选停止回调，旧调用默认行为不变。
- `src/evaluation/freeze_acceptance.py` / `configs/m3_acceptance_lock.json`：冻结 80 个任务、115 轮及来源隔离协议，gold 与检索库分开。
- `src/evaluation/agent_run.py`：执行冻结业务验收，按轮保存恢复游标；30 个单题另跑同适配器固定流程对照。`validate_agent_run.py` 独立检查完整性、原文引用、数值和报告唯一性。
- `src/evaluation/fault_smoke.py`：实际 GPU 取消、OOM、恢复后推理与进程崩溃验收；`tests/test_store.py`、`test_runner.py` 覆盖状态和工具拒绝边界。

任务输入 → 持久化队列 → 单模型循环 → 受范围约束的工具 → 原子报告。执行状态与业务状态分别保存；`awaiting_input` 没有最终报告。当前模型仍操作 FinQA 数值候选，不等同于已核验财务口径的业务 Fact。运行源码快照和记录位于 `artifacts/m3_acceptance_run_v1/`，故障证据在 `artifacts/m3_fault_v1/`。

当前默认采用 `FixedRunner`，动态 `Runner` 仅供实验。`tests/test_fixed_runner.py` 覆盖固定流程提交、澄清、取消和生成后恢复；`tests/test_model_lock.py` 用另一工作目录的子进程验证全局锁。任务与报告绑定资料哈希及请求范围，恢复时拒绝同名快照下的资料变化。阶段质量与采用依据仅在 [result.md](result.md) 维护。


## M4：原始 PDF、检索和披露约束

- `configs/report_sources.json` 固定发行人 URL、文件 SHA、披露日期、入库时间、报告期间及版本关系；`src/data/reports.py` 按 PDFium 原始页提取文本，空页报错，不静默遗漏。合法资料在 `data/staged/reports_v2/`，快照标识为 `reports-v1`；旧错误目录保留拒绝说明。
- `src/data/report_corpus.py` 将日期、版本和快照约束应用到检索与读取；缺少核实披露日期的资料不参与严格截止查询。取数仍经过 M3 的范围、缓存、状态和报告约束。
- `src/retrieval/report_index.py` 提供中文双字词/英文词 BM25、稀疏 TF-IDF 融合及词面重排；截止范围内单独建立统计，未来文本不能通过 IDF 影响过去排名。这不是神经向量模型。
- `configs/m4_retrieval_tasks.json` 固定 12 个查询及页面相关标签；`retrieval_run.py` 比较排名与成本，`retrieval_answers.py` 固定模型核验生成引用，`pit_run.py` 检查实际披露边界。采用入口为 `configs/report_retrieval_choice.json`。
- `tests/test_report_corpus.py` 覆盖版本替换、未知日期、资料哈希、未来 IDF 隔离及完整固定流程的截止引用；真实披露回放与合成修订测试分开保存。

披露日期采用包含当日的日粒度；入库日期不能代替披露日期。年度报告章节无可核实发布时间时仍可用于非严格查询，但不能进入严格 PIT 上下文。原文资料仅本地研究使用，Git 记录来源及哈希，不重新分发 PDF。


## M5：原页多模态与独立中文验证

- `src/data/prepare_visual.py` 从官方 TAT-DQA 测试档案连接 PDF、OCR、区域标注与来源分组，固定 12 个允许留出来源。公开 PNG 是缩略图，实际输入由原 PDF 按 OCR 页面尺度渲染；保存图像/PDF 哈希和原页关联，不能把裁选 PDF 页号冒充原财报页号。
- `configs/m5_protocol.json` 冻结输入/输出预算、像素上限、三路对照及控制实验；`m5_label_audit.json` 保留推理前发现的原标签单位问题，主成绩不改原 gold，原文复核成绩单列。
- `src/evaluation/visual_run.py` 执行文本、视觉、按需结合及遮挡/基础单位控制，记录视觉 token、原始输出、耗时和显存；`numeric_score.py` 只实现固定标量子集的 TAT 数值/尺度语义，不声称完整官方 F1。
- `configs/m5_chinese_tasks.json` 与 `src/data/prepare_chinese.py` 绑定中国移动原文数字、指标、单位与页码，用既有 Decimal 业务工具产生参考计算；`chinese_run.py` 单列数值、单位和格式错误。
- `src/evaluation/validate_visual_run.py` 从原始生成重新评分，并核对输入哈希、配对完整性；`tests/test_numeric_score.py` 覆盖百分比、单位换算与严格标量解析。

有效英文输入为 `data/staged/m5_visual_v3/`，中文输入为 `m5_chinese_v1/`；前两版英文准备失败的拒绝说明保留。gold 与可见材料物理分开；运行前源码/依赖在 `artifacts/m5_provenance/`。本阶段是局部页面验证，未微调视觉参数，未实现或声称区域定位、完整文档理解或中文自主 Agent。


## M6：本地服务与可复查展示

- `src/service/server.py` 提供回环地址 HTTP API 和一个持久化队列 worker；模型、检索索引在所属线程创建/释放。默认使用 `FixedRunner`；报告库模式复用 M4 的范围过滤。SQLite 绑定资料哈希、快照和流程，换资料拒绝误用旧数据库。
- `src/service/index.html` 定义中文页面；`app.js` 管理稳定请求 ID、事件游标、刷新恢复、澄清新修订、取消、来源查看和报告导出；`style.css` 提供桌面/移动布局。所有资料文本使用安全文本节点显示，不能执行原文脚本。
- `tests/test_service.py` 验证 HTTP 契约、来源权限、报告版本和数据库快照；`src/evaluation/browser_smoke.py` 在真实 Chromium 上测试交互，使用合成状态夹具，不假冒模型质量测试。
- `configs/m6_demo.json` 固定已知来源的展示题与追问；`src/evaluation/deployment_smoke.py` 在新进程加载实际模型，通过 HTTP 验证计算、取消、导出和崩溃后恢复，只终止自身启动的服务进程。
- `README.md` 仅提供项目文档导航；`requirements-dev.txt` 是可选浏览器测试依赖，运行恢复、入口和外部资产边界集中在 [environment.md](environment.md)。

默认入口 `.venv/bin/python -m src.service.server` → loopback API → SQLite 请求/队列 → 单 GPU 固定流程 → 原子报告/事件 → 网页和 JSON 导出。报告保留原始计算尺度，不能把基准比例自动显示为已核验百分比。网页状态来自实际事件；等待澄清、取消或失败都不显示为已回答。动态 Agent 与 M5 视觉仍为离线实验入口。
