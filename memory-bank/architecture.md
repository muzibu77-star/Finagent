# 当前架构

更新：2026-10-15。本文件只描述仓库中现有的代码与资产；目标形态与阶段见 [plan.md](plan.md)。

## 现状一句话

阶段 1 的出处库已建成：原始披露文件 → 版面解析 → 与 PDF 文本层逐一对照的表格事实 → 可按数字和截止日查询的资料库。阶段 2 的评测集已冻结、阶段 3 的免训练核查基线可用（见“核查”）；阶段 4 训练出可本地部署的 Qwen3.5-9B LoRA 抽取模型；阶段 5 的 LangGraph 核查流程、复查 Agent、工作底稿、审核页面与 MCP 工具已完成（见“编排与底稿”）；阶段 6 验收已按写定协议完成。旧英文财报问答路线（执行与服务、检索、模型驱动、样本准备）已于 2026-10-14 删除，训练只保留核查任务的 LoRA 入口。

**出处库的数据流**：`cninfo_reports`（年报与更正公告）、`policy_sources`（政策与统计发布）下载并记录披露日与哈希 → `parse_reports` 调 MinerU 得到版面与表格 → `table_map` 把表格数字对回 PDF 文本层的字符位置 → `table_facts` 绑定行标签、列表头、单位等 → `library` 写入 SQLite 并提供查询。

## 模块

### 证据与计算

- `src/evidence/models.py`：`Evidence`（原文、页码、表格、行、列、位置框）与 `Fact`（指标、单位、币种、量级、期间、口径）；业务事实须已核验元数据且数值与所引原文一致。
- `src/tools/calculator.py`：28 位 Decimal、HALF_EVEN 的白名单业务计算；换算量级，拒绝期间、单位、币种、口径不兼容及零分母，负基数增长改报差额。

### 资料入库

- `src/data/cninfo_reports.py`：按 `configs/cninfo_issuers.json` 的名单查询巨潮资讯公告索引，下载年度报告与更正公告，记录标题、网址、披露日与哈希到 `configs/cninfo_sources.json`；缺失的财年、失败与被跳过的公告都写入收据。
- `src/data/policy_sources.py`：按 `configs/policy_feeds.json` 收集带发布日期的政策文件与统计发布；支持三种列表（新华网活动数据文件、政府网政策数据文件、可配置正则与翻页规则的 HTML 列表）和证监会栏目接口；保存原始网页与按容器标记提取的正文，找不到正文记为失败。
- `src/data/parse_reports.py`：按下载收据逐份调用 MinerU 解析，校验来源哈希，结果一份一个文件、原子落盘，可断点续跑，失败单独记录；要求已配置 GPU 上的文档模型服务，否则拒绝启动。
- `src/data/table_map.py`：把版面解析器（MinerU）给出的 HTML 表格中的数值单元格，按位置与数字逐一对回 PDFium 页面文本的字符区间和位置框。PDF 文本层为准：数字不一致的单元格与解析器漏掉的数字分别报告，不做修正。
- `src/data/table_facts.py`：在映射之上，把每个数字与行标签、列表头、小节、表题、报表日期、单位、币种和合并／母公司口径绑定为候选事实；续表继承上一段的表头与单位。缺单位、续表缺开头、并格与漏识别记入问题清单，缺失字段留空而不猜测。命令行入口 `python -m src.data.table_facts --pdf --middle --output`。
- `src/data/library.py`：出处库。`build` 把年报的逐页文本（与 PDFium 返回的完全一致，事实的字符区间以它为准）、表格事实、绑定问题以及政策与统计发布写入一个 SQLite 文件，并核对来源哈希；`--restatement` 可加入人工转写复核过的历史更正案例（原报告页与更正值，后者无位置框）；`Library` 提供只读查询：`find_number`（按书写的数字精确查找事实）、`search_pages`（中文二元词 BM25）与 `page_text`，均可限定发行人和截止日（披露日为日粒度、包含当日）。

### 核查

- `src/verify/checker.py`：确定性核查器。输入结构化断言，按写稿日期限定可用文件、以更正公告覆盖旧数，在出处库中取对应指标与年份的合并口径数值，比较后给出一致／不一致／无依据／存疑，并对不一致判断错因。`src/evaluation/run_checker.py` 在冻结断言集上运行并汇总。
- `src/verify/extract.py`：用模型把点评中的数字抽成结构化断言，科目必须从该年报合并报表的科目清单中选择；`src/verify/pipeline.py` 的 `judge()` 把抽取结果逐条变成核查结论（含证据事实编号），是评测与编排共用的唯一实现；`src/evaluation/run_reviews.py` 串起抽取与核查并按核实标签计分，`--direct` 为“模型读原文直接审稿”的对照组。

### 编排与底稿

- `src/agent/llm.py`：LangChain `BaseChatModel` 适配器，内部沿用 `ApiClient` 的缓存、计价、预算与本地模型规则（DeepSeek API 或本机 vLLM 均可）。
- `src/agent/graph.py`：LangGraph 五节点流程 extract（抽取 Agent，只看稿件与科目清单）→ check（确定性核查器）→ investigate（复查 Agent：对核查器交给人工的数字用工具检索与计算，**只能把数字标为不一致、不能判一致**，判“正确”只作提示）→ consistency（稿内一致性，代码；排除季度／分部措辞，按措辞分组）→ revise（修订 Agent：只为核查器判不一致的数字写改写句，代码核对除该数字外逐字不变，否则标“需人工改写”）→ report（工作底稿：每个数字的结论、来源〔核查器／Agent〕、错因、应有值、出处文件页码与字符区间、修改建议或 Agent 依据、模型节点的错误；另有 Markdown 版）。模型节点失败不抛出：抽取失败时全部数字交给人，其他节点失败只记 `errors`。入口 `review_draft()` 与 `python -m src.agent.graph`。稿内一致性节点曾实现，因真实稿件上 0／19 正确于 2026-10-15 移除。
- `src/agent/single.py`：ReAct 工具 Agent（`create_react_agent`），四个受限工具：`find_number`、`search_pages`（限定发行人与写稿截止日）、`page_text`（拒绝截止日后披露的文件）、`calculate`（AST＋Decimal，无 `eval`）。既是“单 Agent 直接审稿”对照组，也是 investigate 节点的实现。
- `src/service/app.py`＋`index.html`：本机审核页面（127.0.0.1:18100，同一时间只跑一篇，参数校验返回 400，底稿 JSON／Markdown 下载）。`src/service/mcp_server.py`：stdio MCP 服务，工具 `check_draft`、`find_number`、`search_pages`、`page_text`。

### 训练

- `src/training/m0_lora_check.py`：LoRA 训练入口，含完整状态检查点（适配器、优化器、调度器、随机数、样本游标）与恢复校验。数据装载只读 `chat_samples`（`{review_id, messages}` 的 JSONL，只监督最后一个 assistant 回合，样本必须来自训练集发行人）；`src/training/chat_encoding.py` 提供只监督 assistant 跨度的对话模板编码；`src/training/loss.py` 只对受监督位置计算 FP32 交叉熵。基座模型配置见 `configs/base_model.yaml`。
- `src/training/distributed.py`、`src/training/deepspeed_lora.py`：多进程梯度累积（`global_batch = micro_batch × grad_accum ÷ world_size × world_size`，不能整除时报错）、FSDP 冻结权重分片、DeepSpeed ZeRO-3。
- `src/training/extraction_samples.py`：从核实过的点评标签构造阶段 4 的抽取训练样本（目标来自 Codex 核实标签，不用模型输出）。
- `src/training/merge_adapter.py`：在 CPU 上把 LoRA 适配器合并进基座并保存完整权重（评测实际用 vLLM 原生 LoRA，合并为备用）。
- 损失：只对受监督位置分块投影词表并逐块重算（`loss.py`），多进程时各进程块数取最大值、以空块补齐，使 FSDP 分片的输出层集合通信对齐。

### 评测工具

- `src/model/api_client.py`：计量的 API 客户端。应答按请求哈希缓存到文件（重跑不重复计费），按高峰价累计花费，预算用尽后拒绝新请求；密钥只从被忽略的 `.env` 读取。
- `src/evaluation/draft_pilot.py`：试点入口。让模型按年报合并报表原文（或不给原文）写点评，再由代码检查文中每个金额与百分比能否在该年报的事实中找到依据。
- `src/evaluation/splits.py`：按行业交错把 30 家发行人划为训练、开发、验收三部分并写入 `configs/issuer_splits.json`；文件已存在时拒绝覆盖。
- `src/evaluation/freeze_set.py`：按发行人划分冻结评测集：训练保留全部种子，开发与验收每类错误取固定条数；模型稿件整篇保留并附代码判定的弱标签；写出带哈希的 `lock.json`，目录已存在时拒绝写入。
- `src/evaluation/run_graph.py`：在稿件集上逐篇运行完整流程并按核实标签计分（含 Agent 来源的真值拆分）；`src/evaluation/run_single.py`：单 Agent 对照组的同口径计分；`src/evaluation/paired_bootstrap.py`：两次运行按点评篇配对重抽的差值区间。
- `src/evaluation/review_labels.py`：逐数字人工核实的离线辅助模块，不自动选择指标或推断金标。按点评精度以 Decimal 比较、记录推导算式和每个操作数的事实编号或原页字符区间，验证覆盖、出处和算式一致性；支持明确的上下限断言与冻结抽取遗漏中文千分位的完整原句证据。核实结果在 `data/staged/eval_v2_labels/`，与冻结输入和出处库分开；`summary.json` 明示尚未完成的部分，只有完整首遍和固定种子 10% 独立盲审达标后才视为完成。用户另行批准的无文本层 PDF 页例外使用 `evidence/` 下截图及页码、像素坐标，不伪造字符区间。
- `src/evaluation/seed_claims.py`：评测集的种子。从出处库中合并三大报表、同一行同时有本年与上年数的事实，按模板生成正确断言（金额换算为亿元的直接引用、同比增速），再各做一次有记录的改动得到错误断言（E1 改数字、E2 改单位、E3 上年数当本年、E4 母公司数当合并数、E5 增速分母用错、E9 同一指标前后两个数不同），E8 注明不含该数字的旧年报为出处；另从国常会新闻稿生成日期断言及 E7 日期错误，从复核过的更正案例生成 E6（更正后仍引用旧值）。报表未绑定的发行人改用“主要会计数据”表；每条带数值在句中的位置、所依据的事实编号和应有的值。句子是模板，只检验核查逻辑，不代表真实稿件的写法。

## 边界与注意

- 两套计算合同不可混用：数值跨度登记得到的是“原文里有这个数”，不等于已核验指标、单位、期间和口径的业务 `Fact`。
- 引用可回查只证明原文存在该数值，不证明选对了指标或期间。
- 模型、数据、索引和运行产物不进 Git，位置见 [environment.md](environment.md)。
