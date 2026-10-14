# 当前架构

更新：2026-10-13。本文件只描述仓库中现有的代码与资产；目标形态与阶段见 [plan.md](plan.md)。

## 现状一句话

阶段 1 的出处库已建成：原始披露文件 → 版面解析 → 与 PDF 文本层逐一对照的表格事实 → 可按数字和截止日查询的资料库。面向核查任务的断言抽取、核查流程、多智能体编排和评测集还没有实现；下文“执行与服务”“训练”各节是此前留下、尚未接到出处库上的基础模块。

**出处库的数据流**：`cninfo_reports`（年报与更正公告）、`policy_sources`（政策与统计发布）下载并记录披露日与哈希 → `parse_reports` 调 MinerU 得到版面与表格 → `table_map` 把表格数字对回 PDF 文本层的字符位置 → `table_facts` 绑定行标签、列表头、单位等 → `library` 写入 SQLite 并提供查询。

## 旧有模块的数据流

原始 PDF → 分页文本与来源元数据（哈希、披露日、版本）→ 按公司／期间／截止日过滤的资料库 → 检索（BM25、向量、融合、重排）→ 数值登记为带原文字符位置的事实 → 模型选择事实与运算 → 受控计算 → 结果连同引用原子写入 SQLite。

## 模块

### 证据与计算

- `src/evidence/models.py`：`Evidence`（原文、页码、表格、行、列、位置框）与 `Fact`（指标、单位、币种、量级、期间、口径）；业务事实须已核验元数据且数值与所引原文一致。
- `src/evidence/identity.py`：仅在报告与结构化原文完全相同时承认两个证据编号为同一来源。
- `src/evidence/compact.py`：保留原坐标的证据裁剪视图（供样本准备使用）。
- `src/tools/calculator.py`：28 位 Decimal、HALF_EVEN 的白名单业务计算；换算量级，拒绝期间、单位、币种、口径不兼容及零分母，负基数增长改报差额。
- `src/tools/benchmark_tools.py`：为表格单元格与正文数值登记事实编号和字符位置；按步骤执行只引用事实、先前结果与有限常量的算式；`explain_calculation_error` 只生成给模型看的报错说明。底层算术语义来自 `src/evaluation/vendor/finqa_execution.py`（附 MIT 许可）。

### 资料入库与检索

- `src/data/cninfo_reports.py`：按 `configs/cninfo_issuers.json` 的名单查询巨潮资讯公告索引，下载年度报告与更正公告，记录标题、网址、披露日与哈希到 `configs/cninfo_sources.json`；缺失的财年、失败与被跳过的公告都写入收据。
- `src/data/reports.py`：按配置下载并校验原始 PDF，用 PDFium 逐页提取文本，空页报错；记录来源、文件哈希、披露日、报告期与版本关系。
- `src/data/policy_sources.py`：按 `configs/policy_feeds.json` 收集带发布日期的政策文件与统计发布；支持三种列表（新华网活动数据文件、政府网政策数据文件、可配置正则与翻页规则的 HTML 列表）和证监会栏目接口；保存原始网页与按容器标记提取的正文，找不到正文记为失败。
- `src/data/parse_reports.py`：按下载收据逐份调用 MinerU 解析，校验来源哈希，结果一份一个文件、原子落盘，可断点续跑，失败单独记录；要求已配置 GPU 上的文档模型服务，否则拒绝启动。
- `src/data/table_map.py`：把版面解析器（MinerU）给出的 HTML 表格中的数值单元格，按位置与数字逐一对回 PDFium 页面文本的字符区间和位置框。PDF 文本层为准：数字不一致的单元格与解析器漏掉的数字分别报告，不做修正。
- `src/data/table_facts.py`：在映射之上，把每个数字与行标签、列表头、小节、表题、报表日期、单位、币种和合并／母公司口径绑定为候选事实；续表继承上一段的表头与单位。缺单位、续表缺开头、并格与漏识别记入问题清单，缺失字段留空而不猜测。命令行入口 `python -m src.data.table_facts --pdf --middle --output`。
- `src/data/library.py`：出处库。`build` 把年报的逐页文本（与 PDFium 返回的完全一致，事实的字符区间以它为准）、表格事实、绑定问题以及政策与统计发布写入一个 SQLite 文件，并核对来源哈希；`--restatement` 可加入人工转写复核过的历史更正案例（原报告页与更正值，后者无位置框）；`Library` 提供只读查询：`find_number`（按书写的数字精确查找事实）、`search_pages`（中文二元词 BM25）与 `page_text`，均可限定发行人和截止日（披露日为日粒度、包含当日）。
- `src/data/corpus.py`、`src/data/report_corpus.py`、`src/data/rag_corpus.py`：按公司、报告期、快照限定可读资料；披露日为日粒度并包含当日，缺少核实披露日的资料不进入严格截止查询；后续更正按日替换旧版本。
- `src/retrieval/bm25.py`、`src/retrieval/report_index.py`：SQLite FTS5 的 BM25、中文双字词／英文词检索与词面重排；截止范围内单独统计词频，未来文本不影响过去的排名。
- `src/retrieval/neural.py`：原字符跨度分块、BGE-M3 向量、本地 Qdrant、与 BM25 的 RRF 融合及交叉编码重排；权限过滤先于向量查询，索引身份绑定资料、配置与实现哈希。
- `src/data/leakage.py`：5-token shingles、Jaccard ≥ 0.9 的近重复检测，用于留出集隔离。
- `src/data/prepare_chinese.py`、`src/data/expand_chinese.py`：从中文原页构造带原值、单位、页码和哈希的算术题；`src/evaluation/restatement_run.py` 用康美药业、新纶科技的真实更正公告回放“截止日前后引用哪个版本”。

### 执行与服务

- `src/harness/store.py`：SQLite 原子保存请求、任务修订、事件、报告和 TTL 缓存；重复请求不重复提交；公司、期间、截止日或快照变化时清空旧事实；先记预算再调用模型，取消与失败同样计成本。
- `src/agent/fixed_runner.py`：固定的“取证 → 一次生成 → 受控计算 → 提交”流程，默认入口。`src/agent/runner.py`：动态的检索、读取、计算、澄清、提交循环，含可操作报错反馈等可选项，默认关闭。`src/agent/context.py`、`src/agent/inline.py`：对话上下文去重，以及在原文数值旁标注事实编号。
- `src/model/agent.py`：本地模型生命周期、单卡进程互斥锁、生成取消。`src/model/financial.py`：计算工具声明、证据提示与输出解析。`src/model/http_driver.py`：回环 vLLM 文本客户端。`src/evaluation/m0_inference_check.py`：生成与工具调用解析的公共函数。
- `src/service/server.py` 与 `index.html`／`app.js`／`style.css`：回环 HTTP 服务、单 worker 队列、事件重连、取消、来源查看与报告导出。`src/service/mcp_server.py`：以 stdio 暴露范围受限的搜索、读取、计算；`skills/financial-research/SKILL.md` 经 MCP prompt 加载。

### 核查

- `src/verify/checker.py`：确定性核查器。输入结构化断言，按写稿日期限定可用文件、以更正公告覆盖旧数，在出处库中取对应指标与年份的合并口径数值，比较后给出一致／不一致／无依据／存疑，并对不一致判断错因。`src/evaluation/run_checker.py` 在冻结断言集上运行并汇总。
- `src/verify/extract.py`：用模型把点评中的数字抽成结构化断言，科目必须从该年报合并报表的科目清单中选择；`src/evaluation/run_reviews.py` 串起抽取与核查并按核实标签计分，`--direct` 为“模型读原文直接审稿”的对照组。

### 训练

- `src/training/m0_lora_check.py`：LoRA 训练入口，含完整状态检查点（适配器、优化器、调度器、随机数、样本游标）与恢复校验；`src/training/loss.py` 只对受监督位置计算 FP32 交叉熵；`src/training/preference_loss.py` 为 sigmoid DPO。
- `src/training/distributed.py`、`src/training/deepspeed_lora.py`：多进程梯度累积（`global_batch = micro_batch × grad_accum ÷ world_size × world_size`，不能整除时报错）、FSDP 冻结权重分片、DeepSpeed ZeRO-3。
- **待重写**：`prepare_sft.py`、`autonomous_sft.py`、`complex_autonomous.py`、`teacher.py`、`preference_data.py` 及其依赖的 `src/data/build_manifest.py`、`src/data/tatqa_source_audit.py`、`src/evaluation/validate_agent_run.py`。它们按英文财报问答数据的格式准备样本，训练入口的数据装载仍引用它们；阶段 4 按核查任务的数据重写后清理。

### 评测工具

- `src/model/api_client.py`：计量的 API 客户端。应答按请求哈希缓存到文件（重跑不重复计费），按高峰价累计花费，预算用尽后拒绝新请求；密钥只从被忽略的 `.env` 读取。
- `src/evaluation/draft_pilot.py`：试点入口。让模型按年报合并报表原文（或不给原文）写点评，再由代码检查文中每个金额与百分比能否在该年报的事实中找到依据。
- `src/evaluation/splits.py`：按行业交错把 30 家发行人划为训练、开发、验收三部分并写入 `configs/issuer_splits.json`；文件已存在时拒绝覆盖。
- `src/evaluation/freeze_set.py`：按发行人划分冻结评测集：训练保留全部种子，开发与验收每类错误取固定条数；模型稿件整篇保留并附代码判定的弱标签；写出带哈希的 `lock.json`，目录已存在时拒绝写入。
- `src/evaluation/review_labels.py`：逐数字人工核实的离线辅助模块，不自动选择指标或推断金标。按点评精度以 Decimal 比较、记录推导算式和每个操作数的事实编号或原页字符区间，验证覆盖、出处和算式一致性；支持明确的上下限断言与冻结抽取遗漏中文千分位的完整原句证据。核实结果在 `data/staged/eval_v2_labels/`，与冻结输入和出处库分开；`summary.json` 明示尚未完成的部分，只有完整首遍和固定种子 10% 独立盲审达标后才视为完成。用户另行批准的无文本层 PDF 页例外使用 `evidence/` 下截图及页码、像素坐标，不伪造字符区间。
- `src/evaluation/seed_claims.py`：评测集的种子。从出处库中合并三大报表、同一行同时有本年与上年数的事实，按模板生成正确断言（金额换算为亿元的直接引用、同比增速），再各做一次有记录的改动得到错误断言（E1 改数字、E2 改单位、E3 上年数当本年、E4 母公司数当合并数、E5 增速分母用错、E9 同一指标前后两个数不同），E8 注明不含该数字的旧年报为出处；另从国常会新闻稿生成日期断言及 E7 日期错误，从复核过的更正案例生成 E6（更正后仍引用旧值）。报表未绑定的发行人改用“主要会计数据”表；每条带数值在句中的位置、所依据的事实编号和应有的值。句子是模板，只检验核查逻辑，不代表真实稿件的写法。

- `src/evaluation/statistics.py`、`src/evaluation/compare_agent_runs.py`：按发行人聚类的配对 bootstrap 区间。
- `src/evaluation/serving_benchmark.py`：TTFT、端到端时延与吞吐测量。

## 边界与注意

- 两套计算合同不可混用：数值跨度登记得到的是“原文里有这个数”，不等于已核验指标、单位、期间和口径的业务 `Fact`。
- 引用可回查只证明原文存在该数值，不证明选对了指标或期间。
- 服务配置 `configs/model_choice.json` 指向的适配器权重已不在本机，本地服务在新模型就绪前不能启动；测试使用替身模型，不受影响。
- 模型、数据、索引和运行产物不进 Git，位置见 [environment.md](environment.md)。
