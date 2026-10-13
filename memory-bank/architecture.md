# 当前架构

更新：2026-10-13。本文件只描述仓库中现有的代码与资产；目标形态与阶段见 [plan.md](plan.md)，任务合同见 [task_definition.md](task_definition.md)。

## 现状一句话

仓库目前是一套“取证 → 受控计算 → 可持久化执行”的基础设施，加上刚接入的表格位置映射。面向核查任务的断言抽取、核查流程、多智能体编排和评测集都还没有实现。

## 数据流（现有部分）

原始 PDF → 分页文本与来源元数据（哈希、披露日、版本）→ 按公司／期间／截止日过滤的资料库 → 检索（BM25、向量、融合、重排）→ 数值登记为带原文字符位置的事实 → 模型选择事实与运算 → 受控计算 → 结果连同引用原子写入 SQLite。

## 模块

### 证据与计算

- `src/evidence/models.py`：`Evidence`（原文、页码、表格、行、列、位置框）与 `Fact`（指标、单位、币种、量级、期间、口径）；业务事实须已核验元数据且数值与所引原文一致。
- `src/evidence/identity.py`：仅在报告与结构化原文完全相同时承认两个证据编号为同一来源。
- `src/evidence/compact.py`：保留原坐标的证据裁剪视图（供样本准备使用）。
- `src/tools/calculator.py`：28 位 Decimal、HALF_EVEN 的白名单业务计算；换算量级，拒绝期间、单位、币种、口径不兼容及零分母，负基数增长改报差额。
- `src/tools/benchmark_tools.py`：为表格单元格与正文数值登记事实编号和字符位置；按步骤执行只引用事实、先前结果与有限常量的算式；`explain_calculation_error` 只生成给模型看的报错说明。底层算术语义来自 `src/evaluation/vendor/finqa_execution.py`（附 MIT 许可）。

### 资料入库与检索

- `src/data/reports.py`、`src/data/expand_reports.py`：按配置下载并校验原始 PDF，用 PDFium 逐页提取文本，空页报错；记录来源、文件哈希、披露日、报告期与版本关系。
- `src/data/table_map.py`：把版面解析器（MinerU）给出的 HTML 表格中的数值单元格，按位置与数字逐一对回 PDFium 页面文本的字符区间和位置框。PDF 文本层为准：数字不一致的单元格与解析器漏掉的数字分别报告，不做修正。
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

### 训练

- `src/training/m0_lora_check.py`：LoRA 训练入口，含完整状态检查点（适配器、优化器、调度器、随机数、样本游标）与恢复校验；`src/training/loss.py` 只对受监督位置计算 FP32 交叉熵；`src/training/preference_loss.py` 为 sigmoid DPO。
- `src/training/distributed.py`、`src/training/deepspeed_lora.py`：多进程梯度累积（`global_batch = micro_batch × grad_accum ÷ world_size × world_size`，不能整除时报错）、FSDP 冻结权重分片、DeepSpeed ZeRO-3。
- **待重写**：`prepare_sft.py`、`autonomous_sft.py`、`complex_autonomous.py`、`teacher.py`、`preference_data.py` 及其依赖的 `src/data/build_manifest.py`、`src/data/tatqa_source_audit.py`、`src/evaluation/validate_agent_run.py`。它们按英文财报问答数据的格式准备样本，训练入口的数据装载仍引用它们；阶段 4 按核查任务的数据重写后清理。

### 评测工具

- `src/evaluation/statistics.py`、`src/evaluation/compare_agent_runs.py`：按发行人聚类的配对 bootstrap 区间。
- `src/evaluation/serving_benchmark.py`：TTFT、端到端时延与吞吐测量。

## 边界与注意

- 两套计算合同不可混用：数值跨度登记得到的是“原文里有这个数”，不等于已核验指标、单位、期间和口径的业务 `Fact`。
- 引用可回查只证明原文存在该数值，不证明选对了指标或期间。
- 服务配置 `configs/model_choice.json` 指向的适配器权重已不在本机，本地服务在新模型就绪前不能启动；测试使用替身模型，不受影响。
- 模型、数据、索引和运行产物不进 Git，位置见 [environment.md](environment.md)。
