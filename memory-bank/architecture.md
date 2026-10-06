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
