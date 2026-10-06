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
