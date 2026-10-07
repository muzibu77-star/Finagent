# 实验与验证记录

采用用户指定的单数文件名 `result.md`。下列记录追加维护，失败和被替代的结果不删除。

## 2026-10-06｜M0 第 1 步

**结论：BF16 固定短任务验收通过；INT8/INT4 未通过全部任务。** 三种配置均可加载，但不将能运行等同于业务任务通过。

环境：RTX 4090 24564 MiB，驱动 580.76.05；Python 3.12.13，Torch 2.11.0+cu128，Transformers 5.18.0，Accelerate 1.15.0，bitsandbytes 0.50.2，PEFT 0.21.2，tokenizers 0.23.2，PyYAML 6.0.3。

模型：`Qwen/Qwen3.5-9B`，配置 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，本地位置见架构文档。四个权重分片及 tokenizer 与既存 `SHA256SUMS.expected` 全部一致；这验证本地资产完整性，不单独证明远程来源。词表与嵌入形状检查通过。

固定条件：seed 0、greedy、关闭 thinking；12 题 × 3 次；输入上限 4096、输出上限 512 token。长度探针输出上限 64，允许越过常规输入上限。未改模型、精度、模板或任务集合。

产物根目录：`/test/Finagent/artifacts/m0_step1/`。每个运行目录保存 `records.jsonl` 与 `summary.json`。

| 运行目录（UTC） | 来源 | 任务通过 | 短任务峰值 allocated / reserved MiB | 时延中位数 | 失败 |
| --- | --- | --- | --- | --- | --- |
| `bf16/20261006T133813Z` | 既存结果，本次重评分 | 36/36 | 18204.3 / 18322.0 | 1.653 s | 无 |
| `int8/20261006T134017Z` | 既存结果，本次重评分 | 30/36 | 11190.0 / 11318.0 | 4.6715 s | 收入增速、工具澄清各 3 次 |
| `int4/20261006T134452Z` | 既存结果，本次重评分 | 33/36 | 8098.7 / 8228.0 | 1.4885 s | 工具澄清 3 次 |
| `bf16/20261006T141612Z` | 本次实际执行，退出码 0 | 36/36 | 18204.3 / 18322.0 | 1.502 s | 无；`passed=true` |

四次运行均三次输出一致，无记录中的生成错误、解析错误或短任务输出截断。量化失败结果保留，未通过修改题目来提高分数。旧记录未保存源码/任务哈希，归因能力弱于本次运行；本次重评分没有改变其原始产物。INT8 历史日志还记录了内部 BF16→FP16 转换警告，不能把它描述成纯 BF16 计算。

本次长度探针实际输入 971/1920/3818/7627 token，均通过；最长探针峰值 allocated 20084.4 MiB、reserved 21554.0 MiB，时延 2.143 s。目标长度 8192 不等于实测输入 8192；输出提前结束，因此未证明满 4096 输入加满 512 输出的资源上界，也未发现显存的最大可运行边界。

复跑命令（项目根目录）：

```bash
timeout 600 .venv/bin/python src/evaluation/m0_inference_check.py --config configs/m0_inference.yaml --variant bf16
```

上限 10 分钟；逐题日志显示完成数，结果写入新的 UTC 时间目录。中断后用同一命令重新运行，不覆盖旧目录；推理探针没有 checkpoint 恢复。

本次源码 SHA256：`6326e0cef85d829ae878f3315ba06976ba66251aa8b7662c867fb8fce84d82a8`。配置 SHA256：`63d33d61c35cc53f06b8b8f674c47fb258db58c23a5b5a6b143e75e1f16d8428`。模板 SHA256：`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`。任务/工具哈希见本次 summary。目录无 Git 元数据，无法填写 commit。

代码验证：先复现非法数组、重复/残缺参数和多余调用被接受；初始测试出现 7 个失败及 4 个尚未实现重复检查函数导致的错误。修复后 `python -m unittest discover -s tests -v` 的 12 项测试全部通过，包括入口拒绝错误答案、解析错误和截断输出。Python 编译通过。一次“配置 EOS 必须等于 tokenizer EOS”的检查失败；检查 token 含义后确认原停止集合覆盖两者，修正检查断言，未改生成语义。

## 2026-10-06｜M0 第 2 步前置核查（未验收）

- [FinQA](https://github.com/czyssrs/FinQA/tree/0f16e2867befa6840783e58be38c9efb9229d742)：train 6251 条，ID 示例 `ADI/2009/page_49.pdf-1` 可提供公司/年份线索；尚未审计其他划分。仓库提供 MIT LICENSE，未据此推断第三方原财报的所有使用权。
- [TAT-QA](https://github.com/NExTplusplus/TAT-QA/tree/870accc41953dcde885aabeb963d94aabdc0fbc3)：train 2201 个上下文，样本字段只有 table/paragraphs/questions，没有显式源报告、公司或年份字段。[官方 README](https://github.com/NExTplusplus/TAT-QA#license) 区分数据 CC BY 4.0 与代码许可。
- 官方 TAT-QA 文件树未发现独立报告映射；[TAT-DQA](https://github.com/NExTplusplus/TAT-DQA) 指向 Google Drive 数据，并说明来自 TAT-QA。尚未读取该下载资产，不能声称来源映射不存在。
- 本次仅网络读取结构与版本，未接入训练数据、生成 manifest、冻结划分或运行训练。是否补齐来源映射，或先隔离 TAT-QA 并缩小首轮数据范围，待用户确认。

## 2026-10-06｜TAT-QA 来源审计

官方依据：[TAT-DQA 数据格式](https://nextplusplus.github.io/TAT-DQA/) 提供 `doc.uid/page/source`，可用于关联原财报文件名。使用的 Hugging Face 版本为 `bc46208b22f0bfe47152c98fd0938d238ce868c2`；train/dev/test/test_gold 四个 JSON 均与官方页面链接的 Google Drive 文件逐字节一致。版本、地址及全部原始哈希固定在 `configs/tatqa_sources.json`。

首次探索使用 test_gold：2756 个上下文、16546 题，2706 个匹配、50 个未匹配，173 份来源报告。后续发现公开 test 是 278 个上下文/1669 题，而 test_gold 是 277 个/1663 题，且上下文 ID 不直接兼容。因此该次结果不作为完整公开输入审计结论；保留在 `data/audits/tatqa_sources_v1/`。

最终完整公开输入审计（v3，v2 同样本但尚未强制输入哈希）：

| 官方划分 | 上下文数 | 唯一报告关联 | 隔离 |
| --- | --- | --- | --- |
| train | 2201 | 2158 | 43 |
| dev | 278 | 271 | 7 |
| test | 278 | 273 | 5 |
| 合计 | 2757 | 2702 | 55 |

共 16552 题，隔离 330 题。关联覆盖约 98.0%，来源文件名 171 个，其中 145 个出现在 TAT-QA/TAT-DQA 的多个官方划分中。隔离项均为无整组精确匹配，不通过单题或答案相似度补齐。匹配仅验证数据集记录关联，尚未逐份读取原 PDF；不能声称已经完成来源分组去泄漏。

运行：`.venv/bin/python -m src.data.tatqa_source_audit --output-dir data/audits/tatqa_sources_v3`。输出为 `source_links.jsonl`、`summary.json`，后者包含代码和输入哈希；复跑需使用新目录，以免覆盖旧结果。新增 7 项回归测试通过，全套共 19 项通过。本次无模型推理、训练、依赖安装或大规模 PDF 下载。

第 2 步仍需完成 FinQA/ConvFinQA 来源核对、统一 manifest、可见材料与标签物理分离，以及官方/报告分组两种协议冻结。来源检索问题已取得可执行解法，不需要用户自行查找报告映射。

## 2026-10-06｜统一接入初稿（划分未冻结）

FinQA revision `0f16e2867befa6840783e58be38c9efb9229d742`；ConvFinQA revision `cf3eed2d5984960bf06bb8145bcea5e80b0222a6`，官方 `data.zip` 含 3037/421/434 个 train/dev/test_private 会话，另有 turn 派生文件。本次仅按会话文件核对来源，未重复计入 turn 派生数据。

运行：`.venv/bin/python -m src.data.build_manifest --output-dir data/staged/m0_v1`。原始输入保留，24,833 个问题及 11,038 份证据已分离写入。候选官方协议按报告采取 test > dev > train 的留出优先级，尚非冻结或可报告基准成绩的协议。

| 数据/原划分 | 接入题数 | 候选保留 | 排除 |
| --- | --- | --- | --- |
| FinQA train | 6251 | 2328 | 3923 |
| FinQA dev | 883 | 489 | 394 |
| FinQA test | 1147 | 1137 | 10 |
| TAT-QA train | 13215 | 584 | 12631 |
| TAT-QA dev | 1668 | 222 | 1446 |
| TAT-QA test | 1669 | 1639 | 30 |

排除原因总数：报告与留出划分重叠 18023；FinQA 2019 年公司别名待核验 81；TAT 来源未精确匹配 330。未删除原样本，也未将排除记录计作保留集成绩。

23 项单元测试通过；真实产物验证通过：问题/标签/manifest ID 完整对应、引用存在、证据字段白名单通过、候选保留集间源报告交集为空。原始文件与代码 SHA256 见 summary。仍需近重复检查、公司别名核验和公开测试输入与 gold 版本对齐；`split_protocol_frozen=false`，未训练、未将 M0 第 2 步标为完成。

## 2026-10-06｜M0 第 2 步验收通过

冻结快照 `data/staged/m0_frozen_v1/`：24,833 题、11,038 份证据，独立验收回执 `validation.json` 通过。近重复规则固定为 5-token shingles/Jaccard ≥ 0.9，检出 16 对跨报告近重复；整报告隔离后，保留官方归属的子集为 FinQA train/dev/test = 2316/485/1137，TAT-QA = 584/222/1639，总计 6383 题。

排除原因：留出报告重叠 18013，未决 FinQA 2019 公司别名 81，跨报告近重复 26，TAT 未匹配来源 330。另一个 `report_group_v1` 协议按报告归属为 train/dev/test = 2900/2784/18712，隔离 437；两个协议分别命名，不混用成绩。未决来源按既定计划隔离，不宣称已验证公司别名。

TAT 公开测试 1669 题中 1626 题通过严格上下文＋问题关联到发布标签，其余 43 题保留输入但不可评分；不按不兼容 ID 硬拼。ConvFinQA 固定 revision 的 MIT LICENSE 原文已核验，本次只用于来源隔离；两套 TAT 数据许可为 CC BY 4.0。原始文档第三方权益不作扩张性解释。

验收命令：`python -m src.data.validate_manifest data/staged/m0_frozen_v1`。所有输入哈希固定，记录/引用完整，证据字段白名单通过，候选官方保留集之间没有源报告或既定近重复交叉。恢复数据可运行 `python -m src.data.fetch_sources`，再用新输出目录执行 build_manifest 和 validate_manifest。

## 2026-10-06｜M0 第 3 步验收通过

环境和模型沿用第 1 步，未安装或更改精度依赖。训练配置见 `configs/m0_training.json`，本次代码与包版本见 `artifacts/m0_lora_v1/run_metadata.json`。2900 条可训练候选中 1381 条超过 1024 token；不截断证据，从完整可用样本固定选择 100 条，并包含最长 1024-token 样本。

训练前曾因 token 前缀比较触发 mask 检查失败，未启动 GPU 训练；改为官方模板文本与 tokenizer offsets 对齐后通过。真实试跑使用 BF16 基础权重、FP32 LoRA（21,639,168 参数），effective batch 8，20 次 AdamW 更新。视觉冻结、可训练模块清单、loss mask 和参数变化检查通过；所有 loss/梯度有限。峰值 allocated 21559.8 MiB，参考算子较慢但没有 OOM。

三条命令依次执行且全部退出码 0：

```bash
timeout 1800 .venv/bin/python -m src.training.m0_lora_check --output-dir artifacts/m0_lora_v1 --stop-after 10
timeout 1800 .venv/bin/python -m src.training.m0_lora_check --output-dir artifacts/m0_lora_v1 --resume artifacts/m0_lora_v1/step_0010
timeout 600 .venv/bin/python -m src.training.m0_lora_check --output-dir artifacts/m0_lora_v1 --resume artifacts/m0_lora_v1/step_0020 --verify-only
```

前后各 10 步训练分别约 83.2/82.8 秒（不含加载及数据准备）。第 10 步退出后用新进程恢复优化器、调度器、RNG 和样本游标，完成第 20 步。第三个进程重载时 loss = reference_loss = 0.2921101748943329；`verified.json` 为 passed=true。此处只证明训练/恢复可行，不据 100 条训练样本的 loss 宣称泛化提升。

全套 26 项 CPU 回归、Python 编译及 diff 检查通过。M0 三步均有实际验收依据，可以进入 M1。按最新授权，只本地阶段提交，用户最终审核前不推送。

## 2026-10-06｜M1 验收与失败记录

新增业务 Decimal 计算、原文坐标/事实引用、CPU BM25 和固定取证计算流程。47 项回归通过，覆盖期间/单位/币种/范围错配、零分母、负基数、百分点、非法引用和原生工具边界；编译与 diff 检查通过。

冻结 FinQA 来源隔离开发集的 40 个不同报告，种子 29；检索池 485 份可见证据，BM25 Recall@5 = 29/40。按官方执行值相等评分，仅是该开发子集，未声称完整 FinQA/TAT-QA 成绩。

初版命令 `python -m src.evaluation.m1_baseline --output-dir artifacts/m1_baseline_v1` 完成 80 次生成：直接回答 8/40、2 次接口错误；自定义 JSON 计算 0/40、34 次接口错误，多为格式/截断失败。两组累计生成时间 36.100/499.433 秒。无条件百分比提示亦存在口径问题；此失败实验保留，不能作为后续提示相同的基座对照。

改用原生工具声明并修正提示后，运行 `python -m src.evaluation.financial_run --output-dir artifacts/m1_native_v1`，相同证据、任务和 4096/512 token 预算，重新执行直接回答控制组：

| 条件 | 数值正确 | 接口/预算错误 | 正确且命中指定来源 | 累计生成秒 |
| --- | --- | --- | --- | --- |
| 给定证据直接回答 | 7/40 | 4 | 7/40 | 34.425 |
| 给定证据工具计算 | 13/40 | 16 | 13/40 | 114.883 |
| BM25 top-1 后工具计算 | 9/40 | 22 | 3/40 | 180.316 |

总墙钟 331.758 秒（不含模型加载）。错误和超预算均计入分母；没有截断输入或修改标签。局部工具计算优于本轮直接回答，但端到端存在碰巧数值一致、来源错误的问题；不能把 9/40 报作引用支持正确率。模型仍有选数/口径错误及无效事实编号，未作生产质量承诺。

`python -m src.evaluation.validate_financial_run --run artifacts/m1_native_v1` 通过：120 个任务-条件组合完整且唯一，评分与源 gold 一致，88 个已执行计算引用全部回查原始跨度。配置、源码/数据指纹、逐题输出、显存、耗时和错误在两个产物目录中。该审计证明结果可追溯，不替代语义正确性。M1 完成，允许进入训练对照。

## 2026-10-06｜M2 监督转换、回归基线与显存失败

`python -m src.training.prepare_sft --output-dir data/staged/m2_sft_v1` 固定种子 31，只检查来源隔离 FinQA train 的前 500 条候选。94 条通过程序执行与标签一致性、唯一原文数值绑定和完整轨迹 ≤2048 token 检查；排除重复/缺失数值绑定 149、完整轨迹过长 247、未支持表格操作 10。保留所有排除 ID/原因，未扩展到 2000—5000 条。

两个监督组同 94 个问题：答案式共 118460 token，其中监督 1245；动作式共 167143 token，其中监督 6650。动作式工具观察来自实际计算，只监督 assistant 动作和最终回答；这是给定 gold 局部证据监督，不是自主检索轨迹。抽查 RL/2012、AWK/2015、ORLY/2009 的动作、结果和 mask，工具观察/用户资料均不参与损失。

训练前冻结通用与合成页面回归：`python -m src.evaluation.regression --output-dir artifacts/m2_regression_base`，基座文本/工具 12/12，合成图像 3/3。图片仅检查能力退化，不能替代真实财报视觉验收。

原始完整词表损失的 2048 上限试跑：`python -m src.training.m0_lora_check --config configs/m2_action_training.json --output-dir artifacts/m2_action_v1 --stop-after 1`，最长动作样本 2039 token，在首次反向触发 OOM，未完成更新/保存；失败日志 `/tmp/finagent-m2-action-probe.log`，选择及模块清单在 `artifacts/m2_action_v1/`。异常时 allocated 约 22.00 GiB，再申请 1.89 GiB，空闲约 1.01 GiB。拟核验仅计算受监督位置 logits 的等价实现后重试；不截断输入、不改变 BF16、LoRA rank 或有效 batch。

## 2026-10-06｜M2 损失优化与训练完成

保留两次未通过的数值实验：直接缩减 BF16 投影位置的梯度相对差异 1.83%；保持完整 BF16 投影、仅缩减 FP32 交叉熵位置时为 1.71%，都超过最初 1% 阈值。对应记录为 `artifacts/m2_loss_equivalence_rejected_v1.json`、`m2_loss_equivalence_rejected_v2.json`，未将这两次检查写成通过。

进一步加入原实现自身的重复对照：完整 1024-token 样本中，原实现重复梯度相对差异 1.555%，与优化对照的 1.555% 接近。更直接的检查显示，两者 loss 均为 2.484687089920044，输出层传回解码器的梯度逐元素相同；FP32 小模型的损失及全部参数梯度也一致。因此采用保留完整 BF16 投影、只对监督位置计算 FP32 交叉熵的方案；不将参考算子的下游运行波动误归因为训练目标变化。验收记录 `artifacts/m2_loss_equivalence.json`，1024-token 峰值 allocated 21220.85→19290.97 MiB。

两组保持相同 94 个问题、40 次更新、有效 batch 8、BF16 基座及 rank-8 FP32 LoRA，完整输入上限 2048。动作组 `artifacts/m2_action_v2/` 在第 1/20/40 步保存并跨进程恢复，峰值 20853.54 MiB；答案组 `artifacts/m2_answer_v1/` 在第 20/40 步保存恢复，峰值 20234.99 MiB。两组第 40 步独立重载验证均通过，loss 与保存参考值分别完全相等：动作 0.00010536138142924756、答案 0.0036869850009679794。这些训练 loss 不代表泛化成绩。

`configs/m2_selection.json` 在能力回归前固定采用门槛：给定证据工具计算的配对改善通过精确 McNemar p<0.05，通用及合成视觉探针不退化，累计生成耗时不超过基座两倍，否则保留基座。后续真实对照结果另行追加；暂不扩展监督数据规模。

## 2026-10-06｜更正 M1 引用支持判定

复核发现：同一财报页的多个问题拥有不同 `evidence_id`。此前 `correct_evidence` 只比较 ID，导致 6 个真实正确且有支持的答案被误记为来源错误；这 6 个案例的报告来源和完整结构化原文均相同。撤回前文对这些案例“碰巧数值一致、来源错误”的归因。

新增独立 `citation_audit.json`，不修改原始生成或数值评分。规则为报告来源相同且去掉 evidence_id 后的完整结构化原文相同，不能仅凭相同数字或同一报告合并。M1 检索后计算的数值正确且来源支持应为 **9/40**；原始 ID 精确命中指标仍为 3/40。动作 SFT 同条件为 **14/40**，原始 ID 命中 6/40。两组使用相同规则，所有阶段后续引用评分以此审计为准。

## 2026-10-06｜M2 验收完成与采用决定

两组均完成相同 40 题 × 3 条件，完整性、gold 评分一致性及原文数值跨度审计通过；动作/答案组分别解析 145/135 个计算引用。结构化来源别名审计单列，原始输出保留。模型和预算沿用 M1，数据、适配器与实际运行源码可在 `artifacts/m2_provenance/` 核对。

| 模型 | 直接回答 | 给定证据计算 | 检索计算且来源支持 | 文本/工具回归 | 合成视觉回归 |
| --- | --- | --- | --- | --- | --- |
| 基座 | 7/40 | 13/40 | 9/40 | 12/12 | 3/3 |
| 答案式 SFT | 14/40 | 21/40 | 16/40 | 11/12 | 3/3 |
| 动作式 SFT | 8/40 | 21/40 | 14/40 | 12/12 | 3/3 |

两种 SFT 的工具计算均相对基座新增答对 10 题、退步 2 题，精确 McNemar p=0.038574；这是未作多重比较校正的小规模开发探索，不据此声称普遍能力提升。动作组计算生成累计 176.745 秒，约为基座 1.538 倍；答案组 174.707 秒。失败调用及超预算保留在分母。答案组 `t09_tool_clarify` 改为自然语言追问，没有按要求调用澄清工具，违反冻结的工具契约。

按 `configs/m2_selection.json` 选择 **动作适配器**，路径 `artifacts/m2_action_v2/step_0040`；`configs/model_choice.json` 是后续入口。答案组虽有更高直接回答/检索数值成绩，但未通过通用工具回归门槛，不采用。保留全部基座、答案组和失败实验，暂不扩展到 2000—5000 条；94 条小规模训练尚不足以证明生产可靠性。

实际 320 个 micro batch：动作组输入 576446 token/监督 22706，答案组输入 410649/监督 4249；监督 token 量不同，不能把全部差异归因于动作表示本身。M2 相关 54 项 CPU 回归、编译与 diff 检查通过；另有尚未完成真实验收的 M3 基础测试。只提交本地阶段结果，不推送。

## 2026-10-06｜M3 冻结协议与真实故障验收

独立任务在调优前冻结：30 个 FinQA 单题、20 个同公司跨报告任务、15 个来源隔离 ConvFinQA 会话、15 个缺失条件/资料不足控制，共 80 个任务、115 轮。输入与 gold 分离，检索库 1152 项；完整哈希见 `configs/m3_acceptance_lock.json`。不得用该验收集调参后仍称其独立。

`python -m src.evaluation.fault_smoke --adapter artifacts/m2_action_v2/step_0040 --output-dir artifacts/m3_fault_v1` 通过 8 项：进程互斥、真实生成取消、实际 CUDA OOM 不提交报告、OOM 后健康检查、文档无法扩张取证范围、故障后真实推理、模型显存释放、提交后子进程崩溃不重复报告。取消响应 0.858 秒；这是执行不变量验收，不是金融答案正确率。

业务命令：`timeout 3600 .venv/bin/python -m src.evaluation.agent_run --adapter artifacts/m2_action_v2/step_0040 --output-dir artifacts/m3_acceptance_run_v1`。逐轮结果和 SQLite 事件实时保存，日志 `/tmp/finagent-m3-acceptance.log`；异常退出后用原命令加 `--resume` 恢复，已消耗调用和原截止时间不重置。配置为每轮最多 10 次模型调用、180 秒、8192 输入/512 输出 token；真实业务尚在运行，结果另行追加。

## 2026-10-07｜M3 独立业务验收与恢复记录

`artifacts/m3_acceptance_run_v1/` 完成全部 80 项、115 轮；`validate_agent_run` 独立核对 41 份唯一报告、65 个原文数值引用及逐轮评分通过。这里的审计通过不等于业务答案全部正确。

| 类别 | 整项通过 | 补充 |
| --- | --- | --- |
| 单题计算 | 5/30 | 同适配器、同检索资料的固定流程为 18/30 |
| 跨证据 | 1/20 | 要求全部计算与来源匹配 |
| 多轮会话 | 0/15 | 单轮正确且来源支持 17/50，不能据此称会话通过 |
| 条件缺失/资料不足 | 15/15 | 范围守卫可不调用模型；澄清不生成最终报告 |

动态流程实际发起 756 次生成，755 次返回，观察到 40151 个输出 token、3615.538 秒生成耗时；中断调用的输出未知，保留预算消耗，不记为免费调用。失败包括输入超预算 20 轮、调用预算耗尽 43 轮、原截止时间耗尽 1 轮。固定对照 30 次生成共 1683 个输出 token、143.037 秒。重复检索、非法提交格式及指标选择错误均保留原始事件，不在验收集上调参。

首进程达到 3600 秒上限退出；续跑坚持使用 `source/` 中的原源码。归档漏存 `configs/m0_inference.yaml` 导致恢复启动失败，未调用模型；核实配置与原 HEAD 完全相同后补存，哈希与说明在 `resume_config_recovery.json`。实际续跑命令在 `artifacts/m3_acceptance_run_v1/source` 执行：`timeout 1800 /root/Finagent/.venv/bin/python -m src.evaluation.agent_run --adapter artifacts/m2_action_v2/step_0040 --output-dir artifacts/m3_acceptance_run_v1 --resume`。原始截止时间、调用数和中断事件均未重置。

按原计划“无收益时保留基线”，默认交付改用固定取证计算流程；动态 Agent 保留为实验入口。新增的跨工作目录 GPU 锁、任务资料哈希绑定和固定流程恢复不改变这次冻结实验的模型提示或评分。M3 所属 81 项 CPU 回归通过；真实故障复验另记。

最终源码真实故障复验：`timeout 600 .venv/bin/python -m src.evaluation.fault_smoke --adapter artifacts/m2_action_v2/step_0040 --output-dir artifacts/m3_fault_v2`，8/8 通过，真实生成取消 0.851 秒，OOM 后健康检查、再次推理及显存释放通过。原始日志已保存各实验目录。M3 工程与验证工作完成，金融质量限制如上，不宣称生产可用。


## 2026-10-07｜M4 原始财报与检索验收

最初 Microsoft 年报 HTML 请求返回 403，另一 HTML 请求 60 秒超时，未接入失败来源。最终使用发行人公开 PDF：中国移动 [2022 业绩](https://www.chinamobileltd.com/sc/media/press/p230323.pdf)、[2023 业绩](https://www.chinamobileltd.com/sc/media/press/p240321.pdf)及 2023 年报 C13/C14/C16 财务章节，共 5 份、17 页。发布日在新闻稿首页核对为 2023-03-23、2024-03-21；年度章节的董事会批准日不冒充披露日。来源、SHA 和当地研究使用边界见 `configs/report_sources.json`。

初版 `data/staged/reports_v1/` 因证据字段不符合既有结构被拒绝，未运行模型；修正后有效资料为 `reports_v2/`。运行 `src.evaluation.retrieval_run --output-dir artifacts/m4_retrieval_v2`、`src.evaluation.pit_run --output-dir artifacts/m4_pit_v2`；v1 对照保留，v2 加入截止范围内 IDF 隔离。

| 检索 | Recall@5 | MRR | nDCG@5 | 首位相关页 | 中位查询耗时 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 1.000 | 0.8611 | 0.9020 | 9/12 | 0.258 ms |
| 稀疏融合 | 1.000 | 0.9167 | 0.9437 | 10/12 | 1.250 ms |
| 词面重排 | 1.000 | 0.9583 | 0.9692 | 11/12 | 1.280 ms |

按预先固定的收益/成本规则采用重排。这是一个发行人的小规模开发比较；首位相关页不等于生成答案正确。真实截止日 2023-03-22/23、2024-03-20/21 分别允许 0/6/6/11 页，所有越界读取均被拒绝。其余 6 页因发布时间未知被排除；不声称当时已经入库，也不提供日内交易时点保证。9 项模块测试覆盖合成修订替换、缓存/历史失效及未来文本统计隔离；未伪造真实重述案例。

固定动作适配器、8192/512 token 预算的 36 次真实生成对照完成，产物 `artifacts/m4_answers_v1/`。BM25/融合/重排的有效计算分别 10/9/9，引用可回查且页面相关均为 **8/12**，累计生成耗时 51.214/44.293/44.240 秒。8 次输入超预算均保留为失败，未裁剪页面。独立核对全部 36 条、计算值及原文跨度通过。此处只验证数值引用和页面相关性，多值问题可能超出单计算契约，不能称完整金融答案正确率；检索排名收益没有转化为本轮生成引用收益。

M4 的真实资料、检索比较及时间约束验证完成，采用仅限小库检索改进；中文答案、视觉和端到端展示继续单独验收。新增 PDFium 依赖并固定现有核心包版本，未更换系统 Torch 或精度。


## 2026-10-07｜M5 官方报告资产与冻结输入

沿 [TAT-DQA 官方项目页](https://nextplusplus.github.io/TAT-DQA/) 找到测试文档档案，官方 Drive 下载与固定镜像文件的 SHA-256 均为 `83c3baf38102fe5b17041efe41ea770912e6afa32eb603fece5337318259ffc8`，173931848 字节。`data/raw/tatdqa/test_archive_provenance.json` 保存验证记录。277 个标注上下文均找到配对 PDF/OCR/图像；不是臆测原始报告网址，也不将镜像自称发行人来源。

档案 PNG 仅 224×224，不能直接匹配 OCR 坐标。英文输入 v1 因此拒绝；v2 又发现标量过滤混入日期/文字而拒绝，均未运行模型，保留 `rejection.json`。v3 从原 PDF 重渲染完整页面，按冻结来源分组排除 4 个不合格上下文，因单页预算排除 35 个多页上下文，seed=97 选 12 个不同来源（6 算术、6 标量抽取）。不宣称完整测试集成绩；任务和 gold 哈希在 `lock.json`。

推理前逐页检查发现 KEMET 印刷页 51 的 Film and Electrolytic 表单位是千，发布标签却写百万；问题还省略分部名称。`configs/m5_label_audit.json` 固定该问题说明：主指标仍按原标签，另报原文单位复核指标，不删题、不改主 gold。中文题来自中国移动 2023 年业绩原始中文 PDF 第一页，共 8 题，包含亿元/百万元、人民币与港元每股值、期间比较；原文和 Decimal 参考计算已交叉核对。这是代理原页复核，不声称独立人工标注团队验收。

英文固定 12×3 条件，加 3 个运行时图像张量遮挡、3 个基础单位变换，共 42 次；中文为 8×文本/视觉，共 16 次。BF16 动作适配器、8192 输入/256 输出 token、1048576 像素上限；按需路由规则在输出前冻结，原图和相邻年份不改。评分限标量数值 EM/单位，不报告区域定位。实际成绩另行追加。


## 2026-10-07｜M5 英文页面实测

`timeout 1800 .venv/bin/python -m src.evaluation.visual_run --output-dir artifacts/m5_visual_run_v1` 完成 42 次；`validate_visual_run --run artifacts/m5_visual_run_v1` 从保存生成重新评分及配对审计通过。源图为约 1240×1605—1754，处理器限制像素后最多 1008 个视觉 token；全量上下文保留，未微调或更换适配器。

| 条件 | 原标签数值 EM | 原文单位复核 EM | 标签尺度一致 | 调用总耗时 | 峰值 allocated |
| --- | --- | --- | --- | --- | --- |
| 文本 OCR | 9/12 | 9/12 | 11/12 | 20.921 秒 | 18455.1 MiB |
| 原页视觉 | 4/12 | 5/12 | 5/12 | 100.719 秒 | 18460.3 MiB |
| 按需结合 | 6/12 | 6/12 | 8/12 | 23.026 秒 | 18730.0 MiB |

耗时含页面预处理，不含模型首次加载；按需路线只有 4 次使用图片。视觉失败中 4 次格式不符、1 次输出截断、3 次数值/指标不符；按需路线 4 次格式不符、2 次数值不符。即使解释里出现正确数值，未遵守冻结 JSON 契约仍计失败。KEMET 视觉读取正确千单位而原标签错误；Westell 视觉读到 4611，但附带解释未通过格式。Jabil 同页两处“薪酬增长率”口径未在问题中限定，模型选 4.4%、标签为 4.3%，不事后改评分。

全部图像张量遮挡后 3/3 按要求拒答；基础单位转换 **0/3**：1 次正确数值但格式失败，2 次未乘尺度。控制调用分别耗时 3.462/14.621 秒，无 OOM。主结果不支持把视觉或按需结合替换文本路线；没有把少量读数改善包装为总体收益。失败原文、图像关联和原标签均保留。


## 2026-10-07｜M5 中文验证与阶段结论

`timeout 1800 .venv/bin/python -m src.evaluation.chinese_run --output-dir artifacts/m5_chinese_run_v1` 完成 16 次；`validate_visual_run --run artifacts/m5_chinese_run_v1 --chinese` 独立复核通过。文本数值且单位正确 **7/8**，视觉 **5/8**；调用总耗时 17.713/19.056 秒，无格式、单位标签或运行错误。文本把按原始收入计算的 7.69% 写成 7.68%；视觉另错亿元转百万元和收入差额，并直接采用披露中更粗的 7.7% 而不满足题目两位小数要求。任务要求与错误原文均可回查。

M5 的英文配对、遮挡/单位控制及独立中文原文验证全部执行并留存失败，3 项数值协议测试通过。结论是保留文本路线；视觉只作离线实验，不发布中文自主研究或可靠自动单位转换能力。没有视觉微调，也没有以英文题翻译替代中文资料。下一阶段只封装已验证的固定流程和持久化执行边界。


## 2026-10-07｜M6 部署、交互与发布验收

在 `/tmp/finagent-deploy-check-v1` 复制源码，创建新的 `.venv --system-site-packages` 并安装固定 `requirements.txt`；仅显式链接已核验的数据和适配器，基础 CUDA Torch/驱动沿用同一宿主机。这是干净源码和新虚拟环境验收，不冒称全新操作系统部署。最终 **100 项回归全部通过**，编译与 diff 检查通过。

浏览器准备失败完整保留：`m6_browser_v1` 缺 libnspr4，v2 缺 libXi，v3 是测试工具的字符串求值与 CSP 冲突；改用函数形式等待，未放宽页面 CSP。共享库/中文字体解到本地产物目录，未安装系统包；v4、最终 v5 均 **8/8** 通过：丢失响应重试幂等、刷新事件重连、排队取消、澄清新修订、来源不可执行、导出一致、移动布局和无 JS 错误。初次 HTTP 测试的 502 来自环境代理拦截回环请求，测试客户端仅对本地请求使用无代理连接。浏览器使用合成状态夹具，与真实模型质量分开。

实际命令在独立目录执行：`timeout 900 .venv/bin/python -m src.evaluation.deployment_smoke --output-dir artifacts/m6_deploy_v1`。真实服务 **10/10** 通过：ANSS 2012 收入地域占比计算、2011 连续追问、证据回查、导出一致、资料不足、澄清挂起、生成中取消、崩溃后报告保留、事件不重复、优雅退出释放资源。两次原始计算值为 0.33262 和 0.31228；网页明确它们是计算原值，不自动换算成百分数。取消响应 **0.831 秒**。这是冻结的已知来源展示，不是新增独立正确率估计。

验收产物已归档为 `artifacts/m6_deploy_clean_v1/`（原始 HTTP 记录、SQLite、服务日志、源码/资产 SHA、新环境依赖、100 项测试日志）；界面截图与 8 项交互结果在 `artifacts/m6_browser_v5/`。所有本次模型/服务子进程已退出。默认服务只封装固定取证计算与持久化状态；动态、视觉和中文自主研究不列入已达标成果，MCP/Skill 未实施也不计成果。

M0—M6 共 15 步的实现、实际实验及验证工作完成；低质量和失败实验如实保留。发布使用本地标签 `v0.1.0-research` 标识当前研究原型；阶段提交与标签均不推送，等待用户统一审核。
