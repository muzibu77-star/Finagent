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
