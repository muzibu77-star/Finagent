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
