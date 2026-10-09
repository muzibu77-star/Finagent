# 环境恢复与运行

## 已验证的资产边界

Python 3.12.13；RTX 4090 24GB；驱动 580.76.05。系统提供 Torch 2.11.0+cu128 / torchvision 0.26.0+cu128。项目虚拟环境通过 `--system-site-packages` 复用它们；其余核心版本固定在 `requirements.txt`。参考算子回退较慢，但本轮不安装额外训练内核。

模型 revision、绝对路径和 BF16 推理配置见 `configs/m0_inference.yaml`。采用的适配器路径见 `configs/model_choice.json`。根目录 `Qwen3.5-9B/` 是未完成的另一份下载，不用于运行。Git 不包含模型、数据、适配器、SQLite 或实验产物；移机须恢复这些资产并核对冻结哈希，不能仅靠 clone 启动。

在具备上述系统 Torch 的机器创建环境：

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -q
```

若需要重建数据，依次运行 `src.data.fetch_sources`、`src.data.build_manifest`、`src.data.validate_manifest`，参数及原始快照见 [架构](architecture.md) 和 [结果](result.md)。M3 服务输入为 `data/staged/m3_acceptance_v1/corpus.json`；重建入口是 `src.evaluation.freeze_acceptance`。恢复须包括 `configs/model_choice.json` 引用的历史 `step_0040` 和计算专用 `full_context/step_0196`，不能把基础模型当作适配器恢复。

## 服务与状态

```bash
.venv/bin/python -m src.service.server --database artifacts/calculator-service-v2/tasks.sqlite --port 8765
```

本机地址 `http://127.0.0.1:8765`；远程机器使用端口转发。上例使用新的计算服务数据库，默认固定单题流程。未指定 `--database` 时仍是旧默认 `artifacts/service/tasks.sqlite`；旧库没有本轮模型身份字段，不能直接混用。数据库绑定资料快照、适配器配置/权重及文档 Unicode/查值提示版本，改变任一项须新建数据库；保留旧库和原始资产用于历史恢复。网页刷新会重连事件，取消传到生成，澄清使用原任务的新修订。`Ctrl+C` 请求停止当前任务并释放模型；崩溃后使用相同命令重启，已提交报告不重复生成。未完成的生成调用保留消耗记录，不擅自重置预算。

原始 PDF 文档库入口（单独数据库；数值仍须复核）：

```bash
.venv/bin/python -m src.service.server --reports --corpus data/staged/reports_v2/corpus.json --snapshot reports-v1 --database artifacts/report-service-v2/tasks.sqlite --port 8765
```

动态 Agent 的冻结实验入口是 `src.evaluation.agent_run`，不作为默认服务流程。M5 视觉/中文入口是离线质量实验，不自动成为已验收的中文自主 Agent 或视觉服务。

## 浏览器验收

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
PLAYWRIGHT_BROWSERS_PATH="$PWD/artifacts/browsers" .venv/bin/python -m playwright install chromium --only-shell
PLAYWRIGHT_BROWSERS_PATH="$PWD/artifacts/browsers" .venv/bin/python -m src.evaluation.browser_smoke --output-dir artifacts/browser-new-run
```

本容器缺少浏览器共享库，因此将 Ubuntu 包解到 `artifacts/browser-deps/root/`，没有安装系统包。此环境运行上条命令还需设置 `LD_LIBRARY_PATH=$PWD/artifacts/browser-deps/root/usr/lib/x86_64-linux-gnu`、`FONTCONFIG_FILE=$PWD/artifacts/browser-deps/fonts.conf`。浏览器测试使用合成状态夹具，只验证交互；真实模型部署验收单列在结果记录。

长实验均写入独立目录；不要覆盖唯一有效 checkpoint 或历史结果。训练恢复使用原 checkpoint 的完整状态，推理服务恢复使用原 SQLite。外部 PDF 仅作本地研究与必要引用，不把下载资产随仓库重新发布。


## 发布与验收回放

本地研究版本 `v0.1.0-research`；阶段质量与限制以 [result.md](result.md) 为准。当前本机保留所有模型、适配器、数据和运行产物，尚未推送发布。新环境部署证明在 `artifacts/m6_deploy_clean_v1/`；其中 `source_sha256.json` 固定实际部署源码，`asset_sha256.json` 固定配置、资料和适配器，`environment.json` 保存依赖。模型本体沿用 `configs/m0_inference.yaml` 中的固定 revision。

完整服务演示（约一分钟，最多 15 分钟；输出目录必须不存在）：

```bash
timeout 900 .venv/bin/python -m src.evaluation.deployment_smoke --output-dir artifacts/deployment-new-run
```

输出 `summary.json`、`records.json`、`tasks.sqlite` 和 `server.log`；验收标准为 `passed: true` 且 10 项检查全通过。脚本自行启动和关闭服务，并主动测试一次提交后的进程崩溃。失败目录保留，重跑换新目录；实际服务则使用原数据库恢复，二者不要混用。

英文/中文离线质量回放（逐项串行，每项最多 30 分钟）：

```bash
timeout 1800 .venv/bin/python -m src.evaluation.visual_run --output-dir artifacts/visual-new-run
timeout 1800 .venv/bin/python -m src.evaluation.chinese_run --output-dir artifacts/chinese-new-run
```

输出 `records.jsonl`、`settings.json`、`summary.json`，分别应有 42/16 次调用，答案失败仍保留。中断可对原目录加 `--resume`，输入与协议哈希须一致。独立评分用 `.venv/bin/python -m src.evaluation.validate_visual_run --run 运行目录`，中文加 `--chinese`。这些实验不能代替服务金融答案的人工复核。

## JD 优化环境与当前运行门槛

新增 CPU/检索/MCP/官方评分依赖固定在 `requirements-research.txt`，安装于原 `.venv`。BGE 权重位于 `models/`，revision 见 `configs/jd_retrieval.json`，文件哈希与实际下载失败记录见 [result.md](result.md)。原始扩展 PDF 在 `data/raw/expanded_reports/`、`data/raw/chinese_expanded/`；来源 URL 与哈希在各快照 receipt/source/config 中。不得把下载 PDF 或模型提交 Git。

此前 CUDA 12.8/13.0 权重一致性检查失败；2026-10-08 在未修改驱动、依赖、权重或训练代码的原训练环境复查通过，随后完成配对扩训、跨进程续训和重载一致性检查。根因未定，本次通过不追溯证明旧实验可靠；新实验应先执行下述门槛检查，出现非有限损失或数据差异立即保留失败并停止依赖实验。原失败指纹位于 `artifacts/jd_provenance/`，复查证据位于 `artifacts/device_diagnosis_v2/`，后续模型实验位于 `artifacts/jd_resume_20261008/`。

`.venv-serving` 是隔离环境：Torch 2.11.0+cu130、TorchAudio 2.11.0+cu130、Transformers 5.19.0、vLLM 0.25.1、Triton 3.6.0；对应依赖见 `requirements-serving.txt`。成功启动命令在 `artifacts/jd_resume_20261008/serve_v3.sh`，显式设置 venv `bin` 路径以发现已安装 ninja，并将 CUDA 13 动态库及编译缓存限制在该服务进程。固定长度性能已实测；模型质量状态见 [progress.md](progress.md)，不能从进程启动成功推导质量通过。运行 supervisor 只负责自己创建的服务进程组。

后续正式模型作业前的完整验证命令（输出目录须不存在；使用全局锁避免与其他项目 GPU 作业重叠）：

```bash
flock -n /tmp/finagent-cuda0.lock .venv/bin/python -m src.evaluation.verify_device_copy --output-dir artifacts/device-copy-recovery-check
```

输出 `summary.json`，验收要求 `passed: true` 且三次 `different_elements` 都为 0；失败时退出码为 1，不启动依赖实验。CUDA 13 环境执行相同入口时使用 `.venv-serving/bin/python`，必要时仅为该进程设置 `LD_LIBRARY_PATH=$PWD/.venv-serving/lib/python3.12/site-packages/nvidia/cu13/lib`。诊断不是显存维修，也不是完整硬件健康证明；通过后仍须实际最长样本前向/反向和保存恢复。

新增入口及恢复方式：

- RAG 排名：`src.evaluation.rag_run`，资料 `data/staged/jd_reports_v1/corpus.json`、标签 `data/staged/jd_rag_v1/`；新索引与排名为 `artifacts/jd_resume_20261008/retrieval_index/`、`retrieval/`，原 `artifacts/jd_retrieval_v1/` 保留。当前向量缓存绑定配置、资料和实现哈希；不能跨实现或跨 GPU 检查状态复用旧向量。该入口内部持有 GPU 锁，不要外层重复 flock。
- 扩训：`src.training.m0_lora_check --config configs/jd_action_3072_training.json`，答案臂有对应配置。两组最终完整 checkpoint 为 `artifacts/jd_resume_20261008/{action,answer}/step_0251`，包含优化器、调度器、RNG 和样本游标；中间 50/150 步及原诊断 step 2 均保留。最终 `--verify-only --resume` 通过；不得对已完成目录重复执行训练并覆盖记录，也不能将训练完成当作模型采用通过。
- 全量评分：`src.evaluation.full_benchmark --output-dir ...`；`--resume` 绑定设置和逐题记录。`--split dev/test` 分别使用冻结开发/测试输入。可选 `--server-url http://127.0.0.1:18081 --served-model base --workers 4` 使用独立已启动的 vLLM 文本服务；服务器生命周期与客户端独立，旧 40 题后端校准不得冒称全量评分。
- 新动态实验：`src.evaluation.agent_run --tasks-root data/staged/agent_v2_dev --context-management --output-dir ...`。最终验收资产为 `agent_v2_test`，不得调参；自主轨迹训练检查点在下述外部资产盘，质量采用状态见 [progress.md](progress.md)。
- 实际 MCP 回放：`.venv/bin/python -m src.evaluation.mcp_smoke --output-dir artifacts/mcp-replay-new`；Skill 随仓库通过 MCP prompt 加载，不需要全局安装。原成功记录在 `artifacts/jd_mcp_v3/`。
- 中文当前准备集为 `data/staged/jd_chinese_v4/`；模型入口增加 `--tasks-root data/staged/jd_chinese_v4 --text-only`。跨页标题目前只合入文本，禁止将单页图片冒称等价多页视觉输入。真实更正回放为 `src.evaluation.restatement_run`，新纶另传 `--config configs/jd_restatement_xinlun.json`；两者每次使用新输出目录。

正式服务 TTFT/吞吐入口为 `src.evaluation.serving_benchmark`；冷启动须同时保留服务启动日志，历史 BF16/INT8/INT4 质量不能冒充 vLLM 新测量。P2 在 P0-2 验收前不启动。

新增质量实验检查点使用已有项目资产盘的独立目录 `/test/Finagent/artifacts/jd_quality_20261008/`；本地 `artifacts/jd_resume_20261008/{full_context,autonomous,dpo}` 为对应目录符号链接。初始化时本地 overlay 仅余约 3.8 GiB，因此完整可恢复检查点保存在该资产盘；迁移须包含链接目标。存储凭据为 `quality_checkpoint_storage.json`。三组最终恢复点分别为 `full_context/step_0196`、`autonomous/step_0012`、`dpo/step_0009`；DPO 另须恢复 `preference_reference/` 内固定参考概率及对应数据哈希，不能用新采样偏好替代原恢复数据。

对话压缩完整开发与后续采样的新编排为 `launch_serving_v5.py` → `serving_evaluation_v5.sh`；`continue_followup_v2.py <owner PID>` 等待新服务 owner 退出并核验完成标记后，才启动 `train_followup.py` 和 `launch_followup.py`。当前是否已启动及停止授权以 [progress.md](progress.md) 为准。各项原始输出、恢复点和失败记录保存独立文件；不得把旧失败标记删除后冒充从未失败。

## 本轮采用后的启动与重放

计算服务使用 `calculator_adapter`，报告服务使用历史 `adapter`，均由 `configs/model_choice.json` 固定；不能把计算全量成绩外推成文档问答准确率。同一 GPU 只运行一个模型服务，既有全局锁会拒绝冲突。新库首次创建后，后续重启复用同一命令和库；不要修改权重文件后复用旧库。

扩展文档神经检索服务（CPU BGE/重排，GPU 生成）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m src.service.server --reports --neural-index artifacts/jd_resume_20261008/retrieval_index --corpus data/staged/jd_reports_v1/corpus.json --snapshot jd-reports-v1 --database artifacts/jd-document-service/tasks.sqlite --port 8765
```

启动后等待 `/api/catalogue` 的 `ready: true`；日志输出至终端，任务/事件/报告在指定 SQLite。严格截止查询继续排除没有核实披露日的报告。完整原页可能仍超过输入预算，Unicode 修复不等于无限上下文；不得截断资料后沿用旧实验身份。

严格输出 TAT-QA 的维护入口为 `src.evaluation.full_benchmark --strict-tatqa-output`，需要同时显式选择历史动作适配器（原生 `--adapter artifacts/m2_action_v2/step_0040` 或既有服务 `--served-model historical_action`）。它仍运行所选 split 的全部金融任务；不能把增加提示选项后恢复旧目录，也不能混合提示做模型权重采用对比。新实验先通过上述 GPU 门槛，使用新输出目录；同设置中断续跑才加 `--resume`。原冻结实验可用各自源码 ZIP 重放，不能用修改后的源码冒充原运行指纹。

最终服务验收、测试数量和是否有运行中的进程以 [progress.md](progress.md) 为准，原始证据导航在 [result.md](result.md)。本轮代码以独立本地提交保存，尚未推送；历史 `v0.1.0-research` 不包含新增 JD 功能。
