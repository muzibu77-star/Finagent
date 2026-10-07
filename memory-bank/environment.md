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

若需要重建数据，依次运行 `src.data.fetch_sources`、`src.data.build_manifest`、`src.data.validate_manifest`，参数及原始快照见 [架构](architecture.md) 和 [结果](result.md)。M3 服务输入为 `data/staged/m3_acceptance_v1/corpus.json`；重建入口是 `src.evaluation.freeze_acceptance`。训练资产恢复须包括已选择的 `step_0040`，不能把基础模型当作适配器恢复。

## 服务与状态

```bash
.venv/bin/python -m src.service.server --port 8765
```

本机地址 `http://127.0.0.1:8765`；远程机器使用端口转发。默认数据库 `artifacts/service/tasks.sqlite`，默认固定单题流程；同一数据库绑定资料快照，换资料须用新数据库；更换模型或适配器版本时也应新建数据库，不能混用实验状态。网页刷新会重连事件，取消传到生成，澄清使用原任务的新修订。`Ctrl+C` 请求停止当前任务并释放模型；崩溃后使用相同命令重启，已提交报告不重复生成。未完成的生成调用保留消耗记录，不擅自重置预算。

原始 PDF 文档库入口（单独数据库；数值仍须复核）：

```bash
.venv/bin/python -m src.service.server --reports --corpus data/staged/reports_v2/corpus.json --snapshot reports-v1 --database artifacts/report-service/tasks.sqlite --port 8765
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
