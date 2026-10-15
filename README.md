# Finagent

研报发布前质控预审 Agent：把稿件里的数字、日期和出处标注逐条对回原始披露文件，标出不一致之处并生成工作底稿。

## 快速使用

```bash
# 审核页面（需 .env 中的 DEEPSEEK_API_KEY；本机 9B 模型见 environment.md）
.venv/bin/python -m src.service.app --library data/staged/library_v8/library.sqlite --data-dir runs/web --budget-usd 1
# 命令行：对评测集中的一篇稿件生成工作底稿
.venv/bin/python -m src.agent.graph --library data/staged/library_v8/library.sqlite \
  --reviews data/staged/eval_v2/development_reviews.jsonl --review-id <id> --output runs/demo/workpaper.json --budget-usd 0.5
```

## 文档

- [定位、设计决定与阶段计划](memory-bank/plan.md)
- [当前进度与阻塞](memory-bank/progress.md)
- [架构与模块职责](memory-bank/architecture.md)
- [环境恢复与外部资产](memory-bank/environment.md)
- [结果记录](memory-bank/result.md)
- [协作规则](AGENTS.md)
