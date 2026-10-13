# 常用协作工作流

本文件保存可直接复用的项目提示词和人工流程。强制规则以根目录 `AGENTS.md` 和 `memory-bank/` 为准；这里不重复维护项目事实。

## 1. 新任务梳理

```text
完整阅读 AGENTS.md 和 memory-bank/，结合当前仓库结构说明：
1. 项目当前在做什么；
2. 本任务影响哪些文件、数据或运行入口；
3. 最小实施步骤；
4. 每步的验证命令、预期结果和进入下一步的条件。
先说明范围和方案，不要先写代码。可从仓库确认的事实直接查证；只有会改变方向或授权边界的问题才问我。
```

## 2. 完善计划

```text
阅读 memory-bank/ 全部文件和相关代码，把当前计划项细化为少量、具体、可执行的步骤。每步必须包含前置条件、验证方法和验收标准，不写代码，不制造过细子步骤。将确认后的内容更新到 plan.md；保留已完成阶段和全流程主线，不得把 plan.md 缩减成当前待办。
```

## 3. 实施单个步骤

```text
阅读 AGENTS.md、memory-bank/ 和当前工作树，只实施 plan.md 中的这一步。先告诉我修改范围和方法。执行低成本静态检查，但不要运行训练、评估、推理、抽帧或 cache warm。完成后给我一条可直接复制的完整运行命令、预期产物、进度与 checkpoint 检查方式、验收标准。依赖运行结果的后续步骤等我确认后再继续。
```

用户确认验证通过后，只更新职责发生变化的 memory-bank 文件：当前状态写 `progress.md`，本次实验无论成功、失败或中止都追加到 `results.md`，阶段状态和后续动作写 `plan.md`；架构、环境或物种边界未变化时不更新对应文件。



# 让ai解释一下对应的项目在做一件什么事情，阐明自己要根据项目做对应的改进，给出最为精简的实施步骤

# 登录GitHub仓库，将对应项目clone并push到自己的仓库中

# 创建对应memory-bank，在该目录下创建一个architecture.md ，用来解释项目中现有的整体框架：

# 生成计划
plan模式：阅读plan文件的第一步，结合/home/tiger/BaoBuu/Hungry_Bu/LLaMA_Factory/LLaMA_Factory的仓库结构，完善第一步的详细执行计划，要求每一步要小而具体。每一步都必须包含验证正确性的测试。严禁包含代码，只写清晰、具体的指令，并且尽可能直观、简短、稳健。（自己从0开始构建的项目:先聚焦于基础功能，完整功能后面再加。）
太过细致，可以将一些步骤适当合并
执行模式：认可计划的内容，但不执行计划，只需要将计划的内容加入plan.md里第一步的分点


# 阅读 /memory-bank 里所有文档，plan.md 是否完全清晰？你有哪些问题需要我澄清，让它对你来说 100% 明确？

(它通常会问 9-10 个问题。全部回答完后，让它根你的回答修改 plan.md，让计划更完善。)

# 阅读 /memory-bank 所有文档，创建一个memory-bank/results.md，只包含项目必要的实验记录

# 阅读 /memory-bank 所有文档，然后准备实施计划的第 1 步。你根据计划的内容，告诉我应该如何运行，我会负责跑通对应的步骤。在我执行和验证测试通过前，不要进行其他操作。验证通过后，打开 progress.md 记录你做了什么供后续开发者参考；再把新的架构洞察添加到 architecture.md 中解释每个文件的作用，但需要保留上一步的内容；再将对应步骤的结果记录到result.md中。注意你写文档既要清楚但不要啰嗦。

# 完成了一个工作后提交，进行下面内容

阅读 memory-bank 所有文件，阅读 progress.md 了解之前的工作进度，然后继续实施计划第 n 步。在我验证测试前不要开始下一步，另外请你提供完整的验证命令给我。
验证通过，打开 progress.md 记录你做了什么供后续开发者参考，再把新的架构洞察添加到 architecture.md 中解释每个文件的作用，但需要保留上一步的内容。注意你写文档既要清楚但不要啰嗦。

# 提升提示词效果：

加一句：“慢慢想，不着急，重要的是严格按我说的做，执行完美。如果我表达不够精确请提问。”

# 当git push出现问题
git -c http.proxy=http://127.0.0.1:6984 \
    -c https.proxy=http://127.0.0.1:6984 \
    push origin main

# vscode克隆仓库问题

VS Code 图表里也不显示官方历史，只从你的 Initial commit 开始，像你图里的 LL 仓库那样。

如果你要的是第二种，那它本质上不是普通 fork 历史，而是把官方代码“拷贝成一个全新的仓库初始提交”。否则 Git 一定会保留官方历史。

你现在更像是要第二种。操作如下。

先确认你没有要保留的未提交修改：

cd ~/BaoBuu/Hungry_Bu/FinGPT_baobu
git status

然后备份当前分支：

git branch backup/before-rebuild-main

拉取官方仓库最新代码：

git fetch upstream

从官方 upstream/main 的代码内容创建一个全新的孤儿分支：

git switch --detach upstream/main

git switch --orphan new-main
git add -A
git commit -m "Initial import from official FinGPT"

用这个新历史替换你的 main：

git branch -D main
git branch -m main

强制推送到你的 GitHub fork：

git push --force-with-lease origin main

做完后，你的 main 会变成：

Initial import from official FinGPT

之后你自己的修改就会接在这个提交后面，VS Code 图表会像你图中的 LL 仓库一样干净。

但是注意一个后果：这个 main 和官方 upstream/main 没有共同 Git 历史。以后如果你想同步官方更新，不能像普通 fork 那样正常 merge/rebase。更适合的工作流是：

git fetch upstream
git checkout main
git checkout upstream/main -- .
git add -A
git commit -m "Sync official FinGPT updates"

也就是以后把官方代码当成“文件内容来源”，而不是 Git 历史来源。

## 6. 提交

**提交粒度：只在 `plan.md` 的一个大阶段整体完成后提交一次**，阶段内的改动累积在工作区。历史形态为：初始提交 → 第 1 阶段里程碑 → 第 2 阶段里程碑 → …

```text
审阅当前全部未提交差异，区分本任务改动与已有用户改动，确认静态检查结果和未验证风险。拟定一个聚焦、祈使语气的中文 commit；只有我明确要求后才执行 commit，不自动 push。
```

仅在用户明确授权 push 后推送。只推 GitHub，不推云效（codeup）：

```bash
git push origin main
```

若阶段内已产生多个零碎提交，合并成一个里程碑（`S1` 为上一里程碑、`S2` 为当前终态）：

```bash
git add -A && git commit -m "wip：待合并"
git branch backup-squash && git tag S1 <上一里程碑> && git tag S2 HEAD
git reset --hard <基线提交>
git merge --squash S1 && git commit -m "里程碑：…"
git read-tree -u --reset S2 && git commit -m "里程碑：…"
git diff backup-squash HEAD --stat   # 必须为空，证明内容无遗漏
```

第二次不要用 `git merge --squash`：合并基是基线提交，三方合并会大面积冲突；`read-tree` 直接取终态树才对。
阅读当前目录内容，理解狗侧当前的情况，以及和我讨论一下接下来一个阶段的任务与验收标准


# 让ai解释一下对应的项目在做一件什么事情，阐明自己要根据项目做对应的改进，给出最为精简的实施步骤

# 登录GitHub仓库，将对应项目clone并push到自己的仓库中

# 创建对应memory-bank，在该目录下创建一个architecture.md ，用来解释项目中现有的整体框架：

# 生成计划
plan模式：阅读plan文件的第一步，完善第一步的详细执行计划，要求每一步要小而具体。每一步都必须包含验证正确性的测试。严禁包含代码，只写清晰、具体的指令，并且尽可能直观、简短、稳健。（自己从0开始构建的项目:先聚焦于基础功能，完整功能后面再加。）
太过细致，可以将一些步骤适当合并
执行模式：认可计划的内容，但不执行计划，只需要将计划的内容加入plan.md里第一步的分点


# 阅读 /memory-bank 里所有文档，plan.md 是否完全清晰？你有哪些问题需要我澄清，让它对你来说 100% 明确？

(它通常会问 9-10 个问题。全部回答完后，让它根你的回答修改 plan.md，让计划更完善。)

# 阅读 /memory-bank 所有文档，创建一个memory-bank/results.md，只包含项目必要的实验记录

# 阅读 /memory-bank 所有文档，然后准备实施计划的第 1 步。你根据计划的内容，告诉我应该如何运行，我会负责跑通对应的步骤。在我执行和验证测试通过前，不要进行其他操作。验证通过后，打开 progress.md 记录你做了什么供后续开发者参考；再把新的架构洞察添加到 architecture.md 中解释每个文件的作用，但需要保留上一步的内容；再将对应步骤的结果记录到result.md中。注意你写文档既要清楚但不要啰嗦。

# 完成了一个工作后提交，进行下面内容

阅读 memory-bank 所有文件，阅读 progress.md 了解之前的工作进度，然后继续实施计划第 n 步。在我验证测试前不要开始下一步，另外请你提供完整的验证命令给我。
验证通过，打开 progress.md 记录你做了什么供后续开发者参考，再把新的架构洞察添加到 architecture.md 中解释每个文件的作用，但需要保留上一步的内容。注意你写文档既要清楚但不要啰嗦。

# 提升提示词效果：

加一句：“慢慢想，不着急，重要的是严格按我说的做，执行完美。如果我表达不够精确请提问。”

cd C:\omni_videos
python -m http.server 8765 --bind 127.0.0.1

cpolar http 8765
