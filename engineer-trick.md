# CT-COEVO Engineering Tricks & Architecture Guide

> **核心原则**：我（开发者）在调试 agent 的过程中积累经验，这些经验通过 write_reflection 写入 memory store，成为 agent 的进化记忆。禁止 hardcode，禁止冷启动种子，一切通过进化产生。
>
> **prompts.py**：严格按论文 Listing 1-11，不添加任何工程技巧。
>
> **memory store**：我调试中发现的通用工程技巧，通过正常记忆机制写入。

---

## 1. 核心架构：4 变量干净上下文

每步传给 Agent LLM 的输入由 4 部分组成，**不累积历史消息**：

```
messages = [
    system_prompt,      # 身份 + description.md全文 + 设备信息(GPU/CPU/RAM) + 工具列表摘要
    tried_tools_table,  # 已尝试的所有tool+超参数+结果 (由Memory LLM维护)
    current_output,     # 本步工具执行结果 + 分数反馈
    memory_plan,        # Memory LLM生成的本轮规划(包含战略方向+本轮insight)
]
```

**关键原则**：
- **system_prompt 永不截断**：包含完整 description.md、任务身份、设备信息
- **历史不累积**：每步重新组装 messages，通过 Memory LLM 精炼来压缩历史
- **tried_tools_table 是全局视角**：记录所有尝试过的 tool + 参数 + metric，指导全局方向
- **memory_plan 是当前视角**：Memory LLM 根据本轮情况生成的下一步规划

---

## 2. Memory LLM 的职责

Memory LLM 是一个独立的 API 调用，负责：

**输入**：本轮 system_prompt + 旧 history_summary + 本轮 agent_output + 本轮 tool_outputs

**输出**（结构化）：
```
=== PLAN ===
(下一步的战略规划，包含具体要做什么)

=== INSIGHT ===
(本轮遇到的问题、解决方案、取得的改进、关键数据)

=== TRIED_TOOLS_UPDATE ===
tool_name(param1=val1, param2=val2) -> metric=0.xxxxx
```

**精炼规则**：
- history_summary = LLM(旧 history_summary + 本轮输入输出)
- 不截断，用 LLM 来精炼压缩
- 输出中提取 TRIED_TOOLS_UPDATE 追加到 tried_tools_table
- tried_tools_table 超长时保留最近的条目

---

## 3. Tool 系统设计

### 3.1 Abstract Tool 定义

Tool 必须是**算法工具**，不是任意代码：
- 专注推荐系统算法（DeepFM, BPR, LightGCN, SASRec, XGBoost 等）
- 跨任务通用（不 hardcode 数据集特定逻辑）
- 统一接口：接收 DATA_DIR + CONFIG，输出 submission.csv + report

### 3.2 Multi-Tool Call 并行训练

Agent 可以在一轮中调用多个 tool，**串行执行**（从前往后）：
```python
for tool_call in message.tool_calls:
    result = execute(tool_call)  # 等待完成
    results.append(result)
# 打包所有结果回传
```

**GPU 并行策略**：
- System prompt 中列出所有 GPU 设备及当前状态
- Agent 自主决策如何分配 GPU（如 GPU 0,1 跑 DeepFM，GPU 2,3 跑 LightGCN）
- 鼓励一次调用多个不同模型的 tool，充分利用 4 卡

### 3.3 Tool Review（审核 LLM）

每个新 tool / tool edit 必须经过 Review LLM 审核：
- **输入**：tool 源代码
- **检查**：1) 是否危险（安全） 2) 是否是跨任务先进推荐系统算法（有用）
- **输出**：ACCEPT / REJECT + 原因
- Review LLM 使用 DeepSeek-V3.2-Exp

### 3.4 RecBole 集成

RecBole 是推荐系统的标准框架，支持 70+ 模型：
- Agent 编写 YAML 配置 -> 调用 run_model tool -> RecBole 训练 -> 输出 submission
- 支持多 GPU（torchrun 分布式）
- 模型包括：BPR, NeuMF, LightGCN, NGCF, SASRec, GRU4Rec, BERT4Rec, DeepFM, DCN, DIN 等

---

## 4. 训练策略

### 4.1 大配置原则

- **Epochs 要大**：200+ epochs，不要浪费时间在交互上
- **Batch size 要大**：4096+
- **Embedding size 要大**：128+
- **训练时间要长**：宁可一步训 30 分钟训出好模型，也不要 10 步各训 3 分钟的烂模型

### 4.2 模型复用

- 每个训练好的模型保存 checkpoint，命名包含 metric
- 后续步基于最优 checkpoint 继续训练（fine-tune），不要从头训
- submissions/ 文件夹维护所有提交，submission.csv 始终指向最优

### 4.3 多样性策略

Agent 应该尽可能尝试多种模型族：
- **矩阵分解**：BPR, NMF, SVD++
- **深度学习**：DeepFM, DCN, AutoInt, FiBiNET
- **图模型**：LightGCN, NGCF, GCMC
- **序列模型**：SASRec, GRU4Rec, BERT4Rec
- **树模型**：LightGBM, XGBoost（CTR 任务）
- **集成**：多模型融合

具体选择由 LLM 根据数据集特征自主决策。

---

## 5. 评分与反馈

### 5.1 Agent 可见的反馈

每步 Agent 可以看到：
- 提交是否成功（格式验证）
- 提交是否更新（hash 比较）
- 剩余时间
- **看不到官方分数**（competition-style）

### 5.2 评分流程

- 每步调用官方 metric.py 评测（内部记录，不展示给 agent）
- 最终进入 Reflection 阶段时，将 step score 告知 agent
- DA-Code: (raw - baseline) / (best - baseline)

### 5.3 Train vs Test 模式

| 维度 | Train (EvoSet) | Test (EvalSet) |
|------|---------------|----------------|
| Memory | 可进化，全局写入 | 冻结，仅本地 |
| Tool | 可进化，全局写入 | 冻结，仅本地 |
| 最后阶段 | Critical Reflection + Global Optimize | 直接结束 |
| 分数可见性 | Reflection 阶段可见 | Reflection 阶段可见 |

---

## 6. Logging 规范

### 6.1 Concise Log

每步记录：
```
step N intoken=XXX outoken=XXX submit_ok=Yes/No submit_updated=Yes/No step-score=N/A elapsed=XXXs remaining=XXXs reason=XXX
```

最终记录：
```
====================================
Final Benchmark Summary: {dataset_name}
Raw Score: {score}
DACode Mapped Score [0,1]: {da_code}
Total InTokens: ...
Total OutTokens: ...
Total Tokens: In+Out
Total steps: ...
Total Wall Time: ...s
====================================
```

### 6.2 Message Log

JSONL 格式，记录每步完整输入输出，包括：
- Agent 的 tool_calls 和 arguments（完整，不截断）
- Tool 执行结果（可截断到 TOOL_OUTPUT_MAX_CHARS=50000）
- Memory LLM 的精炼输出
- 分数反馈

### 6.3 Memory Log

Markdown 格式，每步记录：
- PLAN: 当前战略
- INSIGHT: 本轮发现
- TRIED_TOOLS_UPDATE: tool+参数+结果

---

## 7. 反对 Hardcode

**绝对禁止的 hardcode**：
- 数据集特定的列名映射
- 固定的模型选择（应由 LLM 决策）
- 硬编码的截断规则（用 LLM 精炼代替）
- 硬编码的 tool 审核规则（用 Review LLM 代替）
- 硬编码的路径拼接

**允许的配置**：
- 全局变量（TOOL_OUTPUT_MAX_CHARS, HISTORY_MAX_CHARS 等）
- API URL 和 Key
- GPU 设备列表

---

## 8. Cold Start 设计

### 8.1 Memory Cold Start (memory_seed.md)

包含推荐系统领域的结构化经验：
- 数据预处理最佳实践
- 模型选择指南（按任务类型）
- 常见失败模式和修复策略
- 训练技巧（learning rate, batch size, negative sampling）
- 提交格式注意事项

### 8.2 Tool Cold Start (tools/)

包含种子工具：
- RecBoleRunner: 通用 RecBole 训练工具
- 支持 70+ 模型的 YAML 配置驱动训练
- 多 GPU 分布式训练

---

## 9. 性能优化

### 9.1 Token 优化

- System prompt 固定，不追加变长
- History 用 LLM 精炼，不用截断
- Tool output 截断到 50000 chars（这是唯一允许的截断）
- 鼓励 Agent 输出详细（outtoken 有价值），减少无意义的探索

### 9.2 时间优化

- Multi-tool call 减少轮次
- 大配置减少重复训练
- 模型复用减少从头训练
- 4 GPU 并行训练

### 9.3 磁盘优化

- Workspace 使用 symlink 链接到 public 数据（节省磁盘）
- 每个任务独立 workspace，结束后清理
- submissions/ 文件夹管理所有提交版本

---

## 10. 已知 Bug 和修复

| 严重度 | 位置 | 描述 | 状态 |
|--------|------|------|------|
| 高 | grader.py | answer_candidates 缺少 data/private/ | ✅ 已修复 |
| 高 | __init__.py | openai import 失败导致崩溃 | ✅ 已修复 |
| 中 | agent.py | 前几步浪费在 list_tools 探索 | 需优化 prompt |
| 中 | agent.py | Agent 不调用 run_tool | 需优化 prompt |

---

---

## 11. Prompt 优化经验（mimo-v2.5-pro 适配）

### 11.1 问题诊断

**症状**：agent 15 步中 5 步调 `list_tools`，从未调用 `run_tool`，最终 score=0.18

**根因分析**：
- system prompt 包含 12 个工具说明 + 多层指令（STRATEGY, RULES, MODEL DIVERSITY, TRAINING PRINCIPLES 等）
- mimo-v2.5-pro 指令遵循能力弱于 DeepSeek-V3.2，无法处理复杂 prompt
- 工具说明过多导致 agent 困惑，不知道该用哪个

### 11.2 优化策略

**核心原则**：**少即是多**（Less is More）

| 维度 | v1（复杂） | v2（简化） |
|------|-----------|-----------|
| 工具说明 | 12 个工具全部详细说明 | 只强调 run_tool，其他一行带过 |
| 指令层数 | 7+ 层（STRATEGY, RULES, DIVERSITY, PRINCIPLES...） | 3 层（GOAL, TOOLS, RULES） |
| 第一步指令 | "IMMEDIATELY call run_tool"（嵌在大量文本中） | 开头第一行就是 "Call list_tools(), then run_tool()" |
| GPU 限制 | "ONLY use GPU-accelerated models" | 移除（agent 不遵循，反而困惑） |
| 模型列表 | 长列表 + 禁止列表 | 简短列表，无禁止 |

### 11.3 关键发现

1. **mimo-v2.5-pro 对复杂 prompt 的遵循率极低**：超过 5 层指令基本无效
2. **示例比规则更有效**：给出具体的 `run_tool(...)` 示例比说"YOU MUST call run_tool"更有效
3. **第一步指令至关重要**：agent 的第一步行为决定后续走向，必须在最显眼位置
4. **禁止指令适得其反**：说"NEVER use CPU models"反而让 agent 困惑，不如直接列出推荐模型
5. **kickoff message 要极度简洁**：3 行指令 + 1 个示例，不要超过 10 行

### 11.4 优化前后对比

**优化前（v1）**：
```
step 1: list_tools
step 2: list_tools, bash
step 3: bash, bash
step 4: python
step 5: python
...
step 12: python (score=0.18)
Final: score=0.18
```

**优化后（v2）**（预期）：
```
step 1: list_tools, run_tool (BPR)
step 2: run_tool (DeepFM), run_tool (LightGCN)
step 3: run_tool (SASRec)
...
Final: score > 0.5
```

### 11.5 第二轮优化：kickoff message 嵌入 tool_id

**发现**：即使简化了 prompt，agent 仍反复调 `list_tools` 而不调 `run_tool`

**根因**：mimo-v2.5-pro 需要 `list_tools()` 返回结果后才能知道 tool_id，但它拿到结果后仍然不调 `run_tool`——说明它不理解 "看完 tool list 后下一步该做什么"

**修复**：在 kickoff message 中直接嵌入 tool_id：
```python
def build_kickoff_message(tool_id="e4a22524ac6c0f53"):
    return (
        "START NOW. Call run_tool IMMEDIATELY.\n\n"
        f"Available tool:\n  tool_id: {tool_id}\n"
        f'  run_tool(tool_id="{tool_id}", config={{"model_name": "BPR", "epochs": 100}})\n'
    )
```

**效果**：
- ✅ Step 1 终于调了 `run_tool`！训练了 342 秒（BPR 100 epochs）
- ❌ Step 2-4 又退回 `list_tools`（训练完了不知道下一步该做什么）

### 11.6 第三轮诊断：run_tool 训练成功但没生成 submission

**测试结果**（v2 prompt + tool_id 嵌入 kickoff）：
```
step 1: run_tool (333s 训练) → score=N/A (无 submission)
step 2-8: list_tools (浪费)
step 9: run_tool (9s, 快速失败)
step 10: list_tools
Final: score=None
```

**核心问题**：DistributedModelRunner 训了 333 秒但没生成 submission.csv

**可能原因**：
1. TorchEasyRec 输出路径不在 workspace 目录
2. 模型训练完成后 submission 格式转换失败
3. 工具的 `train_and_predict` 返回时 submission.csv 不在 cwd

**下一步**：
- [ ] 检查 DistributedModelRunner 源码，查看 submission 生成逻辑
- [ ] 检查 run_tool 的 stdout/stderr 输出
- [ ] 修复 DistributedModelRunner 确保 submission.csv 生成在 workspace
- [ ] 在 Memory LLM plan 中嵌入 "call run_tool with different model" 指令

### 11.7 优化总结

| 轮次 | 优化内容 | 效果 |
|------|---------|------|
| v1 | 原始复杂 prompt | agent 只调 list_tools，score=0.0 |
| v2 | 简化 prompt | agent 仍调 list_tools，score=0.18（用 python） |
| v3 | kickoff 嵌入 tool_id | ✅ step 1 调 run_tool（333s），但无 submission |
| v4 | 修复 DistributedModelRunner | 待验证 |

**关键教训**：
- mimo-v2.5-pro 指令遵循极弱，必须在 kickoff message 中直接给出完整命令
- 工具本身的 bug 比 prompt 问题更致命
- 优化顺序：先修工具 → 再优化 prompt

### 11.8 第四轮诊断：DistributedModelRunner 训练成功但不生成 submission

**测试结果**（v3 prompt + correct model names）：
```
step 1: run_tool (deepfm, 191s) → score=N/A (无 submission)
step 2-10: list_tools (浪费)
Final: score=N/A
```

**核心问题**：DistributedModelRunner (TorchEasyRec) 训了 191 秒但没生成 submission.csv

**可能原因**：
1. TorchEasyRec (`tzrec`) 可能没安装或版本不对
2. 训练过程中可能报错但被静默吞掉
3. 输出路径不在 workspace 目录
4. 训练完成后 submission 格式转换失败

**下一步**：
- [ ] 检查 TorchEasyRec 是否安装：`python -c "import tzrec; print(tzrec.__version__)"`
- [ ] 检查 run_tool 的 stdout/stderr 输出
- [ ] 如果 TorchEasyRec 不可用，改用 RecBole 作为工具
- [ ] 考虑用 `python` 工具直接写训练代码（绕过 DistributedModelRunner）

### 11.9 总结：优化历程

| 轮次 | 优化内容 | 效果 | 核心问题 |
|------|---------|------|---------|
| v1 | 原始复杂 prompt | score=0.0 | agent 只调 list_tools |
| v2 | 简化 prompt | score=0.18 | agent 用 python 写代码 |
| v3 | kickoff 嵌入 tool_id | step 1 调 run_tool | 工具不支持 BPR |
| v4 | 修正模型名为 deepfm | step 1 训练 191s | 工具不生成 submission |
| v5 | 待验证 | — | — |

**核心发现**：
1. **Prompt 优化有效但有限**：kickoff 嵌入 tool_id 是关键突破
2. **工具 bug 是最大瓶颈**：DistributedModelRunner 不生成 submission
3. **Agent 行为模式**：step 1 能调 run_tool，但后续步骤退回 list_tools
4. **需要修复工具**：要么修 DistributedModelRunner，要么换 RecBole

---

### 11.10 mimo-v2.5-pro 不遵循 create_tool 指令

**现象**：prompt 明确说 "Call create_tool (id=xxx)" 但 LLM 返回 bash 的 ID

**测试**：
- Prompt: "Call create_tool (id=tool_1780763698112_2) with name, code, description"
- LLM 返回: `[{"tool_id": "tool_1780583844094", "code": "ls -la"}]`（bash 的 ID）

**根因**：mimo-v2.5-pro 不理解 create_tool 的语义，总是选择 bash（因为它更"简单"）

**教训**：
- mimo-v2.5-pro 不适合做 tool selection agent
- 需要更强的模型（DeepSeek-V3.2）或者更直接的干预方式
- 或者需要在 prompt 中用更具体的示例说明 create_tool 的用法

### 11.11 上下文注入问题

**发现**：prompt 中的 HISTORY 和 PREVIOUS STEP OUTPUT 没有注入到 LLM

**原因**：
1. `_log_message` 截断到 2000 chars，看不到完整 prompt
2. 代码中 `history_text` 和 `prev_output_section` 变量定义在 prompt 之前，但 f-string 没有正确插值

**修复**：
1. 移除 `_log_message` 中的截断
2. 确保 `history_text` 和 `prev_output_section` 在 prompt 构建之前定义

**教训**：
- 日志截断会隐藏 bug（看不到完整 prompt）
- f-string 插值需要变量在作用域内
- 验证 prompt 时要检查完整内容，不能只看日志

*Last updated: 2026-06-07*
