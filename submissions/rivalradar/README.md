# RivalRadar 竞品雷达 — Hy3 驱动的多 Agent 竞品分析协作系统

> 犀牛鸟实战 Issue：**Build a vibe-coded application powered by Hy3**
> 选题：挑战课题「AI 驱动的竞品分析 Agent 协作系统」
> **独立仓库（完整代码 / 运行指南 / Demo）：https://github.com/mackinhos/rivalradar**

## 项目说明

输入一个调研主题，一支由 **4 个 Hy3 Agent** 组成的「数字调研小组」自动开工：
拆解调研方案 → 并行检索公开信息 → 结构化交叉对比 → 流式撰写竞品分析报告。
全程通过 **Hy3 API（OpenAI 兼容协议）** 完成推理，不训练、不微调、不本地部署模型。

### 功能特性

- 🧭 **多 Agent 协作**：规划师 / 调研员 / 分析师 / 撰稿人各司其职，前端卡片状态实时流转
- 🔍 **公开信息采集**：调研员按「竞品 × 维度」多路检索（ddgs），证据逐条溯源；断网自动降级为模型内置知识并明确标注
- 📊 **结构化产出**：维度 × 竞品评分矩阵、雷达图、优劣势画像、总体结论
- 📝 **流式报告**：Markdown 报告边生成边渲染，一键下载 / 复制
- 🛡️ **工程健壮性**：JSON 输出自动纠偏重试、指数退避重试、单竞品失败容错
- 🔌 **零密钥可体验**：未配置 Key 时进入演示模式，完整回放全链路交互

### 架构与 Hy3 承担的角色

```
用户输入（主题/竞品/维度）
        │
        ▼
┌────────────────────────────────────────────────────┐
│  Orchestrator（FastAPI + SSE 事件流）               │
│                                                    │
│  ① 规划师 Plan    ── Hy3: 拆解方案，输出 JSON 计划  │
│  ② 调研员 Scout   ── Web 搜索 + Hy3: 逐竞品提炼     │
│     （3 路并行）      资料卡片（证据→要点+来源）      │
│  ③ 分析师 Insight ── Hy3: 交叉对比，输出评分矩阵    │
│                     与优劣势 JSON                   │
│  ④ 撰稿人 Report  ── Hy3: 流式生成 Markdown 报告    │
└────────────────────────────────────────────────────┘
        │ SSE (agent_start / log / plan / card / comparison / report_chunk)
        ▼
   Web 前端（单页：Agent 状态机 + 实时日志 + 雷达图 + 报告渲染）
```

**Hy3 承担系统中全部智能环节**（4 类调用）：

| 环节 | Hy3 调用 | 输出约束 |
| --- | --- | --- |
| 调研规划 | chat_json | JSON：竞品清单 / 维度 / 关键问题 |
| 信息提炼 | chat_json（每竞品 1 次，并行） | JSON：概述 / 要点 / 来源 |
| 对比分析 | chat_json | JSON：评分矩阵 / 优劣势 / 结论 |
| 报告撰写 | chat_stream | Markdown 流式输出 |

工程侧另利用 Hy3 完成 **JSON 格式自愈**（解析失败时模型自我修正后重解析）。
调用参数遵循官方推荐：`temperature=0.9, top_p=1.0`，思考模式经
`extra_body.chat_template_kwargs.reasoning_effort` 控制。

### 端到端 Demo 流程

1. **Demo A（指定对象）**：主题「国内主流 AI 编程助手竞品分析」+ 竞品
   `GitHub Copilot, Cursor, 通义灵码, CodeBuddy` → 4 个 Agent 接力，
   产出对比矩阵 + 雷达图 + 约 1500 字报告。
2. **Demo B（自动识别）**：主题「中国新能源 SUV 市场竞品分析」，竞品留空 →
   规划师自动识别理想 L6 / 问界 M7 / Model Y / 比亚迪唐等对象并跑通全链路。

（演示 GIF / 视频见独立仓库 README，单个流程 ≤ 2 min。）

### 快速开始

```bash
git clone https://github.com/mackinhos/rivalradar.git
cd rivalradar
cp .env.example .env   # 填入 HY3_API_KEY
pip install -r requirements.txt
python run.py          # http://127.0.0.1:8000
```
