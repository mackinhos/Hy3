# PR: Hy3 API Quickstart & Examples

## 概述

本 PR 为 Hy3 仓库新增面向开发者的 Quickstart 文档和配套 Examples 集合，帮助开发者「5 分钟跑通第一次调用、半小时上手主要能力」。

## 交付物清单

### 1. `quickstart.md` — 快速入门文档

面向开发者的完整接入指南，包含：

- **基本信息**：base_url、api_key、model name、速率限制（本地部署 + 腾讯云 API 两种方式）
- **最小可运行示例**：curl + Python OpenAI SDK
- **核心参数详解**：`temperature`、`top_p`、`max_tokens`、`stop`、`tools`、`reasoning_effort`（推理模式开关）
- **常见错误排查**：连接拒绝、超时、429 限流、404 模型不存在、401 认证失败、推理模式不生效，附错误码速查表

### 2. `examples/` — 6 个独立示例

每个示例均包含 **完整请求 + 完整响应解析 + 一段示例输出**，可直接运行（仅需配置环境变量）。

| 文件 | 示例 | 内容 |
|------|------|------|
| `01_basic_chat.py` | 基础对话 | 单轮对话 + 多轮对话 + 参数对比（temperature/max_tokens） |
| `02_streaming.py` | 流式输出 | 基础流式 + 流式推理模式 + 手动 SSE 原始数据解析 |
| `03_nonstreaming_vs_streaming.py` | 性能对比 | TTFT / 总耗时 / tokens/s 对比，含汇总报告 |
| `04_tool_calling.py` | 工具调用 | 单次工具调用 + 多轮工具循环（while 循环处理） |
| `05_reasoning_mode.py` | 推理模式 | no_think / low / high 三模式对比 + 流式推理对比 |
| `06_error_handling_retry.py` | 错误处理 | 错误类型识别 + 指数退避重试 + 降级链 + 并发限流 |

## 设计决策

### 接入方式
同时支持 **本地部署（vLLM/SGLang）** 和 **腾讯云 API** 两种方式，通过环境变量切换：

```bash
# 本地部署
export HY3_BASE_URL="http://127.0.0.1:8000/v1"
export HY3_API_KEY="EMPTY"
export HY3_MODEL="hy3"

# 腾讯云 API
export HY3_BASE_URL="https://api.hunyuan.cloud.tencent.com/v1"
export HY3_API_KEY="sk-xxx"
export HY3_MODEL="hy3-295b"
```

### 推理模式
Hy3 的推理模式通过 `extra_body.chat_template_kwargs.reasoning_effort` 控制，三种模式：
- `no_think`：直接回答（默认，最快）
- `low`：轻量思考
- `high`：深度推理（展示完整思维链）

### SDK 兼容性
所有示例基于 OpenAI Python SDK（`openai>=1.0.0`），无需额外依赖（示例 02 额外使用 `httpx` 演示原始 SSE 解析）。

## 文件结构

```
├── quickstart.md                          # 快速入门文档
├── examples/
│   ├── 01_basic_chat.py                   # 基础对话
│   ├── 02_streaming.py                    # 流式输出
│   ├── 03_nonstreaming_vs_streaming.py    # 性能对比
│   ├── 04_tool_calling.py                 # 工具调用
│   ├── 05_reasoning_mode.py               # 推理模式
│   └── 06_error_handling_retry.py         # 错误处理
└── PULL_REQUEST_TEMPLATE.md               # 本文件
```

## 验证方式

1. **语法检查**：所有 `.py` 文件通过 Python 语法检查
2. **结构检查**：每个示例包含完整请求、响应解析、示例输出注释
3. **可运行性**：配置环境变量后可直接 `python examples/0X_xxx.py` 运行

```bash
# 语法检查
python -m py_compile examples/01_basic_chat.py
python -m py_compile examples/02_streaming.py
python -m py_compile examples/03_nonstreaming_vs_streaming.py
python -m py_compile examples/04_tool_calling.py
python -m py_compile examples/05_reasoning_mode.py
python -m py_compile examples/06_error_handling_retry.py

# 运行示例（需先启动 vLLM/SGLang 服务或配置腾讯云 API）
HY3_API_KEY=EMPTY HY3_BASE_URL=http://127.0.0.1:8000/v1 HY3_MODEL=hy3 python examples/01_basic_chat.py
```

## 对应 Issue

【犀牛鸟实战issue】Hy3 API quickstart & examples

## 许可证

Apache License 2.0（与仓库一致）
