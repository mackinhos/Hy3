# Hy3 API Quickstart

> **目标**：5 分钟跑通第一次调用，半小时上手主要能力。

Hy3 是腾讯混元团队开发的 295B 参数 MoE（Mixture-of-Experts）大语言模型，激活参数 21B，支持 256K 上下文，兼容 OpenAI API 规范。本指南帮助你快速接入并使用 Hy3 API。

---

## 目录

- [1. 基本信息](#1-基本信息)
- [2. 环境准备](#2-环境准备)
- [3. 5 分钟跑通第一次调用](#3-5-分钟跑通第一次调用)
- [4. 核心参数详解](#4-核心参数详解)
- [5. 流式输出](#5-流式输出)
- [6. 推理模式（Thinking Mode）](#6-推理模式thinking-mode)
- [7. 工具调用（Tool Calling）](#7-工具调用tool-calling)
- [8. 常见错误排查](#8-常见错误排查)
- [9. 进阶示例导航](#9-进阶示例导航)

---

## 1. 基本信息

### 1.1 两种接入方式

Hy3 提供两种接入方式，API 调用格式完全一致（OpenAI 兼容），只需替换 `base_url`、`api_key` 和 `model`：

| 配置项 | 本地部署（vLLM / SGLang） | 腾讯云 API |
|--------|--------------------------|------------|
| **base_url** | `http://127.0.0.1:8000/v1` | `https://api.hunyuan.cloud.tencent.com/v1` |
| **api_key** | `EMPTY`（任意非空字符串） | 在[混元控制台](https://console.cloud.tencent.com/hunyuan/start)创建，`sk-` 开头 |
| **model** | `hy3` | `hy3-295b` |
| **完整端点** | `http://127.0.0.1:8000/v1/chat/completions` | `https://api.hunyuan.cloud.tencent.com/v1/chat/completions` |

> **本地部署**：参考仓库 [Deployment](README.md#deployment) 章节，使用 vLLM 或 SGLang 启动服务。启动命令中通过 `--served-model-name hy3` 指定模型名。
>
> **腾讯云 API**：无需自行部署，在控制台创建 API Key 后即可调用。详见[混元 OpenAI 兼容接口文档](https://cloud.tencent.com/document/product/1729/111007)。

### 1.2 模型规格

| 属性 | 值 |
|------|-----|
| 架构 | Mixture-of-Experts (MoE) |
| 总参数量 | 295B |
| 激活参数量 | 21B |
| 上下文长度 | 256K |
| 支持精度 | BF16 |
| 词汇表大小 | 120,832 |

### 1.3 速率限制

| 限制类型 | 本地部署 | 腾讯云 API |
|----------|----------|------------|
| 并发数 | 取决于 GPU 数量和显存 | 默认 5 个并发（主子账号共享） |
| TPM（每分钟 Token 数） | 取决于硬件 | 按套餐等级分配 |
| 超时建议 | 60-180 秒 | 60-120 秒 |

> **提示**：Hy3 是千亿参数模型，推理速度比 7B/14B 小模型慢。建议将 HTTP 超时设置为至少 60 秒，长文本场景建议 120 秒以上。

---

## 2. 环境准备

### 2.1 安装 Python SDK

```bash
pip install openai
```

> Hy3 兼容 OpenAI Python SDK（`openai>=1.0.0`），无需安装额外依赖。

### 2.2 配置 API Key

**本地部署**——无需真实 Key，任意非空字符串即可：

```bash
export HY3_API_KEY="EMPTY"
export HY3_BASE_URL="http://127.0.0.1:8000/v1"
export HY3_MODEL="hy3"
```

**腾讯云 API**——在控制台创建 Key 后设置：

```bash
export HY3_API_KEY="sk-your-api-key-here"
export HY3_BASE_URL="https://api.hunyuan.cloud.tencent.com/v1"
export HY3_MODEL="hy3-295b"
```

Windows PowerShell：

```powershell
$env:HY3_API_KEY = "sk-your-api-key-here"
$env:HY3_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"
$env:HY3_MODEL = "hy3-295b"
```

---

## 3. 5 分钟跑通第一次调用

### 3.1 curl 方式

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "hy3",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍你自己。"}
    ],
    "temperature": 0.9,
    "top_p": 1.0
  }'
```

腾讯云 API 只需替换 URL、Key 和模型名：

```bash
curl https://api.hunyuan.cloud.tencent.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HY3_API_KEY" \
  -d '{
    "model": "hy3-295b",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍你自己。"}
    ],
    "temperature": 0.9,
    "top_p": 1.0
  }'
```

### 3.2 Python OpenAI SDK 方式

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "EMPTY"),
    base_url=os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1"),
)

response = client.chat.completions.create(
    model=os.environ.get("HY3_MODEL", "hy3"),
    messages=[
        {"role": "user", "content": "你好，请用一句话介绍你自己。"}
    ],
    temperature=0.9,
    top_p=1.0,
)

print(response.choices[0].message.content)
```

**预期输出**：

```
你好！我是 Hy3，由腾讯混元团队开发的大语言模型，擅长中文理解、代码生成和复杂推理任务。
```

**完整响应结构**（便于理解返回格式）：

```json
{
  "id": "chatcmpl-xxxxx",
  "object": "chat.completion",
  "created": 1722400000,
  "model": "hy3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！我是 Hy3，由腾讯混元团队开发的大语言模型……"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 35,
    "total_tokens": 47
  }
}
```

---

## 4. 核心参数详解

### 4.1 参数速查表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 0.9 | 采样温度，控制输出随机性。值越高越发散，越低越确定 |
| `top_p` | float | 1.0 | 核采样概率阈值，与 temperature 二选一调节 |
| `max_tokens` | int | 无上限 | 生成 Token 数上限 |
| `stop` | string / array | null | 停止序列，遇到时立即停止生成 |
| `stream` | bool | false | 是否流式输出 |
| `tools` | array | null | 工具/函数定义列表，启用 Function Calling |
| `reasoning_effort` | string | `"no_think"` | 推理模式：`no_think`（直接回答）、`low`（轻量思考）、`high`（深度推理） |

### 4.2 temperature

控制生成随机性。**推荐值 `0.9`**。

```python
# 低温度：输出稳定、确定性强（适合代码生成、事实问答）
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "用 Python 写一个快速排序"}],
    temperature=0.1,
)

# 高温度：输出多样、创意性强（适合创意写作、头脑风暴）
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "写一首关于秋天的诗"}],
    temperature=1.2,
)
```

### 4.3 top_p

核采样（Nucleus Sampling）。**推荐值 `1.0`**（不限制），通过 `temperature` 调节即可。

```python
# top_p=0.1：只从概率前 10% 的 Token 中采样，输出更保守
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "解释什么是递归"}],
    temperature=0.9,
    top_p=0.1,
)
```

> **注意**：建议只调节 `temperature` 或 `top_p` 中的一个，不要同时调节。

### 4.4 max_tokens

限制生成 Token 数量，防止输出过长导致超时或超额。

```python
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "详细介绍 Python 的 GIL"}],
    max_tokens=2048,  # 最多生成 2048 个 Token
)
```

### 4.5 stop

自定义停止序列。当输出中出现这些字符串时，模型立即停止生成。

```python
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "列出三种排序算法，用 ### 分隔每种算法"}],
    stop="###",  # 遇到 ### 即停止
)
```

也支持多个停止序列：

```python
stop=["###", "END", "<stop>"]
```

### 4.6 tools（工具调用）

传入函数定义，让模型决定是否调用外部工具。详见 [第 7 节](#7-工具调用tool-calling) 和 [示例 04](examples/04_tool_calling.py)。

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
)
```

### 4.7 reasoning_effort（推理模式开关）

Hy3 独有的推理模式控制，通过 `extra_body` 传入：

```python
# 直接回答（默认，不展示思考过程）
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "1+1等于几？"}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
)

# 深度推理（展示完整思维链，适合数学/编程/逻辑推理）
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "证明根号2是无理数"}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
)
```

| 值 | 适用场景 | 特点 |
|----|----------|------|
| `"no_think"` | 日常问答、翻译、简单任务 | 不生成思考过程，响应快 |
| `"low"` | 中等复杂度任务 | 轻量思考，平衡速度与质量 |
| `"high"` | 数学证明、代码调试、复杂推理 | 完整思维链，质量最高但耗时更长 |

---

## 5. 流式输出

流式输出（Server-Sent Events）让 Token 逐块返回，大幅改善首字延迟体验：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "EMPTY"),
    base_url=os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1"),
)

stream = client.chat.completions.create(
    model=os.environ.get("HY3_MODEL", "hy3"),
    messages=[{"role": "user", "content": "用 200 字介绍量子计算"}],
    stream=True,  # 开启流式
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

> 完整流式示例见 [examples/02_streaming.py](examples/02_streaming.py)。

---

## 6. 推理模式（Thinking Mode）

开启 `reasoning_effort: "high"` 后，响应中的 `reasoning_content` 字段包含模型的思维链：

```python
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "一个水池有两个进水管……（略）"}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
)

# 思考过程
if hasattr(response.choices[0].message, "reasoning_content"):
    print("【思考过程】")
    print(response.choices[0].message.reasoning_content)

# 最终回答
print("\n【最终回答】")
print(response.choices[0].message.content)
```

> 完整对比示例见 [examples/05_reasoning_mode.py](examples/05_reasoning_mode.py)。

---

## 7. 工具调用（Tool Calling）

Hy3 支持 OpenAI 兼容的 Function Calling。完整流程：

1. 定义工具 → 2. 发送请求 → 3. 模型返回工具调用 → 4. 执行工具 → 5. 将结果回传 → 6. 模型生成最终回答

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "EMPTY"),
    base_url=os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1"),
)

# 1. 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 2+3*4"}
                },
                "required": ["expression"]
            }
        }
    }
]

# 2. 发送请求
messages = [{"role": "user", "content": "帮我算一下 (15 + 27) * 3"}]
response = client.chat.completions.create(
    model="hy3",
    messages=messages,
    tools=tools,
)

# 3. 检查模型是否请求调用工具
msg = response.choices[0].message
if msg.tool_calls:
    messages.append(msg)
    for call in msg.tool_calls:
        # 4. 执行工具
        expr = call.function.arguments  # 模型填充的参数
        print(f"模型请求调用: {call.function.name}({expr})")
        result = eval(expr)  # 实际项目中替换为你的函数

        # 5. 将结果回传
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": str(result),
        })

    # 6. 模型生成最终回答
    final = client.chat.completions.create(
        model="hy3",
        messages=messages,
        tools=tools,
    )
    print(final.choices[0].message.content)
```

> 完整工具调用示例（含多轮循环）见 [examples/04_tool_calling.py](examples/04_tool_calling.py)。

---

## 8. 常见错误排查

### 8.1 连接被拒绝（Connection Refused）

```
ConnectionError: Connection refused.
```

**原因**：本地 vLLM/SGLang 服务未启动，或端口不对。

**解决**：
1. 确认服务已启动：`curl http://127.0.0.1:8000/v1/models`
2. 检查端口是否被占用：`lsof -i :8000`
3. 确认 `base_url` 端口与启动命令 `--port` 一致

### 8.2 请求超时（Timeout）

```
requests.exceptions.Timeout: Request timed out
```

**原因**：千亿参数模型推理较慢，默认超时不够。

**解决**：
```python
client = OpenAI(
    api_key="...",
    base_url="...",
    timeout=120.0,  # 设置 120 秒超时
    max_retries=2,   # 自动重试 2 次
)
```

### 8.3 频率限制（429 Too Many Requests）

```
RateLimitError: 429 Too Many Requests
```

**原因**：超过并发数或 TPM 限制。

**解决**：
1. 降低请求频率，增加请求间隔
2. 使用指数退避重试（见 [示例 06](examples/06_error_handling_retry.py)）
3. 腾讯云用户可在控制台申请提升配额

### 8.4 模型名称错误（404 Model Not Found）

```
NotFoundError: Model hy3 not found
```

**原因**：`model` 参数与服务端注册的模型名不匹配。

**解决**：
- 本地部署：确认 `--served-model-name` 参数值
- 腾讯云：使用 `hy3-295b`，通过 `GET /v1/models` 查看可用模型

### 8.5 认证失败（401 Unauthorized）

```
AuthenticationError: Invalid API key
```

**原因**：API Key 无效或未设置。

**解决**：
- 本地部署：`api_key` 设为任意非空字符串（如 `"EMPTY"`）
- 腾讯云：在[控制台](https://console.cloud.tencent.com/hunyuan/start)重新创建 Key，确认环境变量已正确设置

### 8.6 推理模式不生效

**原因**：`reasoning_effort` 未通过 `extra_body.chat_template_kwargs` 传递。

**解决**：
```python
# 正确写法
extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}}

# 错误写法（不会生效）
# extra_body={"reasoning_effort": "high"}
```

### 8.7 错误码速查表

| HTTP 状态码 | 含义 | 处理方式 |
|-------------|------|----------|
| 400 | 请求参数错误 | 检查 messages 格式、参数类型 |
| 401 | 认证失败 | 检查 API Key |
| 404 | 模型不存在 | 检查 model 名称 |
| 422 | 请求体格式错误 | 检查 JSON 格式 |
| 429 | 频率限制 | 降低频率 + 指数退避重试 |
| 500 | 服务端错误 | 重试，若持续联系支持 |
| 503 | 服务不可用 | 稍后重试 |

---

## 9. 进阶示例导航

| 示例 | 文件 | 说明 |
|------|------|------|
| 基础对话 | [01_basic_chat.py](examples/01_basic_chat.py) | 单轮对话 + 多轮对话 |
| 流式输出 | [02_streaming.py](examples/02_streaming.py) | 流式请求 + 逐块解析 |
| 性能对比 | [03_nonstreaming_vs_streaming.py](examples/03_nonstreaming_vs_streaming.py) | 首字延迟 / 总耗时对比 |
| 工具调用 | [04_tool_calling.py](examples/04_tool_calling.py) | 单次调用 + 多轮工具循环 |
| 推理模式 | [05_reasoning_mode.py](examples/05_reasoning_mode.py) | 思考开关对比 |
| 错误处理 | [06_error_handling_retry.py](examples/06_error_handling_retry.py) | 超时/限流/网络错误重试 |

每个示例均包含完整请求、完整响应解析和示例输出，可直接运行（仅需配置环境变量）。

---

## 附录：快速参考

```python
# ── 最小可运行模板 ──
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "EMPTY"),
    base_url=os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1"),
)

response = client.chat.completions.create(
    model=os.environ.get("HY3_MODEL", "hy3"),
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.9,
    top_p=1.0,
    # 推理模式: "no_think" | "low" | "high"
    extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
)

print(response.choices[0].message.content)
```

---

*Hy3 by Tencent Hunyuan Team · Apache License 2.0*
