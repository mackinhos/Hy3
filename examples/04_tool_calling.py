"""
Hy3 API Example 04: Tool Calling (工具调用)
=============================================

本示例演示：
1. 单次工具调用：模型选择并调用一个工具
2. 多轮工具循环：模型连续调用多个工具完成复杂任务

工具调用流程：
  定义工具 → 发送请求 → 模型返回 tool_calls → 执行工具 →
  结果回传 → 模型生成最终回答（或继续调用工具）

运行：
  python 04_tool_calling.py
"""

import os
import json
from openai import OpenAI

# ─────────────────────────────────────────────────────
# 初始化客户端
# ─────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "EMPTY"),
    base_url=os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1"),
    timeout=120.0,
)

MODEL = os.environ.get("HY3_MODEL", "hy3")


# ─────────────────────────────────────────────────────
# 工具函数定义（实际业务中替换为你的函数）
# ─────────────────────────────────────────────────────

def get_weather(city: str) -> str:
    """模拟天气查询。实际项目中调用天气 API。"""
    mock_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，28°C，湿度 65%",
        "广州": "雷阵雨，30°C，湿度 80%",
        "深圳": "晴转多云，29°C，湿度 70%",
    }
    return mock_data.get(city, f"暂无 {city} 的天气数据")


def calculate(expression: str) -> str:
    """安全计算数学表达式。"""
    try:
        allowed = set("0123456789+-*/(). ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "错误：表达式包含非法字符"
    except Exception as e:
        return f"计算错误: {e}"


def search_knowledge(query: str) -> str:
    """模拟知识库搜索。"""
    mock_db = {
        "Python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        "MoE": "MoE（Mixture of Experts）是一种混合专家模型架构，通过动态激活部分专家来提升效率。",
    }
    for key, val in mock_db.items():
        if key.lower() in query.lower():
            return val
    return f"未找到与 '{query}' 相关的知识。"


# 工具注册表：函数名 → Python 函数
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
}

# OpenAI 格式的工具定义
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如：15 + 27 * 3",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库中搜索技术概念的解释",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如：Python、MoE",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool(tool_call) -> str:
    """执行模型请求的工具调用，返回结果字符串。"""
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)

    print(f"    [执行工具] {func_name}({func_args})")

    if func_name in TOOL_REGISTRY:
        result = TOOL_REGISTRY[func_name](**func_args)
        print(f"    [工具结果] {result}")
        return str(result)
    return f"错误：未知工具 {func_name}"


# ─────────────────────────────────────────────────────
# 1. 单次工具调用
# ─────────────────────────────────────────────────────

def single_tool_call():
    """模型选择一个工具并调用。"""
    print("=" * 60)
    print("【单次工具调用】")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "北京今天天气怎么样？"}
    ]

    print(f"\n[用户] {messages[0]['content']}")

    # ── 第一次请求：模型决定是否调用工具 ──
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        temperature=0.9,
        top_p=1.0,
    )

    msg = response.choices[0].message
    print(f"\n[模型响应] finish_reason: {response.choices[0].finish_reason}")

    # ── 检查是否请求工具调用 ──
    if msg.tool_calls:
        print(f"[模型请求工具调用] 共 {len(msg.tool_calls)} 个")

        # 将助手消息（含 tool_calls）加入历史
        messages.append(msg)

        for tc in msg.tool_calls:
            print(f"\n  tool_call_id: {tc.id}")
            print(f"  function: {tc.function.name}")
            print(f"  arguments: {tc.function.arguments}")

            # 执行工具
            result = execute_tool(tc)

            # 将工具结果加入历史
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # ── 第二次请求：模型基于工具结果生成最终回答 ──
        print(f"\n[第二次请求] 携带工具结果，请求最终回答...")
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.9,
            top_p=1.0,
        )

        final_answer = final_response.choices[0].message.content
        print(f"\n[最终回答]")
        print(final_answer)

        # ── 完整响应解析 ──
        print(f"\n[响应解析]")
        print(f"  finish_reason: {final_response.choices[0].finish_reason}")
        print(f"  prompt_tokens: {final_response.usage.prompt_tokens}")
        print(f"  completion_tokens: {final_response.usage.completion_tokens}")
        print(f"  total_tokens: {final_response.usage.total_tokens}")
    else:
        print("[模型未请求工具调用，直接回答]")
        print(msg.content)


# ─────────────────────────────────────────────────────
# 2. 多轮工具循环
# ─────────────────────────────────────────────────────

def multi_turn_tool_loop():
    """
    多轮工具循环：模型可能连续调用多个工具。
    使用 while 循环处理，直到模型不再请求工具调用。
    """
    print("\n" + "=" * 60)
    print("【多轮工具循环】")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个助手，可以使用提供的工具来回答问题。每次只调用必要的工具。"},
        {"role": "user", "content": "帮我做三件事：1.查一下北京和上海的天气 2.算一下 125 * 48 3.搜索一下什么是 MoE"},
    ]

    print(f"\n[用户] {messages[-1]['content']}")

    max_rounds = 10  # 防止无限循环
    round_num = 0

    while round_num < max_rounds:
        round_num += 1
        print(f"\n--- 第 {round_num} 轮请求 ---")

        # ── 请求模型 ──
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.9,
            top_p=1.0,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print(f"[finish_reason] {finish_reason}")

        # ── 如果模型请求调用工具 ──
        if msg.tool_calls and len(msg.tool_calls) > 0:
            print(f"[模型请求 {len(msg.tool_calls)} 个工具调用]")
            messages.append(msg)

            for tc in msg.tool_calls:
                result = execute_tool(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # 继续循环，让模型处理工具结果
            continue

        # ── 模型不再调用工具，输出最终回答 ──
        print(f"\n[最终回答]")
        print(msg.content)

        # ── 完整对话历史 ──
        print(f"\n[完整对话历史]")
        for i, m in enumerate(messages):
            role = m["role"]
            if role == "assistant" and hasattr(m, "tool_calls") and m.tool_calls:
                tools_called = [tc.function.name for tc in m.tool_calls]
                print(f"  [{i}] assistant (tool_calls={tools_called})")
            elif role == "tool":
                content = m["content"][:50] + "..." if len(m["content"]) > 50 else m["content"]
                print(f"  [{i}] tool: {content}")
            elif hasattr(m, "content"):
                content = m.content[:50] + "..." if m.content and len(m.content) > 50 else (m.content or "")
                print(f"  [{i}] {role}: {content}")
            else:
                content = m.get("content", "")[:50]
                print(f"  [{i}] {role}: {content}")

        print(f"\n[统计]")
        print(f"  总轮数: {round_num}")
        print(f"  prompt_tokens: {response.usage.prompt_tokens}")
        print(f"  completion_tokens: {response.usage.completion_tokens}")
        print(f"  total_tokens: {response.usage.total_tokens}")

        break

    else:
        print(f"\n[警告] 达到最大轮数 {max_rounds}，可能存在无限循环")


# ─────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Hy3 API Example 04: Tool Calling")
    print(f"Endpoint: {os.environ.get('HY3_BASE_URL', 'http://127.0.0.1:8000/v1')}")
    print(f"Model:    {MODEL}")
    print()

    single_tool_call()
    multi_turn_tool_loop()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
# 示例输出（实际输出因模型版本和参数不同会有差异）
# ═══════════════════════════════════════════════════════
#
# ============================================================
# 【单次工具调用】
# ============================================================
#
# [用户] 北京今天天气怎么样？
#
# [模型响应] finish_reason: tool_calls
# [模型请求工具调用] 共 1 个
#
#   tool_call_id: call_abc123
#   function: get_weather
#   arguments: {"city": "北京"}
#     [执行工具] get_weather({'city': '北京'})
#     [工具结果] 晴天，25°C，湿度 40%
#
# [第二次请求] 携带工具结果，请求最终回答...
#
# [最终回答]
# 北京今天的天气是晴天，气温 25°C，湿度 40%。适合外出活动！
#
# [响应解析]
#   finish_reason: stop
#   prompt_tokens: 156
#   completion_tokens: 28
#   total_tokens: 184
#
# ============================================================
# 【多轮工具循环】
# ============================================================
#
# [用户] 帮我做三件事：1.查一下北京和上海的天气 2.算一下 125 * 48 3.搜索一下什么是 MoE
#
# --- 第 1 轮请求 ---
# [finish_reason] tool_calls
# [模型请求 3 个工具调用]
#     [执行工具] get_weather({'city': '北京'})
#     [工具结果] 晴天，25°C，湿度 40%
#     [执行工具] get_weather({'city': '上海'})
#     [工具结果] 多云，28°C，湿度 65%
#     [执行工具] calculate({'expression': '125 * 48'})
#     [工具结果] 6000
#
# --- 第 2 轮请求 ---
# [finish_reason] tool_calls
# [模型请求 1 个工具调用]
#     [执行工具] search_knowledge({'query': 'MoE'})
#     [工具结果] MoE（Mixture of Experts）是一种混合专家模型架构，通过动态激活部分专家来提升效率。
#
# --- 第 3 轮请求 ---
# [finish_reason] stop
#
# [最终回答]
# 以下是您要查询的三项结果：
#
# 1. **天气查询**
#    - 北京：晴天，25°C，湿度 40%
#    - 上海：多云，28°C，湿度 65%
#
# 2. **数学计算**
#    - 125 × 48 = 6000
#
# 3. **知识搜索**
#    - MoE（Mixture of Experts）是一种混合专家模型架构，通过动态激活
#      部分专家来提升效率。
#
# [统计]
#   总轮数: 3
#   prompt_tokens: 512
#   completion_tokens: 95
#   total_tokens: 607
#
# ============================================================
# 示例完成！
# ============================================================
