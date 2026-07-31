"""
Hy3 API Example 01: Basic Chat (单轮 / 多轮对话)
=================================================

本示例演示：
1. 单轮对话 —— 一次请求，一次回答
2. 多轮对话 —— 携带上下文的连续对话

运行前提：
  - 本地已启动 vLLM/SGLang 服务（默认 http://127.0.0.1:8000）
  - 或设置环境变量使用腾讯云 API

环境变量：
  HY3_API_KEY   API Key（本地部署填 EMPTY，腾讯云填 sk-xxx）
  HY3_BASE_URL  API 地址（默认 http://127.0.0.1:8000/v1）
  HY3_MODEL     模型名称（默认 hy3，腾讯云用 hy3-295b）

运行：
  python 01_basic_chat.py
"""

import os
import json
from openai import OpenAI

# ─────────────────────────────────────────────────────
# 1. 初始化客户端
# ─────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "EMPTY"),
    base_url=os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1"),
    timeout=120.0,
)

MODEL = os.environ.get("HY3_MODEL", "hy3")


# ─────────────────────────────────────────────────────
# 2. 单轮对话
# ─────────────────────────────────────────────────────

def single_turn_chat():
    """单轮对话：发送一条消息，获取回答。"""
    print("=" * 60)
    print("【单轮对话】")
    print("=" * 60)

    # ── 完整请求 ──
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "用三句话解释什么是 MoE（混合专家模型）。"}
        ],
        temperature=0.9,
        top_p=1.0,
        # 推理模式：no_think 直接回答（默认）
        extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
    )

    # ── 完整响应解析 ──
    print(f"\n[请求] model={MODEL}")
    print(f"[请求] messages=[{{'role': 'user', 'content': '用三句话解释什么是 MoE（混合专家模型）。'}}]")
    print(f"\n[响应] id: {response.id}")
    print(f"[响应] model: {response.model}")
    print(f"[响应] finish_reason: {response.choices[0].finish_reason}")
    print(f"\n[响应] usage:")
    print(f"  prompt_tokens:     {response.usage.prompt_tokens}")
    print(f"  completion_tokens: {response.usage.completion_tokens}")
    print(f"  total_tokens:      {response.usage.total_tokens}")
    print(f"\n[响应] 回答内容:")
    print(response.choices[0].message.content)

    return response


# ─────────────────────────────────────────────────────
# 3. 多轮对话
# ─────────────────────────────────────────────────────

def multi_turn_chat():
    """
    多轮对话：将之前的对话历史拼接在 messages 中传递。
    Hy3 兼容 OpenAI 格式，需要手动维护 messages 列表。
    """
    print("\n" + "=" * 60)
    print("【多轮对话】")
    print("=" * 60)

    # 初始化对话历史
    messages = [
        {"role": "system", "content": "你是一个简洁的助手，回答控制在两句话以内。"},
        {"role": "user", "content": "世界上最高的山是什么？"},
    ]

    # ── 第一轮 ──
    print(f"\n--- 第 1 轮 ---")
    print(f"User: {messages[-1]['content']}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.9,
        top_p=1.0,
    )

    assistant_reply = response.choices[0].message.content
    print(f"Assistant: {assistant_reply}")

    # 将助手回复加入对话历史
    messages.append({"role": "assistant", "content": assistant_reply})

    # ── 第二轮（携带上下文）──
    messages.append({"role": "user", "content": "第二高的呢？"})
    print(f"\n--- 第 2 轮 ---")
    print(f"User: {messages[-1]['content']}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.9,
        top_p=1.0,
    )

    assistant_reply = response.choices[0].message.content
    print(f"Assistant: {assistant_reply}")

    # 将助手回复加入对话历史
    messages.append({"role": "assistant", "content": assistant_reply})

    # ── 第三轮（携带完整上下文）──
    messages.append({"role": "user", "content": "它们的高度差大约是多少？"})
    print(f"\n--- 第 3 轮 ---")
    print(f"User: {messages[-1]['content']}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.9,
        top_p=1.0,
    )

    assistant_reply = response.choices[0].message.content
    print(f"Assistant: {assistant_reply}")

    # ── 完整响应解析（以最后一轮为例）──
    print(f"\n[最后一轮完整响应]")
    print(f"  finish_reason: {response.choices[0].finish_reason}")
    print(f"  prompt_tokens: {response.usage.prompt_tokens}")
    print(f"  completion_tokens: {response.usage.completion_tokens}")
    print(f"  total_tokens: {response.usage.total_tokens}")

    # 打印完整对话历史
    print(f"\n[完整对话历史]")
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        print(f"  {role}: {content}")

    return messages


# ─────────────────────────────────────────────────────
# 4. 带参数控制的单轮对话
# ─────────────────────────────────────────────────────

def chat_with_parameters():
    """演示 temperature 和 max_tokens 参数的效果。"""
    print("\n" + "=" * 60)
    print("【参数控制对比】")
    print("=" * 60)

    prompt = "用 Python 写一个冒泡排序"

    # 低温度 + 限制 token
    print(f"\n--- temperature=0.1, max_tokens=256 ---")
    r1 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=256,
    )
    print(r1.choices[0].message.content)
    print(f"\n(tokens: {r1.usage.completion_tokens}, finish: {r1.choices[0].finish_reason})")

    # 高温度
    print(f"\n--- temperature=1.2, max_tokens=256 ---")
    r2 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.2,
        max_tokens=256,
    )
    print(r2.choices[0].message.content)
    print(f"\n(tokens: {r2.usage.completion_tokens}, finish: {r2.choices[0].finish_reason})")


# ─────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Hy3 API Example 01: Basic Chat")
    print(f"Endpoint: {os.environ.get('HY3_BASE_URL', 'http://127.0.0.1:8000/v1')}")
    print(f"Model:    {MODEL}")
    print()

    single_turn_chat()
    multi_turn_chat()
    chat_with_parameters()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
# 示例输出（实际输出因模型版本和参数不同会有差异）
# ═══════════════════════════════════════════════════════
#
# ============================================================
# 【单轮对话】
# ============================================================
#
# [请求] model=hy3
# [请求] messages=[{'role': 'user', 'content': '用三句话解释什么是 MoE（混合专家模型）。'}]
#
# [响应] id: chatcmpl-abc123
# [响应] model: hy3
# [响应] finish_reason: stop
#
# [响应] usage:
#   prompt_tokens:     18
#   completion_tokens: 52
#   total_tokens:      70
#
# [响应] 回答内容:
# MoE（Mixture of Experts，混合专家模型）是一种通过动态激活部分
# 神经网络参数来提升效率的架构。它包含多个"专家"子网络，每次推理
# 只激活其中少数几个，从而在不增加计算量的情况下扩大模型容量。
# 这种设计让大规模模型在保持推理速度的同时获得更强的表达能力。
#
# ============================================================
# 【多轮对话】
# ============================================================
#
# --- 第 1 轮 ---
# User: 世界上最高的山是什么？
# Assistant: 珠穆朗玛峰，海拔 8848.86 米，位于中国与尼泊尔边境。
#
# --- 第 2 轮 ---
# User: 第二高的呢？
# Assistant: 乔戈里峰（K2），海拔 8611 米，位于中国与巴基斯坦边境。
#
# --- 第 3 轮 ---
# User: 它们的高度差大约是多少？
# Assistant: 珠穆朗玛峰（8848.86米）与乔戈里峰（8611米）的高度差约为
# 238 米。
#
# [最后一轮完整响应]
#   finish_reason: stop
#   prompt_tokens: 78
#   completion_tokens: 35
#   total_tokens: 113
#
# [完整对话历史]
#   system: 你是一个简洁的助手，回答控制在两句话以内。
#   user: 世界上最高的山是什么？
#   assistant: 珠穆朗玛峰，海拔 8848.86 米，位于中国与尼泊尔边境。
#   user: 第二高的呢？
#   assistant: 乔戈里峰（K2），海拔 8611 米，位于中国与巴基斯坦边境。
#   user: 它们的高度差大约是多少？
#   assistant: 珠穆朗玛峰（8848.86米）与乔戈里峰（8611米）的高度差约为 238 米。
#
# ============================================================
# 示例完成！
# ============================================================
