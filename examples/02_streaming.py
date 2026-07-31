"""
Hy3 API Example 02: Streaming (流式输出 + 逐块解析)
====================================================

本示例演示：
1. 流式请求的基本用法
2. 逐块（chunk）解析 SSE 数据
3. 流式模式下的 reasoning_content（思考过程）解析
4. 手动解析原始 SSE 数据流（不依赖 SDK）

运行前提：
  - 本地已启动 vLLM/SGLang 服务
  - 或设置环境变量使用腾讯云 API

运行：
  python 02_streaming.py
"""

import os
import json
import time
import httpx
from openai import OpenAI

# ─────────────────────────────────────────────────────
# 1. 初始化客户端
# ─────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "EMPTY"),
    base_url=os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1"),
    timeout=180.0,
)

MODEL = os.environ.get("HY3_MODEL", "hy3")


# ─────────────────────────────────────────────────────
# 2. 基础流式输出
# ─────────────────────────────────────────────────────

def basic_streaming():
    """最简单的流式调用：逐块打印文本。"""
    print("=" * 60)
    print("【基础流式输出】")
    print("=" * 60)

    # ── 完整请求 ──
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "用 100 字介绍量子纠缠现象。"}
        ],
        temperature=0.9,
        top_p=1.0,
        stream=True,  # 开启流式
    )

    # ── 逐块解析 ──
    full_content = ""
    chunk_count = 0
    first_token_time = None
    start_time = time.time()

    print("\n[流式输出开始]\n")

    for chunk in stream:
        chunk_count += 1

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # 记录首字时间
        if first_token_time is None and delta.content:
            first_token_time = time.time()

        # 提取文本内容
        if delta.content:
            full_content += delta.content
            print(delta.content, end="", flush=True)

    elapsed = time.time() - start_time
    ttft = (first_token_time - start_time) if first_token_time else 0

    print(f"\n\n[流式输出结束]")
    print(f"  总块数:         {chunk_count}")
    print(f"  首字延迟(TTFT): {ttft:.3f}s")
    print(f"  总耗时:         {elapsed:.3f}s")
    print(f"  完整文本长度:   {len(full_content)} 字符")

    return full_content


# ─────────────────────────────────────────────────────
# 3. 流式 + 推理模式（解析 reasoning_content）
# ─────────────────────────────────────────────────────

def streaming_with_reasoning():
    """
    流式模式下开启推理模式（reasoning_effort=high）。
    思考过程通过 delta.reasoning_content 逐块返回。
    """
    print("\n" + "=" * 60)
    print("【流式 + 推理模式】")
    print("=" * 60)

    # ── 完整请求 ──
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "一个水池有 A、B 两个进水管。A 单独开 6 小时注满，B 单独开 8 小时注满。两管同时开几小时注满？"}
        ],
        temperature=0.9,
        top_p=1.0,
        stream=True,
        extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
    )

    # ── 逐块解析（区分思考过程和最终回答）──
    reasoning_content = ""
    answer_content = ""
    in_reasoning = False

    print("\n[流式输出开始]\n")

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # 检查是否有 reasoning_content（思考过程）
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            if not in_reasoning:
                print("--- [思考过程] ---")
                in_reasoning = True
            reasoning_content += delta.reasoning_content
            print(delta.reasoning_content, end="", flush=True)

        # 检查正式回答内容
        if delta.content:
            if in_reasoning:
                print("\n--- [最终回答] ---")
                in_reasoning = False
            answer_content += delta.content
            print(delta.content, end="", flush=True)

    print(f"\n\n[解析结果]")
    print(f"  思考过程长度: {len(reasoning_content)} 字符")
    print(f"  回答长度:     {len(answer_content)} 字符")


# ─────────────────────────────────────────────────────
# 4. 手动解析 SSE 原始数据流
# ─────────────────────────────────────────────────────

def manual_sse_parsing():
    """
    不使用 OpenAI SDK，直接用 httpx 解析 SSE 数据流。
    适用于需要完全控制数据流的场景。
    """
    print("\n" + "=" * 60)
    print("【手动 SSE 解析】")
    print("=" * 60)

    base_url = os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("HY3_API_KEY", "EMPTY")

    # ── 完整请求（原始 HTTP）──
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "说三个关于编程的冷笑话，每个用序号标注。"}
        ],
        "temperature": 0.9,
        "top_p": 1.0,
        "stream": True,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    print("\n[原始 SSE 数据流]\n")

    full_text = ""
    chunk_idx = 0

    # ── 逐行读取 SSE 流 ──
    with httpx.Client(timeout=180.0) as http_client:
        with http_client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue

                # SSE 格式：data: {...}
                if line.startswith("data: "):
                    data_str = line[6:]  # 去掉 "data: " 前缀

                    # [DONE] 标记流结束
                    if data_str.strip() == "[DONE]":
                        print("\n[SSE 流结束: [DONE]]")
                        break

                    # ── 解析 JSON 数据块 ──
                    try:
                        chunk = json.loads(data_str)
                        chunk_idx += 1

                        # 提取内容
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                full_text += content
                                # 显示前 5 个块的完整 JSON 结构
                                if chunk_idx <= 5:
                                    print(f"\n[Chunk #{chunk_idx}] 原始 JSON:")
                                    print(json.dumps(chunk, ensure_ascii=False, indent=2))
                                    print(f"提取内容: '{content}'")
                                else:
                                    print(content, end="", flush=True)

                    except json.JSONDecodeError:
                        pass

    print(f"\n\n[解析统计]")
    print(f"  总数据块数:   {chunk_idx}")
    print(f"  完整文本长度: {len(full_text)} 字符")
    print(f"\n[完整回答]:")
    print(full_text)

    return full_text


# ─────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Hy3 API Example 02: Streaming")
    print(f"Endpoint: {os.environ.get('HY3_BASE_URL', 'http://127.0.0.1:8000/v1')}")
    print(f"Model:    {MODEL}")
    print()

    basic_streaming()
    streaming_with_reasoning()
    manual_sse_parsing()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
# 示例输出（实际输出因模型版本和参数不同会有差异）
# ═══════════════════════════════════════════════════════
#
# ============================================================
# 【基础流式输出】
# ============================================================
#
# [流式输出开始]
#
# 量子纠缠是量子力学中的一种现象：两个或多个粒子相互作用后，
# 形成一个不可分割的整体系统。即使将它们分离到很远的地方，
# 对其中一个粒子的测量会瞬间影响另一个粒子的状态，这种关联
# 不受距离限制。爱因斯坦称之为"幽灵般的超距作用"，它不违反
# 相对论，因为无法用于超光速通信。
#
# [流式输出结束]
#   总块数:         47
#   首字延迟(TTFT): 0.342s
#   总耗时:         3.871s
#   完整文本长度:   156 字符
#
# ============================================================
# 【流式 + 推理模式】
# ============================================================
#
# [流式输出开始]
#
# --- [思考过程] ---
# A管每小时注水 1/6，B管每小时注水 1/8。两管同时开每小时注水
# 1/6 + 1/8 = 4/24 + 3/24 = 7/24。注满需要 24/7 ≈ 3.43 小时。
# --- [最终回答] ---
# 两管同时打开需要 24/7 小时，约 3 小时 26 分钟注满水池。
#
# [解析结果]
#   思考过程长度: 89 字符
#   回答长度:     32 字符
#
# ============================================================
# 【手动 SSE 解析】
# ============================================================
#
# [原始 SSE 数据流]
#
# [Chunk #1] 原始 JSON:
# {
#   "id": "chatcmpl-abc123",
#   "object": "chat.completion.chunk",
#   "created": 1722400000,
#   "model": "hy3",
#   "choices": [
#     {
#       "index": 0,
#       "delta": {
#         "role": "assistant",
#         "content": ""
#       },
#       "finish_reason": null
#     }
#   ]
# }
# 提取内容: ''
#
# [Chunk #2] 原始 JSON:
# {
#   "id": "chatcmpl-abc123",
#   "object": "chat.completion.chunk",
#   "created": 1722400000,
#   "model": "hy3",
#   "choices": [
#     {
#       "index": 0,
#       "delta": {
#         "content": "1"
#       },
#       "finish_reason": null
#     }
#   ]
# }
# 提取内容: '1'
#
# . 为什么程序员偏好黑暗模式？因为光会吸引 bug。
# 2. 世界上有 10 种人：懂二进制的和不懂的。
# 3. 程序员最讨厌什么？注释说"别改这段代码"但没有说为什么。
#
# [SSE 流结束: [DONE]]
#
# [解析统计]
#   总数据块数:   52
#   完整文本长度: 98 字符
#
# [完整回答]:
# 1. 为什么程序员偏好黑暗模式？因为光会吸引 bug。
# 2. 世界上有 10 种人：懂二进制的和不懂的。
# 3. 程序员最讨厌什么？注释说"别改这段代码"但没有说为什么。
#
# ============================================================
# 示例完成！
# ============================================================
