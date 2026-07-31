"""
Hy3 API Example 05: Reasoning Mode (推理模式对比)
===================================================

本示例演示：
1. no_think 模式：直接回答，不展示思考过程
2. low 模式：轻量思考
3. high 模式：深度推理，展示完整思维链

对比维度：
  - 响应内容质量
  - 思考过程（reasoning_content）
  - 耗时和 Token 消耗

运行：
  python 05_reasoning_mode.py
"""

import os
import time
from openai import OpenAI

# ─────────────────────────────────────────────────────
# 初始化客户端
# ─────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "EMPTY"),
    base_url=os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1"),
    timeout=180.0,
)

MODEL = os.environ.get("HY3_MODEL", "hy3")

# 测试问题集（从简单到复杂）
TEST_QUESTIONS = [
    {
        "category": "简单问答",
        "question": "中国的首都是哪里？",
        "expect_reasoning": False,
    },
    {
        "category": "数学推理",
        "question": "一个农场有鸡和兔子共 35 只，脚共 94 只。鸡和兔子各有多少？",
        "expect_reasoning": True,
    },
    {
        "category": "逻辑推理",
        "question": "Alice 比 Bob 高，Bob 比 Charlie 高，David 比 Alice 高。谁最高？谁最矮？",
        "expect_reasoning": True,
    },
    {
        "category": "代码推理",
        "question": "以下 Python 代码的输出是什么？\n\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)",
        "expect_reasoning": True,
    },
]


# ─────────────────────────────────────────────────────
# 1. 单模式调用
# ─────────────────────────────────────────────────────

def call_with_reasoning(question, reasoning_effort="no_think"):
    """
    使用指定的推理模式调用 Hy3。

    Args:
        question: 用户问题
        reasoning_effort: "no_think" | "low" | "high"

    Returns:
        dict: 包含思考过程、回答、耗时、token 使用量
    """
    start = time.time()

    # ── 完整请求 ──
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0.9,
        top_p=1.0,
        # 推理模式通过 extra_body.chat_template_kwargs 传递
        extra_body={"chat_template_kwargs": {"reasoning_effort": reasoning_effort}},
    )

    elapsed = time.time() - start
    msg = response.choices[0].message

    # ── 完整响应解析 ──
    result = {
        "reasoning_effort": reasoning_effort,
        "reasoning_content": None,
        "content": msg.content,
        "elapsed": elapsed,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "finish_reason": response.choices[0].finish_reason,
    }

    # 提取思考过程（reasoning_content）
    # 注意：no_think 模式下通常没有 reasoning_content
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        result["reasoning_content"] = msg.reasoning_content

    return result


# ─────────────────────────────────────────────────────
# 2. 三种模式对比
# ─────────────────────────────────────────────────────

def compare_modes(question, category):
    """对同一个问题用三种推理模式分别调用并对比。"""
    print(f"\n{'─' * 70}")
    print(f"测试类别: {category}")
    print(f"问题: {question}")
    print(f"{'─' * 70}")

    modes = ["no_think", "low", "high"]
    results = {}

    for mode in modes:
        print(f"\n  ── 模式: {mode} ──")
        result = call_with_reasoning(question, mode)
        results[mode] = result

        # 显示思考过程（如果有）
        if result["reasoning_content"]:
            # 截取前 300 字符展示
            reasoning = result["reasoning_content"]
            if len(reasoning) > 300:
                reasoning = reasoning[:300] + " ...(截断)"
            print(f"  [思考过程] ({len(result['reasoning_content'])} 字符)")
            print(f"  {reasoning}")
            print()

        # 显示最终回答
        print(f"  [最终回答]")
        print(f"  {result['content']}")

        # 显示统计
        print(f"\n  [统计] 耗时: {result['elapsed']:.3f}s | "
              f"completion_tokens: {result['completion_tokens']} | "
              f"total_tokens: {result['total_tokens']}")

        time.sleep(0.5)  # 避免请求过快

    # ── 对比汇总 ──
    print(f"\n  {'模式':<12} {'耗时(s)':<12} {'思考Token':<12} {'回答Token':<12} {'总Token':<12}")
    print(f"  {'─' * 60}")

    for mode in modes:
        r = results[mode]
        # 估算思考 token（如果有 reasoning_content）
        reasoning_tokens = 0
        if r["reasoning_content"]:
            # 粗略估算：中文字符数 / 2
            reasoning_tokens = len(r["reasoning_content"]) // 2
        answer_tokens = r["completion_tokens"] - reasoning_tokens

        print(f"  {mode:<12} {r['elapsed']:<12.3f} {reasoning_tokens:<12} {answer_tokens:<12} {r['total_tokens']:<12}")

    return results


# ─────────────────────────────────────────────────────
# 3. 流式模式下的推理对比
# ─────────────────────────────────────────────────────

def streaming_reasoning_comparison():
    """流式模式下对比 no_think 和 high 的输出体验。"""
    print("\n" + "=" * 70)
    print("【流式模式推理对比】")
    print("=" * 70)

    question = "证明：对于任意正整数 n，n³ - n 能被 6 整除。"

    for mode in ["no_think", "high"]:
        print(f"\n{'─' * 50}")
        print(f"模式: {mode}")
        print(f"{'─' * 50}")

        # ── 流式请求 ──
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            temperature=0.9,
            top_p=1.0,
            stream=True,
            extra_body={"chat_template_kwargs": {"reasoning_effort": mode}},
        )

        start = time.time()
        first_token = None
        in_reasoning = False
        reasoning_text = ""
        answer_text = ""

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 思考过程
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if not in_reasoning:
                    print("\n[思考过程]")
                    in_reasoning = True
                if first_token is None:
                    first_token = time.time()
                reasoning_text += delta.reasoning_content
                print(delta.reasoning_content, end="", flush=True)

            # 最终回答
            if delta.content:
                if in_reasoning:
                    print("\n\n[最终回答]")
                    in_reasoning = False
                if first_token is None:
                    first_token = time.time()
                answer_text += delta.content
                print(delta.content, end="", flush=True)

        total = time.time() - start
        ttft = (first_token - start) if first_token else 0

        print(f"\n\n[统计] 首字延迟: {ttft:.3f}s | 总耗时: {total:.3f}s")
        print(f"[统计] 思考过程: {len(reasoning_text)} 字符 | 回答: {len(answer_text)} 字符")

        time.sleep(0.5)


# ─────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Hy3 API Example 05: Reasoning Mode")
    print(f"Endpoint: {os.environ.get('HY3_BASE_URL', 'http://127.0.0.1:8000/v1')}")
    print(f"Model:    {MODEL}")
    print()
    print("推理模式说明:")
    print("  no_think: 直接回答，不生成思考过程（默认，最快）")
    print("  low:      轻量思考，平衡速度与质量")
    print("  high:     深度推理，展示完整思维链（最慢但质量最高）")

    # 对每个测试问题对比三种模式
    all_results = {}
    for test in TEST_QUESTIONS:
        results = compare_modes(test["question"], test["category"])
        all_results[test["category"]] = results

    # 流式对比
    streaming_reasoning_comparison()

    # 最终汇总
    print("\n" + "=" * 70)
    print("【全部测试汇总】")
    print("=" * 70)
    print(f"\n{'类别':<12} {'模式':<12} {'耗时(s)':<12} {'总Token':<12}")
    print("─" * 50)
    for category, results in all_results.items():
        for mode, r in results.items():
            print(f"{category:<12} {mode:<12} {r['elapsed']:<12.3f} {r['total_tokens']:<12}")

    print("\n结论:")
    print("  - no_think 模式适合简单任务（翻译、问答），响应最快")
    print("  - high 模式适合复杂推理（数学证明、逻辑分析），质量最高但耗时更长")
    print("  - low 模式在速度和质量间取得平衡，适合中等复杂度任务")
    print("  - 流式模式下，high 模式可以先输出思考过程，让用户提前看到进展")

    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


# ═══════════════════════════════════════════════════════
# 示例输出（实际输出因模型版本和参数不同会有差异）
# ═══════════════════════════════════════════════════════
#
# ──────────────────────────────────────────────────────────────
# 测试类别: 数学推理
# 问题: 一个农场有鸡和兔子共 35 只，脚共 94 只。鸡和兔子各有多少？
# ──────────────────────────────────────────────────────────────
#
#   ── 模式: no_think ──
#   [最终回答]
#   设鸡有 x 只，兔子有 y 只。
#   x + y = 35
#   2x + 4y = 94
#   解得：x = 23, y = 12
#   鸡有 23 只，兔子有 12 只。
#
#   [统计] 耗时: 1.234s | completion_tokens: 45 | total_tokens: 78
#
#   ── 模式: high ──
#   [思考过程] (187 字符)
#   设鸡的数量为 x，兔的数量为 y。
#   根据题意列方程组：
#   ① x + y = 35（总头数）
#   ② 2x + 4y = 94（总脚数）
#   由①得 x = 35 - y，代入②：
#   2(35 - y) + 4y = 94
#   70 - 2y + 4y = 94
#   2y = 24
#   y = 12
#   x = 35 - 12 = 23
#   验证：23×2 + 12×4 = 46 + 48 = 94 ✓
#
#   [最终回答]
#   鸡有 23 只，兔子有 12 只。
#
#   验证：23 + 12 = 35（总只数正确），23×2 + 12×4 = 46 + 48 = 94（总脚数正确）。
#
#   [统计] 耗时: 3.567s | completion_tokens: 120 | total_tokens: 153
#
#   模式          耗时(s)      思考Token    回答Token    总Token
#   ────────────────────────────────────────────────────────────
#   no_think      1.234        0            45           78
#   low           2.156        35           48           96
#   high          3.567        94           26           153
#
# ======================================================================
# 【全部测试汇总】
# ======================================================================
#
# 类别         模式          耗时(s)      总Token
# ──────────────────────────────────────────────────────────
# 简单问答     no_think      0.856        52
# 简单问答     low           0.923        55
# 简单问答     high          1.145        68
# 数学推理     no_think      1.234        78
# 数学推理     low           2.156        96
# 数学推理     high          3.567        153
# 逻辑推理     no_think      1.045        65
# 逻辑推理     low           1.678        82
# 逻辑推理     high          2.834        128
# 代码推理     no_think      1.123        72
# 代码推理     low           1.789        89
# 代码推理     high          3.012        145
#
# 结论:
#   - no_think 模式适合简单任务（翻译、问答），响应最快
#   - high 模式适合复杂推理（数学证明、逻辑分析），质量最高但耗时更长
#   - low 模式在速度和质量间取得平衡，适合中等复杂度任务
#   - 流式模式下，high 模式可以先输出思考过程，让用户提前看到进展
#
# ======================================================================
# 示例完成！
# ======================================================================
