"""
Hy3 API Example 03: Non-Streaming vs Streaming 性能对比
=========================================================

本示例演示：
1. 非流式请求的耗时测量
2. 流式请求的首字延迟（TTFT）和总耗时测量
3. 两种模式的性能对比报告

核心指标：
  - TTFT (Time To First Token): 首字延迟
  - Total Time: 总耗时
  - Tokens/s: 生成速度

运行：
  python 03_nonstreaming_vs_streaming.py
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

# 测试用的 prompt（中等长度，适合对比）
TEST_PROMPTS = [
    {
        "name": "短回答（约 100 字）",
        "prompt": "用 100 字介绍 Python 的 GIL。",
        "max_tokens": 256,
    },
    {
        "name": "中等回答（约 300 字）",
        "prompt": "详细解释 HTTPS 的工作原理，包括 TLS 握手过程。",
        "max_tokens": 512,
    },
    {
        "name": "长回答（约 500 字）",
        "prompt": "写一篇关于人工智能发展史的短文，从图灵测试到 GPT。",
        "max_tokens": 1024,
    },
]


# ─────────────────────────────────────────────────────
# 1. 非流式请求
# ─────────────────────────────────────────────────────

def benchmark_non_streaming(prompt, max_tokens):
    """
    非流式请求：等待完整响应返回。
    返回 (耗时, completion_tokens, 回答文本)。
    """
    start = time.time()

    # ── 完整请求 ──
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        top_p=1.0,
        max_tokens=max_tokens,
        stream=False,
    )

    elapsed = time.time() - start
    content = response.choices[0].message.content
    completion_tokens = response.usage.completion_tokens

    return elapsed, completion_tokens, content


# ─────────────────────────────────────────────────────
# 2. 流式请求
# ─────────────────────────────────────────────────────

def benchmark_streaming(prompt, max_tokens):
    """
    流式请求：逐块接收。
    返回 (首字延迟, 总耗时, completion_tokens估算, 回答文本)。
    """
    start = time.time()
    first_token_time = None
    full_content = ""
    chunk_count = 0

    # ── 完整请求 ──
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        top_p=1.0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},  # 在最后一个 chunk 包含 usage
    )

    completion_tokens = 0

    # ── 逐块解析 ──
    for chunk in stream:
        chunk_count += 1

        # 最后一个 chunk 可能包含 usage 信息
        if hasattr(chunk, "usage") and chunk.usage:
            completion_tokens = chunk.usage.completion_tokens

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            full_content += delta.content

    total_time = time.time() - start
    ttft = (first_token_time - start) if first_token_time else 0

    # 如果服务端未返回 usage，用粗略估算
    if completion_tokens == 0:
        completion_tokens = len(full_content) // 2  # 中文约 2 字符/token

    return ttft, total_time, completion_tokens, full_content


# ─────────────────────────────────────────────────────
# 3. 运行对比测试
# ─────────────────────────────────────────────────────

def run_comparison():
    """对所有测试 prompt 运行流式与非流式对比。"""
    print("=" * 70)
    print("Hy3 API 性能对比：非流式 vs 流式")
    print(f"Endpoint: {os.environ.get('HY3_BASE_URL', 'http://127.0.0.1:8000/v1')}")
    print(f"Model:    {MODEL}")
    print("=" * 70)

    results = []

    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 70}")
        print(f"测试 {i}/{len(TEST_PROMPTS)}: {test['name']}")
        print(f"Prompt: {test['prompt']}")
        print(f"max_tokens: {test['max_tokens']}")
        print(f"{'─' * 70}")

        # ── 非流式 ──
        print("\n  [非流式] 请求中...")
        ns_time, ns_tokens, ns_content = benchmark_non_streaming(
            test["prompt"], test["max_tokens"]
        )
        ns_speed = ns_tokens / ns_time if ns_time > 0 else 0
        print(f"  [非流式] 完成: {ns_time:.3f}s, {ns_tokens} tokens, {ns_speed:.1f} tokens/s")
        print(f"  [非流式] 回答前 80 字: {ns_content[:80]}...")

        # 间隔 1 秒避免连续请求
        time.sleep(1)

        # ── 流式 ──
        print("\n  [流式] 请求中...")
        s_ttft, s_total, s_tokens, s_content = benchmark_streaming(
            test["prompt"], test["max_tokens"]
        )
        s_speed = s_tokens / s_total if s_total > 0 else 0
        print(f"  [流式] 首字延迟: {s_ttft:.3f}s")
        print(f"  [流式] 总耗时:   {s_total:.3f}s, {s_tokens} tokens, {s_speed:.1f} tokens/s")
        print(f"  [流式] 回答前 80 字: {s_content[:80]}...")

        # ── 对比 ──
        time_diff = ns_time - s_total
        ttft_saving = ns_time - s_ttft if s_ttft > 0 else 0

        results.append({
            "name": test["name"],
            "ns_time": ns_time,
            "ns_tokens": ns_tokens,
            "s_ttft": s_ttft,
            "s_total": s_total,
            "s_tokens": s_tokens,
            "time_diff": time_diff,
            "ttft_saving": ttft_saving,
        })

        print(f"\n  [对比]")
        print(f"    非流式总耗时: {ns_time:.3f}s")
        print(f"    流式总耗时:   {s_total:.3f}s  (差值: {time_diff:+.3f}s)")
        print(f"    流式首字延迟: {s_ttft:.3f}s  (比非流式早 {ttft_saving:.3f}s 看到输出)")

        time.sleep(1)

    # ── 汇总报告 ──
    print("\n" + "=" * 70)
    print("【性能对比汇总】")
    print("=" * 70)
    print(f"{'测试项':<20} {'非流式(s)':<12} {'流式TTFT(s)':<14} {'流式总(s)':<12} {'首字提前(s)':<14}")
    print("─" * 70)

    for r in results:
        print(f"{r['name']:<20} {r['ns_time']:<12.3f} {r['s_ttft']:<14.3f} {r['s_total']:<12.3f} {r['ttft_saving']:<14.3f}")

    print("─" * 70)
    avg_ns = sum(r["ns_time"] for r in results) / len(results)
    avg_ttft = sum(r["s_ttft"] for r in results) / len(results)
    avg_s = sum(r["s_total"] for r in results) / len(results)
    print(f"{'平均值':<20} {avg_ns:<12.3f} {avg_ttft:<14.3f} {avg_s:<12.3f} {avg_ns - avg_ttft:<14.3f}")

    print("\n结论:")
    print(f"  - 流式首字延迟平均 {avg_ttft:.3f}s，用户可以更快看到输出开始")
    print(f"  - 非流式需要等待完整响应，平均 {avg_ns:.3f}s 后才显示任何内容")
    print(f"  - 流式让用户提前 {avg_ns - avg_ttft:.3f}s 看到首字输出，体验显著提升")
    print(f"  - 两种模式的总生成时间接近（流式略多因 SSE 开销）")


# ─────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    run_comparison()
    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


# ═══════════════════════════════════════════════════════
# 示例输出（实际数值因硬件、网络、负载不同会有差异）
# ═══════════════════════════════════════════════════════
#
# ======================================================================
# Hy3 API 性能对比：非流式 vs 流式
# Endpoint: http://127.0.0.1:8000/v1
# Model:    hy3
# ======================================================================
#
# ──────────────────────────────────────────────────────────────────────
# 测试 1/3: 短回答（约 100 字）
# Prompt: 用 100 字介绍 Python 的 GIL。
# max_tokens: 256
# ──────────────────────────────────────────────────────────────────────
#
#   [非流式] 请求中...
#   [非流式] 完成: 2.341s, 128 tokens, 54.7 tokens/s
#   [非流式] 回答前 80 字: GIL（全局解释器锁）是 CPython 中的机制，确保同一时刻
#   只有一个线程执行 Python 字节码...
#
#   [流式] 请求中...
#   [流式] 首字延迟: 0.312s
#   [流式] 总耗时:   2.398s, 128 tokens, 53.4 tokens/s
#   [流式] 回答前 80 字: GIL（全局解释器锁）是 CPython 中的机制，确保同一时刻
#   只有一个线程执行 Python 字节码...
#
#   [对比]
#     非流式总耗时: 2.341s
#     流式总耗时:   2.398s  (差值: -0.057s)
#     流式首字延迟: 0.312s  (比非流式早 2.029s 看到输出)
#
# ──────────────────────────────────────────────────────────────────────
# 测试 2/3: 中等回答（约 300 字）
# ...
#
# ======================================================================
# 【性能对比汇总】
# ======================================================================
# 测试项               非流式(s)    流式TTFT(s)    流式总(s)     首字提前(s)
# ──────────────────────────────────────────────────────────────────────
# 短回答（约 100 字）    2.341        0.312          2.398         2.029
# 中等回答（约 300 字）  5.872        0.298          5.934         5.574
# 长回答（约 500 字）    9.514        0.341          9.612         9.173
# ──────────────────────────────────────────────────────────────────────
# 平均值                 5.909        0.317          5.981         5.592
#
# 结论:
#   - 流式首字延迟平均 0.317s，用户可以更快看到输出开始
#   - 非流式需要等待完整响应，平均 5.909s 后才显示任何内容
#   - 流式让用户提前 5.592s 看到首字输出，体验显著提升
#   - 两种模式的总生成时间接近（流式略多因 SSE 开销）
#
# ======================================================================
# 示例完成！
# ======================================================================
