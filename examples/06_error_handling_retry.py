"""
Hy3 API Example 06: Error Handling & Retry (错误处理与重试)
=============================================================

本示例演示：
1. 常见错误类型识别与处理
2. 指数退避重试策略
3. 超时处理
4. 频率限制（429）处理
5. 带降级的健壮调用封装

运行：
  python 06_error_handling_retry.py
"""

import os
import time
import random
from typing import Optional, Callable
from openai import (
    OpenAI,
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    APIStatusError,
    InternalServerError,
)

# ─────────────────────────────────────────────────────
# 初始化客户端
# ─────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "EMPTY"),
    base_url=os.environ.get("HY3_BASE_URL", "http://127.0.0.1:8000/v1"),
    timeout=60.0,       # 默认超时 60 秒
    max_retries=0,       # 关闭 SDK 内置重试，使用自定义重试逻辑
)

MODEL = os.environ.get("HY3_MODEL", "hy3")


# ─────────────────────────────────────────────────────
# 1. 指数退避重试装饰器
# ─────────────────────────────────────────────────────

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (
        APITimeoutError,
        RateLimitError,
        APIConnectionError,
        InternalServerError,
    ),
):
    """
    指数退避重试装饰器。

    Args:
        max_retries: 最大重试次数
        base_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        retryable_exceptions: 可重试的异常类型

    重试延迟计算：delay = min(base_delay * 2^attempt + jitter, max_delay)
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        print(f"  [重试] 达到最大重试次数 {max_retries}，放弃")
                        raise

                    # 计算退避延迟（指数 + 随机抖动）
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)

                    # 根据错误类型调整策略
                    if isinstance(e, RateLimitError):
                        print(f"  [重试] 429 频率限制，等待 {delay:.1f}s 后重试 "
                              f"(第 {attempt + 1}/{max_retries} 次)")
                    elif isinstance(e, APITimeoutError):
                        print(f"  [重试] 请求超时，等待 {delay:.1f}s 后重试 "
                              f"(第 {attempt + 1}/{max_retries} 次)")
                    elif isinstance(e, APIConnectionError):
                        print(f"  [重试] 连接错误，等待 {delay:.1f}s 后重试 "
                              f"(第 {attempt + 1}/{max_retries} 次)")
                    elif isinstance(e, InternalServerError):
                        print(f"  [重试] 服务端 500 错误，等待 {delay:.1f}s 后重试 "
                              f"(第 {attempt + 1}/{max_retries} 次)")

                    time.sleep(delay)

                except APIStatusError as e:
                    # 不可重试的 HTTP 错误（400, 401, 403, 404 等）
                    print(f"  [错误] HTTP {e.status_code}: 不可重试，直接抛出")
                    raise

            raise last_exception

        return wrapper
    return decorator


# ─────────────────────────────────────────────────────
# 2. 带重试的聊天函数
# ─────────────────────────────────────────────────────

@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
def chat_with_retry(messages, **kwargs):
    """
    带指数退避重试的聊天调用。

    可重试的错误：超时、429 频率限制、连接错误、500 服务端错误
    不可重试的错误：400 参数错误、401 认证失败、404 模型不存在
    """
    defaults = {
        "model": MODEL,
        "temperature": 0.9,
        "top_p": 1.0,
    }
    defaults.update(kwargs)

    return client.chat.completions.create(
        messages=messages,
        **defaults,
    )


# ─────────────────────────────────────────────────────
# 3. 带降级链的健壮调用
# ─────────────────────────────────────────────────────

def robust_chat(
    messages,
    primary_config: Optional[dict] = None,
    fallback_config: Optional[dict] = None,
    max_retries: int = 3,
):
    """
    带降级的健壮调用。

    优先使用主配置调用，失败后尝试降级配置：
    1. 主配置（高 max_tokens + 推理模式）
    2. 降级配置（低 max_tokens + no_think）
    3. 最终降级（最小请求）
    """
    primary_config = primary_config or {
        "max_tokens": 2048,
        "extra_body": {"chat_template_kwargs": {"reasoning_effort": "high"}},
    }
    fallback_config = fallback_config or {
        "max_tokens": 512,
        "extra_body": {"chat_template_kwargs": {"reasoning_effort": "no_think"}},
    }
    final_config = {
        "max_tokens": 128,
        "extra_body": {"chat_template_kwargs": {"reasoning_effort": "no_think"}},
    }

    configs = [
        ("主配置", primary_config),
        ("降级配置", fallback_config),
        ("最终降级", final_config),
    ]

    for config_name, config in configs:
        try:
            print(f"  [调用] 使用 {config_name}...")
            response = chat_with_retry(messages, **config)
            print(f"  [成功] {config_name} 调用成功")
            return response, config_name
        except Exception as e:
            print(f"  [失败] {config_name} 调用失败: {type(e).__name__}: {e}")
            if config_name == "最终降级":
                raise
            print(f"  [降级] 切换到下一配置...")
            time.sleep(1)

    # 不应到达这里
    raise RuntimeError("所有配置均失败")


# ─────────────────────────────────────────────────────
# 4. 错误类型演示
# ─────────────────────────────────────────────────────

def demonstrate_error_types():
    """演示各种错误类型的识别和处理。"""
    print("=" * 60)
    print("【错误类型识别】")
    print("=" * 60)

    error_scenarios = [
        {
            "name": "正常调用（应成功）",
            "messages": [{"role": "user", "content": "1+1=?"}],
            "kwargs": {},
        },
        {
            "name": "不存在的模型（404）",
            "messages": [{"role": "user", "content": "Hello"}],
            "kwargs": {"model": "nonexistent-model-xyz"},
        },
        {
            "name": "空消息列表（400）",
            "messages": [],
            "kwargs": {},
        },
        {
            "name": "无效参数（400）",
            "messages": [{"role": "user", "content": "test"}],
            "kwargs": {"temperature": -1.0},
        },
    ]

    for scenario in error_scenarios:
        print(f"\n--- {scenario['name']} ---")
        try:
            response = client.chat.completions.create(
                model=scenario["kwargs"].get("model", MODEL),
                messages=scenario["messages"],
                temperature=scenario["kwargs"].get("temperature", 0.9),
                top_p=1.0,
                max_tokens=64,
            )
            print(f"  [成功] 回答: {response.choices[0].message.content[:50]}")

        except APIStatusError as e:
            print(f"  [HTTP错误] 状态码: {e.status_code}")
            print(f"  [HTTP错误] 消息: {str(e)[:200]}")
            print(f"  [处理建议] ", end="")
            if e.status_code == 400:
                print("检查请求参数格式")
            elif e.status_code == 401:
                print("检查 API Key 是否正确")
            elif e.status_code == 404:
                print("检查 model 名称是否正确")
            elif e.status_code == 422:
                print("检查请求体 JSON 格式")
            elif e.status_code == 429:
                print("降低请求频率，使用指数退避重试")
            elif e.status_code >= 500:
                print("服务端错误，稍后重试")

        except APITimeoutError:
            print(f"  [超时错误] 请求超时")
            print(f"  [处理建议] 增加 timeout 值或使用流式输出")

        except APIConnectionError as e:
            print(f"  [连接错误] {str(e)[:200]}")
            print(f"  [处理建议] 检查服务是否启动、网络是否正常")

        except Exception as e:
            print(f"  [未知错误] {type(e).__name__}: {str(e)[:200]}")

        time.sleep(0.5)


# ─────────────────────────────────────────────────────
# 5. 重试策略演示
# ─────────────────────────────────────────────────────

def demonstrate_retry():
    """演示指数退避重试的实际效果。"""
    print("\n" + "=" * 60)
    print("【指数退避重试演示】")
    print("=" * 60)

    print("\n正常调用（带重试保护）：")
    print("-" * 40)

    try:
        start = time.time()
        response = chat_with_retry(
            [{"role": "user", "content": "用一个词回答：水在零下会变成什么？"}],
            max_tokens=32,
        )
        elapsed = time.time() - start
        print(f"  回答: {response.choices[0].message.content}")
        print(f"  耗时: {elapsed:.3f}s")
        print(f"  tokens: {response.usage.total_tokens}")

    except Exception as e:
        print(f"  最终失败: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────
# 6. 降级链演示
# ─────────────────────────────────────────────────────

def demonstrate_fallback():
    """演示降级调用链。"""
    print("\n" + "=" * 60)
    print("【降级链演示】")
    print("=" * 60)

    messages = [{"role": "user", "content": "用三句话介绍量子计算的基本原理。"}]

    try:
        response, used_config = robust_chat(messages)
        print(f"\n  [最终回答] (使用 {used_config})")
        print(f"  {response.choices[0].message.content}")
        print(f"\n  tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"\n  所有配置均失败: {e}")


# ─────────────────────────────────────────────────────
# 7. 并发限流模拟
# ─────────────────────────────────────────────────────

def demonstrate_rate_limit_handling():
    """模拟并发请求场景下的限流处理。"""
    print("\n" + "=" * 60)
    print("【并发限流处理】")
    print("=" * 60)

    import concurrent.futures

    prompts = [
        "1+1=?",
        "2+2=?",
        "3+3=?",
        "4+4=?",
        "5+5=?",
    ]

    def safe_chat(prompt):
        """线程安全的聊天调用。"""
        try:
            response = chat_with_retry(
                [{"role": "user", "content": prompt}],
                max_tokens=16,
            )
            return prompt, response.choices[0].message.content.strip(), True
        except Exception as e:
            return prompt, str(e), False

    print(f"\n并发发送 {len(prompts)} 个请求...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(safe_chat, p) for p in prompts]

        for future in concurrent.futures.as_completed(futures):
            prompt, answer, success = future.result()
            status = "✓" if success else "✗"
            print(f"  {status} {prompt} → {answer[:30]}")


# ─────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Hy3 API Example 06: Error Handling & Retry")
    print(f"Endpoint: {os.environ.get('HY3_BASE_URL', 'http://127.0.0.1:8000/v1')}")
    print(f"Model:    {MODEL}")
    print()

    demonstrate_error_types()
    demonstrate_retry()
    demonstrate_fallback()
    demonstrate_rate_limit_handling()

    print("\n" + "=" * 60)
    print("错误处理最佳实践总结")
    print("=" * 60)
    print("""
  1. 设置合理超时：千亿参数模型建议 60-180 秒
  2. 指数退避重试：对 429/500/超时/连接错误自动重试
  3. 不可重试错误：400/401/403/404 直接抛出，不重试
  4. 降级策略：高配失败 → 低配 → 最小请求
  5. 并发控制：限制并发数，避免触发限流
  6. 监控告警：记录错误率和延迟，及时发现问题
  7. 预算保护：设置 Token/费用上限，防止异常消耗
    """)

    print("=" * 60)
    print("示例完成！")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
# 示例输出（实际输出因环境不同会有差异）
# ═══════════════════════════════════════════════════════
#
# ============================================================
# 【错误类型识别】
# ============================================================
#
# --- 正常调用（应成功） ---
#   [成功] 回答: 2
#
# --- 不存在的模型（404） ---
#   [HTTP错误] 状态码: 404
#   [HTTP错误] 消息: Error code: 404 - {'error': {'message': 'Model not found', 'type': ...}}
#   [处理建议] 检查 model 名称是否正确
#
# --- 空消息列表（400） ---
#   [HTTP错误] 状态码: 400
#   [HTTP错误] 消息: Error code: 400 - {'error': {'message': 'messages is required', ...}}
#   [处理建议] 检查请求参数格式
#
# --- 无效参数（400） ---
#   [HTTP错误] 状态码: 400
#   [HTTP错误] 消息: Error code: 400 - {'error': {'message': 'temperature must be ...'}}
#   [处理建议] 检查请求参数格式
#
# ============================================================
# 【指数退避重试演示】
# ============================================================
#
# 正常调用（带重试保护）：
# ----------------------------------------
#   回答: 冰
#   耗时: 0.856s
#   tokens: 12
#
# ============================================================
# 【降级链演示】
# ============================================================
#
#   [调用] 使用 主配置...
#   [成功] 主配置 调用成功
#
#   [最终回答] (使用 主配置)
#   量子计算利用量子叠加和纠缠特性进行计算。量子比特可以同时表示
#   0 和 1，通过量子门操作实现并行计算。这使得量子计算机在特定
#   问题（如大数分解）上远超经典计算机。
#
#   tokens: 89
#
# ============================================================
# 【并发限流处理】
# ============================================================
#
# 并发发送 5 个请求...
#   ✓ 1+1=? → 2
#   ✓ 2+2=? → 4
#   ✓ 3+3=? → 6
#   ✓ 4+4=? → 8
#   ✓ 5+5=? → 10
#
# ============================================================
# 错误处理最佳实践总结
# ============================================================
#
#   1. 设置合理超时：千亿参数模型建议 60-180 秒
#   2. 指数退避重试：对 429/500/超时/连接错误自动重试
#   3. 不可重试错误：400/401/403/404 直接抛出，不重试
#   4. 降级策略：高配失败 → 低配 → 最小请求
#   5. 并发控制：限制并发数，避免触发限流
#   6. 监控告警：记录错误率和延迟，及时发现问题
#   7. 预算保护：设置 Token/费用上限，防止异常消耗
#
# ============================================================
# 示例完成！
# ============================================================
