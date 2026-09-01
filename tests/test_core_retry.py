import asyncio

import pytest

from app.core.retry import (
    RetryConfig,
    RetryResult,
    retry_async_generator,
    retry_with_backoff,
)

FAST = RetryConfig(max_retries=3, total_timeout_seconds=5.0, base_delay_ms=10)


def test_retry_config_defaults():
    config = RetryConfig()
    assert (config.max_retries, config.total_timeout_seconds, config.base_delay_ms) == (
        5,
        2.0,
        50,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"total_timeout_seconds": 0},
        {"total_timeout_seconds": -1.0},
        {"base_delay_ms": 0},
        {"base_delay_ms": -5},
    ],
)
def test_retry_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RetryConfig(**kwargs)


def test_retry_result_defaults_errors_to_empty_list():
    assert RetryResult(success=True).errors == []


async def test_succeeds_on_first_attempt():
    calls = []

    async def func(value):
        calls.append(value)
        return value * 2

    result = await retry_with_backoff(func, 21, config=FAST)

    assert result.success
    assert result.result == 42
    assert result.attempts == 1
    assert result.errors == []
    assert calls == [21]


async def test_passes_through_keyword_arguments():
    async def func(a, *, b):
        return a + b

    result = await retry_with_backoff(func, 1, config=FAST, b=2)
    assert result.result == 3


async def test_retries_until_success():
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("backend down")
        return "ok"

    result = await retry_with_backoff(flaky, config=FAST)

    assert result.success
    assert result.result == "ok"
    assert result.attempts == 3
    assert len(result.errors) == 2
    assert "ConnectionError: backend down" in result.errors[0]


async def test_exhausts_all_retries_and_reports_failure():
    calls = {"n": 0}

    async def always_fails():
        calls["n"] += 1
        raise ValueError("nope")

    result = await retry_with_backoff(always_fails, config=FAST, operation_name="fetch")

    assert not result.success
    assert result.result is None
    assert calls["n"] == FAST.max_retries + 1
    assert result.attempts == FAST.max_retries + 1
    assert len(result.errors) == FAST.max_retries + 1
    assert result.total_duration_ms > 0


async def test_no_retry_when_max_retries_is_zero():
    calls = {"n": 0}

    async def always_fails():
        calls["n"] += 1
        raise RuntimeError("boom")

    result = await retry_with_backoff(
        always_fails,
        config=RetryConfig(max_retries=0, total_timeout_seconds=1.0, base_delay_ms=10),
    )

    assert not result.success
    assert calls["n"] == 1


async def test_exceptions_outside_retry_list_propagate():
    async def raises_key_error():
        raise KeyError("unexpected")

    with pytest.raises(KeyError):
        await retry_with_backoff(
            raises_key_error, config=FAST, retry_on_exceptions=(ConnectionError,)
        )


async def test_uses_exponential_backoff_delays(monkeypatch):
    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def always_fails():
        raise ConnectionError("down")

    config = RetryConfig(max_retries=3, total_timeout_seconds=30.0, base_delay_ms=100)
    result = await retry_with_backoff(always_fails, config=config)

    assert not result.success
    assert delays == [0.1, 0.2, 0.4]


async def test_delay_is_capped_by_remaining_timeout_budget(monkeypatch):
    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def always_fails():
        raise ConnectionError("down")

    config = RetryConfig(max_retries=5, total_timeout_seconds=0.5, base_delay_ms=1000)
    await retry_with_backoff(always_fails, config=config)

    assert delays
    assert all(delay <= config.total_timeout_seconds for delay in delays)


async def test_stops_retrying_once_timeout_budget_is_spent():
    calls = {"n": 0}

    async def slow_failure():
        calls["n"] += 1
        await asyncio.sleep(0.06)
        raise ConnectionError("slow and broken")

    config = RetryConfig(max_retries=20, total_timeout_seconds=0.15, base_delay_ms=10)
    result = await retry_with_backoff(slow_failure, config=config)

    assert not result.success
    assert calls["n"] < 20


async def test_retry_async_generator_yields_items():
    async def make_generator():
        async def gen():
            for item in ("a", "b", "c"):
                yield item

        return gen()

    items = [item async for item in retry_async_generator(make_generator, config=FAST)]
    assert items == ["a", "b", "c"]


async def test_retry_async_generator_retries_before_streaming():
    attempts = {"n": 0}

    async def make_generator():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("connect failed")

        async def gen():
            yield "chunk"

        return gen()

    items = [item async for item in retry_async_generator(make_generator, config=FAST)]

    assert items == ["chunk"]
    assert attempts["n"] == 2


async def test_retry_async_generator_raises_after_exhausting_retries():
    async def make_generator():
        raise ConnectionError("never connects")

    config = RetryConfig(max_retries=1, total_timeout_seconds=1.0, base_delay_ms=10)

    with pytest.raises(Exception, match="Failed after 2 attempts"):
        [item async for item in retry_async_generator(make_generator, config=config)]
