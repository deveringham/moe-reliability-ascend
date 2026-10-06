"""Inter-token latency capture in the streaming client, against a fake stream."""

from __future__ import annotations

import asyncio
import time
import types

from moe_reliability.core.vllm_serving import measure_request


def _chunk(content=None, usage=None):
    delta = types.SimpleNamespace(content=content)
    choices = [types.SimpleNamespace(delta=delta)] if content is not None or usage is None else []
    return types.SimpleNamespace(choices=choices, usage=usage)


class _Stream:
    """Yields chunks, sleeping the given gaps so real timestamps are taken."""

    def __init__(self, gaps_s, n_output, n_input=7):
        self.gaps_s, self.n_output, self.n_input = gaps_s, n_output, n_input

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i, gap in enumerate(self.gaps_s):
            if i:
                await asyncio.sleep(gap)
            yield _chunk(content=f"t{i}")
        usage = types.SimpleNamespace(completion_tokens=self.n_output, prompt_tokens=self.n_input)
        yield _chunk(usage=usage)


class _Client:
    def __init__(self, stream):
        inner = types.SimpleNamespace(create=self._create)
        self.chat = types.SimpleNamespace(completions=inner)
        self._stream = stream

    async def _create(self, **kw):
        self._kwargs = kw
        return self._stream


def _run(gaps, n_output, **kw):
    client = _Client(_Stream(gaps, n_output))
    return asyncio.run(measure_request(client, "m", 0, "hello", max_new_tokens=n_output, **kw))


def test_itl_series_records_one_gap_between_consecutive_chunks():
    res = _run([0.0, 0.02, 0.02, 0.05], n_output=4, collect_itl=True)
    assert res["n_chunks"] == 4
    assert len(res["itl_ms"]) == 3  # gaps, not chunks
    assert res["itl_ms"][2] > res["itl_ms"][0]  # the 50 ms gap is the largest
    assert all(v >= 0 for v in res["itl_ms"])


def test_itl_capture_is_off_by_default_and_adds_no_fields():
    res = _run([0.0, 0.01], n_output=2)
    assert "itl_ms" not in res and "n_chunks" not in res


def test_a_single_token_request_has_no_gaps():
    res = _run([0.0], n_output=1, collect_itl=True)
    assert res["n_chunks"] == 1
    assert res["itl_ms"] == []


def test_itl_sum_is_consistent_with_the_tpot_it_should_decompose():
    # TPOT is (end - first token) / (tokens - 1); the gaps span the same interval,
    # so their mean must agree with it. This is the property the analysis relies on.
    res = _run([0.0, 0.03, 0.03, 0.03], n_output=4, collect_itl=True)
    mean_itl_s = sum(res["itl_ms"]) / len(res["itl_ms"]) / 1000
    assert abs(mean_itl_s - res["tpot"]) < 0.01
