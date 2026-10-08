"""A failed request, and a server that will not exit, must not cost the node.

Both failure modes were observed together on 2026-10-07: one prompt longer than
``max_model_len`` returned a 400, ``asyncio.gather`` propagated it while the
other requests kept running, and the teardown then waited forever for an API
server that was itself waiting for those requests' connections to close. Four
NPUs sat idle until the job was killed by hand.
"""

from __future__ import annotations

import asyncio
import subprocess
import types

import pytest

from moe_reliability.core import vllm_serving as VS


class _Client:
    """Fails the prompts whose index is in ``fail_on``, with ``error``."""

    def __init__(self, fail_on=(), error=None):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
        self.fail_on, self.error = set(fail_on), error or ValueError("400 Bad Request")
        self.started, self.finished = 0, 0

    async def _create(self, **kw):
        self.started += 1
        index = int(kw["messages"][0]["content"].split()[-1])
        await asyncio.sleep(0.01)
        if index in self.fail_on:
            raise self.error
        self.finished += 1
        usage = types.SimpleNamespace(completion_tokens=1, prompt_tokens=5)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="x"), routed_experts=None)],
            usage=usage, prompt_routed_experts=None)


def _prompts(n):
    return [[{"role": "user", "content": f"prompt {i}"}] for i in range(n)]


def _batch(client, n, **kw):
    return asyncio.run(VS.run_batch(client, "m", _prompts(n), max_new_tokens=1,
                                    concurrency_limit=4, capture_experts=True, **kw))


def test_a_rejected_prompt_does_not_take_the_batch_with_it(capsys):
    client = _Client(fail_on={3})
    results = _batch(client, 20)

    assert len(results) == 19                      # the batch still measures something
    assert 3 not in {r["prompt_id"] for r in results}
    assert client.finished == 19                   # and every other request ran to completion
    assert "1 of 20 requests failed" in capsys.readouterr().out


def test_every_request_is_awaited_so_none_outlives_the_batch():
    # An orphaned request holds its connection open, which is what deadlocked the
    # teardown: the server waits for the connection, the client waits for the server.
    client = _Client(fail_on={0})
    _batch(client, 12)
    assert client.started == 12 and client.finished == 11


def test_a_batch_that_mostly_failed_is_an_error_not_a_measurement():
    client = _Client(fail_on=set(range(10)))
    with pytest.raises(RuntimeError, match="10 of 20 requests failed"):
        _batch(client, 20)


def test_the_failure_budget_scales_with_the_batch():
    # 5 of 1000 is within budget; the same 5 of 20 is not.
    assert VS.MAX_FAILED_REQUESTS == 5
    client = _Client(fail_on=set(range(5)))
    assert len(_batch(client, 1000)) == 995

    client = _Client(fail_on=set(range(6)))
    with pytest.raises(RuntimeError, match="6 of 20"):
        _batch(client, 20)


class _Process:
    """A server process that ignores SIGTERM, as one with open connections does."""

    def __init__(self, pid=4242):
        self.pid = pid
        self.waits: list = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if timeout is None:
            raise AssertionError("teardown waited without a timeout; a server that will not "
                                 "exit would hold the NPUs forever")
        raise subprocess.TimeoutExpired("vllm", timeout)


class _Group:
    """A process group that ignores SIGTERM and dies only on SIGKILL."""

    def __init__(self, dies_on_sigterm=False):
        self.alive, self.signals = True, []
        self.dies_on_sigterm = dies_on_sigterm

    def killpg(self, pgid, sig):
        if sig == 0:                      # the liveness probe
            if not self.alive:
                raise ProcessLookupError
            return
        self.signals.append(sig)
        if sig == VS.signal.SIGKILL or (sig == VS.signal.SIGTERM and self.dies_on_sigterm):
            self.alive = False


@pytest.fixture
def fake_group(monkeypatch):
    monkeypatch.setattr(VS.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(VS.time, "sleep", lambda s: None)
    return monkeypatch


def test_teardown_escalates_to_sigkill_when_the_server_will_not_exit(fake_group):
    group = _Group()
    fake_group.setattr(VS.os, "killpg", group.killpg)
    process = _Process()

    VS.stop_vllm_server(process, timeout=5.0)

    assert group.signals == [VS.signal.SIGTERM, VS.signal.SIGKILL]
    assert not group.alive
    # The bounded wait is what lets the escalation happen at all.
    assert process.waits and all(t is not None for t in process.waits)


def test_a_server_that_exits_on_sigterm_is_not_killed(fake_group):
    group = _Group(dies_on_sigterm=True)
    fake_group.setattr(VS.os, "killpg", group.killpg)

    VS.stop_vllm_server(_Process(), timeout=5.0)

    assert group.signals == [VS.signal.SIGTERM]   # no SIGKILL needed
