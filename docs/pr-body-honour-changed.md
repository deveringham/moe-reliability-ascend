### What this PR does / why we need it?

Each EPLB policy returns `(changed, per_layer_priority, new_placement)`, and `policy_abstract.py` documents the first value as `1 -- table_changed`, `0 -- not_changed`. However, `EplbWorker.do_update` ignores `changed`:

```python
_, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)
```

Therefore currently every plan is applied, including plans which the policy judged not worth applying. In `DefaultEplb` and `SwiftBalanceEplb`, `changed` is 1 only when the plan cuts the summed per-layer peak load by at least 5%. A plan below that threshold still moves expert weights between NPUs (although this may not reflect any actual performance gain).

On our test platform, for four ranks of two experts each and a perfectly balanced per-rank load, `DefaultEplb` returns `changed = 0` and a new placement. Currently, that placement is applied and experts are moved, incurring migration overhead to execute a plan the policy scored as not worth making.

This PR retains the flag value and maintains the current expert placement when it is 0.

No new control flow is required. When `changed` is 0, every layer compares equal in `compose_expert_update_info_greedy` which yields empty send and receive sets, and the loader completes the current cycle without moving anything. This path already runs for any layer whose expert mappin does not change. Returning early instead would not be safe, because `forward_before` calls `pop(0)` on whatever the worker queues.

The same path also replaces the early return when no load measurement is available. That return queues `None`, which `forward_before` stores and the next `pop(0)` turns into an `AttributeError`. Nothing reaches this error currently because `warm_up_eplb` always gathers a load mesurement before the first cycle, but this patch prevents the potential crash. The cost is that such a cycle now executes the remainder of `do_update` when is currently does not, instead of returning immediately: `check_expert_placement`, `local2global`, `pack_update_info`. This keeps the queue aligned with the cycle.

There is precedent upstream. vLLM core skips a rearrangement when the relative improvement in imbalance is under 5% (`vllm/distributed/eplb/eplb_state.py`, added in vllm-project/vllm#47206, currently AMD ROCm only).

`per_layer_priority` is still discarded. It would matter if the number of layers moved per cycle were ever capped, but this PR leaves it alone.

This PR and #19288 (EPLB recording mode) both change `EplbWorker.do_update` and both add `tests/ut/eplb/core/test_eplb_worker.py`, so whichever merges second needs a small rebase.

**AI assistance:** this PR was prepared with Claude (Anthropic), which helped partially write the patch code and tests. I have reviewed every changed line.

### Does this PR introduce _any_ user-facing change?

Yes. With policies 1 and 2, EPLB now skips a rearrangement whose plan improves balance by less than 5%, and logs that it did. This occurs at most once per rearrangement cycle per rank, so the result is a few lines per minute at default intervals. Policy 3 (FlashLB) sets the flag whenever any layer changes and policy 0 (random) always sets it, so their behaviour is unchanged. If maintainers prefer, this can sit behind a config option defaulting to the current behaviour.

### How was this patch tested?

**Unit tests.** `tests/ut/eplb/core/test_eplb_worker.py`, ten cases, with `tests/ut/eplb` successful: 29 passed, against 19 before these tests.

Against a stub policy, `changed = 0` moves nothing and leaves the published map untouched. `changed = 1` is applied as before with `np.int64` and `np.bool_` flags are honoured. An unexpected flag value is treated as "apply", preserving the current behaviour for a policy that returns an unforseen value. A missing load measurement returns a no-op plan instead of `None` and a placement with an expert replicated across ranks survives the keep-placement path through `global2local`/`local2global`.

Against the real policies, with no stub, on four ranks of two experts with a balanced load, `DefaultEplb` returns `changed = 0` with a different placement and the test asserting that nothing moves fails without this change. `SwiftBalanceEplb` happens to return an identical placement in that case so it is covered as a guard rather than a regression.

**On hardware.** DeepSeek-V2-Lite-Chat, 4x Ascend 910B3, policy 2, heat collection interval 50, algorithm execution interval 50, 1200 requests at 300 generated tokens, with a router bias giving a 1.72x busiest-rank load. Two points were observed:

| point | plans | plans declined | skip log lines | rank-consistent | busiest rank, by EPLB's own `_compute_imbalance` |
|---|---|---|---|---|---|
| 1 | 12 | 7 | 28 | yes (28 = 7 x 4) | 1.676 -> 1.016 |
| 2 | 12 | 5 | 20 | yes (20 = 5 x 4) | 1.676 -> 1.040 |

Every rank logs its own exper mapping decision so the line count divides exactly by the world size only if all four ranks declined the same plans. They did on every plan in both points. An unmodified control at the same settings and skew converged to 1.024x so declining the marginal plans does not stop EPLB balancing. The impactful plans that matter are the early ones and they all pass the 5% threshold.

No latency claim is made. The flag asks whether the plan improves balance, not whether the imbalance costing anythign in terms of performance.

---

### Checklist

#### Testing

- [x] **Boundary / edge-case coverage**: zero and non-zero flags, `np.bool_`/`np.int64` flag types, a flag of an unexpected type, a missing load measurement, and a placement with an expert replicated across ranks.
- [x] **Regression test**: five cases fail without this change, including one driven by the real `DefaultEplb`.
- [x] **Error / failure path coverage**: the missing-load measurement path that would otherwise queue `None` into an `AttributeError`.
- [x] **Meaningful & reliable tests**: the real-policy tests assert what the shipped policies actually return, not what a mock was told to return.
