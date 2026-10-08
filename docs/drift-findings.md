# Router Drift Findings

2026-10-02, extended 2026-10-05 with the routing-determinism measurements.

Can a detector tell, from expert load alone, that the router has changed, as
opposed to the traffic? On DeepSeek-V2-Lite: **yes for gross faults, no for
subtle ones.** A router collapse to 6 of 64 experts separates perfectly. A router
that misroutes fewer than roughly 10-20% of its assignments is hidden by ordinary
topical variation in traffic. Replaying a fixed canary set of prompts removes that
variation by construction, and the recorded data suggest it lowers the floor by
about 80x.

This is a claim about the router, not about serving cost. Expert token counts
overstate fused-MoE time imbalance by about six times on this stack (see
[imbalance-findings.md](imbalance-findings.md)), so a drift score does not predict
latency and is not offered as a predictor.

## Detector

`moe_reliability_results.drift`. For each layer, normalise the window's expert
counts to a distribution and take the Jensen-Shannon divergence from a reference,
in bits, then average over layers. JS is symmetric and bounded in [0, 1], so a
single unseen expert cannot dominate it. The module offers two forms:

- `marginal_drift`: pooled distributions, which is what a load counter sees. This
  is the baseline.
- `stratified_drift`: divergence within strata of prompts, weighted by the
  reference's mix. A change in which prompts arrive cancels exactly when the
  strata capture what drives routing (tested). A router change survives.

Alarm thresholds come from empirical quantiles of benign windows
(`threshold_at_fpr`), not from a chi-square null. Tokens within one prompt route
alike, so the effective sample size is far below the token count.

## Separation, 2026-10-02

Offline, on the 3000 MMLU prompts captured in
`20261001-133042_..._alpha-sweep` (DeepSeek-V2-Lite-Chat, 4 NPUs, greedy, 100 new
tokens). Prompt and generated tokens are pooled.

| Condition | Drift (bits) | What it represents |
|---|---|---|
| Stable-mix windows | 0.0014 (p99 0.0028) | Sampling noise floor |
| Single-subject windows | 0.0107 (p99 0.0199) | Benign workload shift |
| Router collapse to 6 experts | 0.7342 | Gross fault (bias-100 checkpoint, `capture-imb100`) |

- **Gross faults are solved.** The collapse scores 37x above the benign p99, AUC
  1.000. The collapse is too easy a positive to say anything about sensitivity,
  though.
- **The sensitivity floor is 10-20% misrouting.** The fault was graded: the same
  prompts and token counts, with a fraction of assignments re-drawn from a biased
  subset of experts. The router has to misroute roughly 10-20% of assignments
  before its score clears the benign-shift p99.
- **Workload, not statistics, sets the floor.** Benign topic shift sits 7x above
  the sampling floor. A threshold calibrated on stable traffic (1% FPR, 0.0029
  bits) false-alarms on essentially every single-subject window.
- **Stratifying by length does not help.** Benign drift goes from 0.0107 to
  0.0115, slightly worse. Topic drives routing and length does not, and the strata
  thin the sample. Topic itself cannot be observed in production.

### Reproducibility of these numbers

The 2026-10-02 evaluation harness was not committed: window size, window count and
the misrouting generator are unrecorded, and the grid behind "10-20%" is known
only as that range. Re-checked on 2026-10-05 with the committed code:

- Collapse against the full reference, pooled: **0.757** bits against 0.734
  reported. That is consistent, given the unrecorded windowing.
- Random windows against the full reference: mean 0.0007 / p99 0.0016 at 50
  records, falling to 0.0002 / 0.0003 at 200. The floor depends on window size,
  so 0.0014 cannot be pinned to a configuration without the harness.

Treat the table as order-of-magnitude until the harness is rebuilt. The
conclusions, the ratios between conditions, do not depend on the exact window.

## Does the same input produce the same routing? 2026-10-05

No. That is the obvious premise of a canary, so it was measured directly
(`scripts/routing_determinism.py`). Two natural experiments exist in the recorded
captures:

- **Same server:** the sweep capture drew 3000 prompts with replacement and served
  90 of them twice, on the same server and config, about 735 requests apart and so
  in different batches.
- **Across configs:** all 300 prompts of the smoke capture (8 NPUs, batch 128)
  recur in the sweep capture (4 NPUs, batch 512).

All runs decode greedily with prefix caching off.

| | Same server | Across configs |
|---|---|---|
| Prompts compared | 90 | 300 |
| Prompt-token assignments changed | 1.87% (1.43-2.66% per prompt) | 2.00% (1.27-2.80%) |
| Bit-identical prompts | 0 | 0 |
| (token, layer) slots: one expert swapped | 10.7% | 11.4% |
| (token, layer) slots: two or more swapped | 0.26% | 0.28% |
| Generated-token assignments changed | 41.8% | 40.0% |
| Pooled prompt-token JS between copies | 0.00001 bits | 0.00001 bits |

**Prefill is nearly deterministic, and the residue is numerical.** About 2% of
assignments differ on identical prompts, and every prompt is affected. The
differences are almost all one expert swapped at the top-6 boundary, which is the
signature of near-tied router logits resolved differently by batch-dependent
floating-point reduction order. They grow with depth, from 0.3% in the first MoE
layer to about 3% in the last, as small differences in the residual stream
accumulate. Changing the parallelism and batch size adds almost nothing (1.87% to
2.00%), so batch composition, not configuration, is the source.

**Decode is not usable for a canary.** Generated tokens start near the prefill
rate (1.6% at the first token) and climb to 70% by token 98. A single flipped
expert changes a logit somewhere, greedy decoding picks a different token, and from
then on the routing is over different text. In 246 of 300 cross-config prompts some
generated token had more than 20% of its assignments changed, a proxy for diverged
text, at a median of token 32.

**The flips are symmetric, so the distribution barely moves.** Pooled over the 90
repeated prompts, the two copies differ by 0.00001 bits. Two sets of 90 *different*
prompts differ by 0.00083 bits on average (minimum 0.00032), so replaying the same
prompts lowers the floor by about 80x. The single-subject benign shift, at 0.0107,
is about 1000x above it.

## What this means for the canary

1. **Use prompt tokens only.** Send canaries with `max_tokens = 1`, or discard
   generated-token routing. Decode routing on identical prompts is 40% different.
2. **Expect a noise floor of about 2% per token, and near zero in distribution.**
   That gives two complementary scores:
   - *Pooled JS on the canary set*: floor about 0.00001 bits. It sees directed
     faults that shift load, and is blind to faults that permute experts of
     similar load.
   - *Per-token change rate against a stored reference*: floor about 1.9%, nearly
     all single swaps. It also sees load-preserving faults, which no
     distributional score can.
3. **Calibrate the floor on the same server.** The 1.87% and 0.00001 bits are
   from one capture of 90 prompts. A canary set should be replayed a few times
   against an unchanged router to set its own threshold.
4. **Expected sensitivity, untested:** if the marginal floor drops 80x and drift
   grows with misrouting fraction, the detection floor should fall from 10-20%
   to low single digits. That is an extrapolation. The graded-misrouting
   evaluation must be rerun on canary windows to confirm it.

A batch-invariant serving mode would remove the 2% floor entirely, so the
comparison could be exact. Whether vllm-ascend offers one has not been checked.

## Caveats

- **One model, offline.** DeepSeek-V2-Lite only. All faults were either the bias-100
  checkpoint, which also collapses the live expert count, or edits to recorded
  routing. No graded fault has been injected into a running router.
- **The prefill/decode split in every capture before 2026-10-05 was wrong.**
  `prompt_routed_experts` held the first 100 prompt tokens and `routed_experts` the
  rest, a bug in `measure_request`, now fixed. The drift table above pools both
  arrays and is unaffected. The determinism analysis re-splits at
  `num_input_tokens`. See [data-format.md](data-format.md).
- **The repeated prompts are a natural experiment.** The 90 duplicates came from
  sampling with replacement, not from design, and both copies ran in one server
  lifetime. Replay across server restarts is unmeasured.

## Next

1. Commit the graded-misrouting generator and window harness alongside `drift.py`,
   with tests, and regenerate the table from it.
2. Rerun that evaluation on canary windows: prompt-token routing of a fixed set,
   both scores, the floor calibrated from repeat replays. Most of this can use the
   existing captures, by treating the duplicated prompts as a miniature canary set.
3. On the node, capture a deliberate canary set several times against an unchanged
   router, including across a server restart, to measure the real floor.
4. Then inject graded faults into a live router, such as a small per-expert gate
   bias at increasing strength, and score the canary end to end.
