# Contributions to vllm-ascend / vLLM

Two bugfix PRs for vllm-ascend - **submitted 2026-10-20 as #19288 (A) and
#19290 (B)** - and one RFC for vLLM core that is not ready to send.

| # | branch | title | body |
|---|---|---|---|
| A | `fix-eplb-record-mode` | `[BugFix][EPLB] Recording mode silent failure` ([#19288](https://github.com/vllm-project/vllm-ascend/pull/19288)) | `docs/pr-body-record-mode.md` |
| B | `fix-eplb-honour-changed` | `[BugFix][EPLB] Honour a policy's changed flag instead of discarding it` ([#19290](https://github.com/vllm-project/vllm-ascend/pull/19290)) | `docs/pr-body-honour-changed.md` |
| C | - | `[RFC][EPLB] Optional imbalance impact threshold before rearranging experts` (vLLM core) | not ready; drafted below |

The two body files are ready to pass to `gh pr create --body-file`. They contain
only what upstream needs. The caveats that are ours rather than theirs are in
this file and in `docs/upstream-eplb.md`.

## State of the branches

Both live on a clone at `~/src/vllm-ascend` (`origin` = the fork
`deveringham/vllm-ascend`, `upstream` = `vllm-project`), rebased onto the fork's
`main` at `9a885476a`, DCO-signed with a `Co-authored-by: Claude` trailer, one
commit each. Rebase onto the *fork's* `main`, not upstream's: a branch older than
the fork's default branch on any `.github/workflows` file is refused on push
without the token's `workflow` scope. The EPLB sources and `tests/ut/eplb` are
identical between `ee29cc6fc` (where the tests ran) and `9a885476a`.

Each branch touches only upstream files — no experiment configs, no analysis
code, nothing from this project:

| branch | files |
|---|---|
| A | `vllm_ascend/eplb/eplb_updator.py`, `vllm_ascend/eplb/core/eplb_worker.py`, `vllm_ascend/worker/model_runner_v1.py`, `tests/ut/eplb/test_eplb_updator.py`, `tests/ut/eplb/core/test_eplb_worker.py` (new) |
| B | `vllm_ascend/eplb/core/eplb_worker.py`, `tests/ut/eplb/core/test_eplb_worker.py` (new) |

Checked: ruff format and ruff check clean, mypy 1.11.1 clean on the changed
files, `tests/ut/eplb` green on both, and no occurrence of this project's
names anywhere in either diff.

Both branch from `main` and both touch `EplbWorker.do_update`, and each adds
`tests/ut/eplb/core/test_eplb_worker.py` as a new file, so whichever is merged
second needs a trivial rebase. Say so in the second PR's description and
cross-reference the first by number.

## Before submitting

Ask first. RFC #18202 deprecates the Model Runner V1 EPLB path before v0.31 and
removes it before v0.32, and both bugs are MRv1-only. `expert_map_record_path`
in particular exists *only* on MRv1, so it is worth asking whether the feature
is meant to survive at all before fixing it. An issue or a comment on #18202
costs nothing and may save the review.

## Submitting

```bash
# 0. Move the clone out of the session scratchpad first - it is temporary.
cp -r <scratchpad>/vllm-ascend ~/src/vllm-ascend
cd ~/src/vllm-ascend

# 1. Fork, and add it as a second remote so `origin` stays upstream.
gh repo fork --clone=false
git remote add fork git@github.com:deveringham/vllm-ascend.git

# 2. Push one branch. Nothing upstream changes.
git push -u fork fix-eplb-record-mode

# 3. Open the PR.
gh pr create --repo vllm-project/vllm-ascend \
  --base main --head deveringham:fix-eplb-record-mode \
  --title "[BugFix][EPLB] Do not start expert transfers in recording mode" \
  --body-file ~/src/moe-reliability-ascend/docs/pr-body-record-mode.md
```

Then the same for `fix-eplb-honour-changed` with
`docs/pr-body-honour-changed.md` and its title.

Things that will bite:

- **The title is CI-enforced.** `.github/workflows/pr_test.yaml` requires it to
  contain one of `[BugFix]`, `[Performance]`, `[Test]`, `[CI]`, `[Feature]`,
  `[Doc]`, `[Misc]`, `[Community]`, `[Refactor]`. `[BugFix][EPLB] ...` passes.
- **DCO is enforced.** Both commits are signed off. If you amend or rebase,
  keep `-s`.
- **AI assistance must be disclosed** (vLLM contributing guide, "AI Assisted
  Contributions"). Both commits carry `Co-authored-by: Claude
  <noreply@anthropic.com>` above the sign-off, and both bodies end their first
  section with an "AI assistance" note. Keep both through any amend or rebase.
- A GitHub Action rewrites the PR description to append `- vLLM version:` and
  `- vLLM main:` lines. The body files deliberately do not contain them; if you
  edit the description later, leave them in place.
- To update a PR after review, commit on the same branch and
  `git push fork <branch>`. Never open a second PR for the same change.

## What the bodies do not say

- **B ran twice on hardware.** The body quotes only `eplb-honour-pr`, which
  installed the branch itself. The earlier `eplb-honour-changed` run used our
  node patch, which reaches the same decision through a different code path,
  and is not cited upstream.
- Neither branch has been through the project's own CI, only ruff, mypy and
  `tests/ut/eplb` run locally against a version-matched package copy.
- B's missing-load path is unit-tested only. It is unreachable in production
  today, because `warm_up_eplb` always gathers a load before the first cycle.
- No latency claim is made for either change, and none should be. See
  `docs/upstream-eplb.md` for why the gate averts less than it looks like it
  should.

## C. `[RFC][EPLB] Optional imbalance impact threshold before rearranging experts`

Goes to vLLM core, not vllm-ascend: Model Runner V2 on Ascend already uses
core's EPLB controller, and the legacy path our prototype patches is scheduled
for removal (RFC #18202 deprecates it before v0.31, removes it before v0.32).
Drafted so the text is ready. **It should not go out until the open items at the
end are closed.**

### Motivation

EPLB rearranges experts on a step counter (`step_interval`). Apart from a
ROCm-only check that skips plans improving balance by less than 5% (#47206),
nothing in core asks whether the current imbalance is worth fixing. A plan almost always
improves balance, so EPLB keeps moving experts when the imbalance costs no
latency.

We measured this on DeepSeek-V2-Lite-Chat (64 experts, 4x Ascend 910B3,
vllm-ascend 0.23). At a 1.73x busiest-rank load that costs no latency on its
own, enabling EPLB added **+3.6% TPOT** (13 paired repeats over three runs, 95%
interval +1.0% to +6.1%, 10 of 13 positive). Separately, coarse MoE models limit
what any rearrangement can reach: on Mixtral 8x7B (8 experts, 4 ranks) EPLB took
2.62x only to 1.58x and recovered no latency.

**Corrected 2026-10-19: do not claim the idea is absent from Ascend.**
vllm-ascend's MRv2 path patches core's default EPLB policy to its own `stair`
policy (`vllm_ascend/patch/platform/patch_eplb.py`, `policy_field.default =
"stair"`), and that policy already gates rearrangement per layer on measured
imbalance: `StairConfig.relative_balance_threshold = 0.95` with hysteresis
against the balance at the last commit, and
`absolute_balance_threshold = 0.90`. It returns the current placement for a
layer whose thresholds do not fire, which is a no-op exactly as a kept
placement is. So on Ascend MRv2 the mechanism this RFC proposes is already
shipped and on by default.

What still stands, and what the RFC should actually argue: core's *own* default
policy has no such gate outside the ROCm path, and both gates are in units of
**balance**, not of impact. Stair's absolute threshold fires at a balance of
0.90, i.e. a mean busiest-rank ratio of about 1.11x - far below any imbalance
we have measured a latency cost for (1.42x on Mixtral at 100-token answers;
nothing below 3.5x on DeepSeek). So stair would still rearrange, at cost, where
there is nothing to win. The contribution is not "add a threshold" but "the
threshold belongs in units of measured impact, calibrated per deployment", and
stair is the precedent to cite rather than the gap to fill. Rewrite the
motivation before sending.

### Proposed change

- A new option, e.g. `eplb_config.min_imbalance` (float, off by default). At each
  rearrangement step, if the measured imbalance is below the threshold, keep the
  current placement and log the reading.
- The statistic: busiest rank over mean rank load, computed **per MoE layer and
  then averaged over layers**. Not pooled across layers first, which lets a rank
  that is hot in some layers and cold in others look balanced. Core's existing
  ROCm check uses the pooled form.
- Rank consistency: compute it from the all-reduced load window, so every rank
  reaches the same decision. `eplb_state.py` already notes that a rank-divergent
  `rebalanced` flag hangs an `all_reduce`, so state this up front and ship a unit
  test for it.
- No built-in default. The right threshold depends on the deployment: 1.42x on
  Mixtral with 100-token answers, substantially different at 300 tokens, and
  DeepSeek has no threshold below 3.5x. Documentation should say the value
  describes a traffic mix, not a model.

### What our prototype showed

A 58-line patch on vllm-ascend 0.23.0 MRv1 (`eplb_worker.py`), run as the
`eplb-gated` arm:

- Decided correctly 96 of 96 times at two skew levels that cost nothing (gate
  shut), and opened at 3.63x.
- All four ranks logged the same reading to three decimals on every decision.
- But it saved less than expected. With the gate shut and nothing moving, EPLB
  still cost **+2.1% TPOT** [+0.01, +4.20] against EPLB off, over 15 paired
  rounds in three runs — against plain EPLB's **+3.1%** [+0.75, +5.38] over 19.
  So the threshold averts roughly a third of the cost; the rest is there because
  dynamic EPLB is enabled at all (an enabled-but-wholly-dormant arm reads +1.3%,
  which does not separate from zero over 11 pairs).

### Open before sending

- **Units.** The prototype compares `_compute_imbalance`'s *worst-layer* value
  against a threshold calibrated on the *layer-averaged* busiest-rank load.
  Worst-layer runs about 0.4x higher, so the gate opens too early. Use the
  per-layer average.
- **Re-measure on Model Runner V2.** All our numbers come from the legacy path;
  the RFC targets core's controller.
- ~~**The +3.2%.**~~ **Closed 2026-10-20** (`eplb-standing`, and pooling four
  runs). The ladder at a free 1.73x is off → idle +1.3% [-1.25, +3.81] → gate
  shut +2.1% [+0.01, +4.20] → on +3.1% [+0.75, +5.38], about a point a rung. So
  a threshold averts roughly a third of EPLB's standing cost, a dormant mode
  would avert about two thirds, and the individual rungs are not separable at
  any sane cost (per-pair sd 3.8-4.8%, so ~70 paired rounds to resolve one
  point). **What the RFC may claim:** the option avoids work whose benefit is
  unmeasurable, and saves of order one percent of TPOT at a skew that costs
  nothing. Not a 3% saving — that was the whole of EPLB being enabled, which a
  threshold does not touch.
- **Make the claim modest.** What the data supports: EPLB never asks whether
  imbalance is worth fixing, applying a plan is not free, and this option lets an
  operator who has measured their threshold skip pointless moves. Not: "saves X%
  latency".

### Smaller item for the RFC

**Expose the imbalance reading as a metric.** Every cycle, rank 0 computes
per-layer imbalance for the current and planned placement, logs it and stores it
in `latest_expert_hotness`. The only consumer is an optional `ms_service_metric`
adapter; on our node it fails to import (`No module named 'ms_service_metric'`),
so the number lives only in the log. vLLM core has `log_balancedness`, which also
only logs. Publishing the per-layer reading as a Prometheus gauge costs nothing
to measure, since the number is already computed — and a threshold option is only
usable if operators can see this number before deciding whether to enable EPLB.
