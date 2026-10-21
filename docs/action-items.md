# What is missing, and what would close it

Written 2026-10-16, from a review of the detection and EPLB work against two
goals: a contribution upstream can use, and a publication characterising and
detecting expert load imbalance. Mitigation and merged implementations are
deliberately out of scope here - the three lists below are the detection method,
the characterisation of EPLB's own problems, and the proposals.

Each item says what is missing, why it matters, what closes it, and where it
runs. "Node" means it needs the NPUs; "desk" means it is re-analysis of data
already under `results/`. Status is kept current as items close.

## Detection

**D1. Every positive is our own injector.** *(DeepSeek prompt-routing done,
2026-10-16: the screen fires on an unskewed Dolly stream, one confirmed alarm per
1000 requests - see B2 in detection-findings.md. Left: the same on Mixtral, the
decode capture, and a hot-locale stream.)*
The screen has never fired on an imbalance it did not cause. The only
natural-skew evidence is pooled corpus loads - Yoruba at 1.47x against Mixtral's
1.42x onset - which is one row of one table, and every capture is 1200-1600
requests of short prompts with `max_new_tokens = 1`, so it holds prompt routing
only. A deployment's screen sees decode routing, long prompts and hours of
traffic.
*Closes it:* long unbiased captures with generation enabled, over homogeneous
per-corpus streams as well as a mixed one, on both models. Node.

**D1c. What a biased level does to each token source.** *(done, 2026-10-16)*
Answered: the floor cancels and the signal does not. At level 0 the two sources
are independent (correlation +0.085) and pooling falls below both; at every
biased level all 26 layers agree and the correlation is +0.94 to +1.000, so
pooling averages and detection is 1.00 with decode included against 0.95
prompt-only at the hardest level. Decode makes the screen *more* sensitive. See
D8 for what that exposed.

**D8. We have never injected a skew that is not common-mode.** *(new, 2026-10-16)*
The bias plugin adds the same per-expert offset to every expert-selection call,
prefill and decode alike, so every injected skew in this project is common-mode
by construction - the easiest possible case for a pooled screen. Natural routing
is measurably not (prefill and decode correlate +0.085 at level 0). A real
imbalance arising in prefill alone, which is where its latency cost lives, would
be diluted by decode tokens the way the natural floor is. **Every detection rate
in docs/detection-findings.md is therefore an upper bound for a non-common-mode
skew.**
*Closes it:* bias one token source only. The plugin decides at selection time,
where prefill and decode are not labelled and chunked prefill mixes them in a
step, so this needs a way to tell them apart - the step's batch composition is
the obvious candidate. Code, then node. This is now the largest single
qualification on the detection results.

**D1b. The screen's cost on DeepSeek is unmeasured again.** *(new, 2026-10-16)*
Found while building D2: with `server.count_experts` on, the router bias was
never applied, because plugins load sorted by name and router_bias skipped any
selection function the counter had already wrapped. `gate-closure`'s count-on arm
read 1.13x at a level whose validation says 1.73x, so its +1.65% / +1.71% is
withdrawn. The bug is fixed and has a regression test; the measurement is not
replaced. Mixtral's +0.28% survives (level 0, and the zero-bias plugin costs
Mixtral nothing), and detection rates are untouched.
*Closes it:* re-run `gate-closure`'s screen arms, in the decode-heavy regime D7
and E1 also want. Node.

**D2. Detection delay has never been measured on hardware.** *(done,
2026-10-16)* `imbalance.bias_schedule` moved the bias at step 1200 of a running
server, rank-consistently (24 of 24 transitions at the same step), and the skew
was caught in the first window that closed after it arrived, 4 of 4 points: 8
steps at k=1, 16 at k=2, 24 at k=3. In windows C4's simulation was exactly
right; in requests it was optimistic by 2.3x, because a live window spans ~18
requests and the bootstrap's spans 8. Two things it opened: D9, and the fact
that a step-cadence window is blind without an evidence floor (C7).

**D9. A window should close on evidence, not on steps.** *(new, 2026-10-16)*
Counted assignments per window ran 128 to 14240 within one point of
`detection-delay` - thin while the server fills, then moving with the
prefill/decode mix. A 128-assignment window reads 1.34-2.96x on benign traffic,
so scored as the server writes them the alarm lands at 2.13x and the screen
never fires at all; an evidence floor puts it at 1.23-1.28x and recovers
everything. Every live-screen number depends on this, and it is an
implementation defect rather than a calibration one.
*Closes it:* close the window in `expert_counter` when the counted assignments
reach a target, with the step cadence as a ceiling, and report the realised
window length. Code, then any live run. Cheap, and it should land before D1b
re-measures the screen's cost.
Every onset figure in `detection-findings.md` is a splice of two static pools;
every server run so far carried a fixed bias for its lifetime. Delay is the
headline metric of a detector, and ours is simulated. This is the largest hole
in the detection story.
*Closes it:* the bias plugin reading its offset from a tensor updated in place,
so one server can change skew mid-stream - which also keeps graph mode working,
since the vector is allocated once and never replaced. Code, then node.

**D3. The pipeline has never run as a pipeline.**
Stage 1 is evaluated; stages 2 (step time) and 3 (pace-setter rank) only
offline, on different runs. C5's cost argument brackets stage 2's filter rate
between 0.2% and 6% because it is unmeasured, and open items 3 and 4 remain: no
step tables for the rotating sweeps, and no balanced baseline per profiled run,
so stage 3 coincides rather than attributes.
*Closes it:* one run where the screen fires and triggers the profiled
confirmation end to end, with a balanced arm in the same rounds. Node.

**D4. No competing statistic has ever been scored.**
We have busiest-rank-over-mean, pooled against per-layer, and three
demonstrations that its mapping to cost goes stale. There is no evidence it is
the *right* statistic: CV, entropy, max-expert share, p99 of per-layer max and
EPLB's own balancedness have never been scored against it.
*Closes it:* score each statistic on the existing captures by how well it
separates costly from free levels. Desk. **First candidate, from B3:** the
prefill-only busiest rank, since the cost of skew lives in prefill-heavy steps
and prefill carries a different skew from decode. D1c's capture is now available
for it - though note that on a common-mode injected skew the prefill-only and
pooled statistics barely differ (correlation +1.000 at the top level), so this
candidate cannot be told apart from the pooled one until D8 provides a skew that
is not common-mode.

**D5. The sampling bias is unquantified and uncorrected.**
`detection_params.py` (2026-10-16) showed that thinning tokens biases
busiest-rank-over-mean upward - benign median 1.137 to 1.459 on Mixtral across
splits at equal cost - because each layer's reading is a maximum over ranks of a
noisy share. So the alarm is in sampling-dependent units the impact threshold is
not, and changing `count_layers` or the token fraction invalidates a calibration
at no change in cost.
*Closes it:* characterise the bias against tokens counted, experts per rank and
window length, then correct for it. Desk, and it would make the
calibrate-per-deployment protocol transferable.

**D6. The screen is undefined on a rearranged server.**
It counts on one rank and `rank_load` assumes contiguous placement, so under
EPLB the reading is not valid - exactly the deployment whose balancer we want to
gate. Using EPLB's own numbers is the operational answer and a hole in a
detection contribution.
*Closes it:* read the live expert map and score per-rank load against it. Code,
then desk.

**D7. Rank count and window units.**
The benign floor and the noise both depend on `n_ranks` (a maximum over four),
so nothing calibrated at EP4 transfers to EP16. Windows are counted in requests,
which a serving monitor cannot do, and length spread alone is now known to move
latency by 20%.
*Closes it:* run the screen at 2 and 8 ranks and show how the floor and the
alarm scale; re-form windows on time or step count. Node, then desk.

## EPLB's problems

**E1. Steady state is unmeasured, and it undercuts the central claim.**
At 3.63x, convergence took about 85 s of a 127 s batch, so "EPLB recovers
nothing" is dominated by the transient. A maintainer will say we measured
warm-up.
*Closes it:* a long run comparing post-convergence TPOT with EPLB on against a
statically loaded placement. Node. Most damaging gap in the EPLB story.

**E2. The counterfactual is wrong.**
"EPLB off" means contiguous placement, so the comparison conflates *is dynamic
rearrangement worth it* with *is the default placement bad*.
*Closes it:* load the best offline placement at startup - now possible since we
know what `eplb_record_map` produces - and ask whether EPLB beats it. Node, and
it is also a feature proposal.

**E3. The +3.23% is unattributed.**
With mitigation out of scope, "having EPLB enabled costs about 3% before it
moves anything" is the most interesting EPLB finding we have, and we cannot say
whether it is the heat-collection `add_`, the all-gather, the planner subprocess
or the loader's empty cycles. One level, one model, one contended node.
*Closes it:* ablate the pieces with the existing patch mechanism, quiet node,
more than one level. Node.

**E4. The coarse-MoE bound is one data point that wants to be a law.**
Eight experts over four ranks floors at 1.58x. Generalised - best achievable
imbalance as a function of experts per rank and skew shape - it tells an
operator before they enable anything whether EPLB can help them.
*Closes it:* derive it, then check against the captures. Desk. The most
underexploited result in the project.

## Proposals

**P1. The first proposal should be the metric, not the gate.**
`_compute_imbalance` already computes per-layer busiest-rank-over-mean;
`latest_expert_hotness` is written on rank 0 and consumed only by an optional
adapter, so everyone else scrapes logs, as we do. Exporting it as a Prometheus
metric costs nothing, measures nothing new, is hard to object to, and every
other proposal needs operators to be able to see the number.
*Closes it:* writing it up. Desk.

**P2. The gate needs a threshold story that is not a 30-minute sweep.**
Three demonstrated staleness axes - model, traffic mix, output length - make
"calibrate per deployment" honest and unsellable.
*Closes it:* a self-calibrating formulation that gates on imbalance relative to
the deployment's own observed benign distribution rather than an absolute load.
The FPR and corpus work provides the material; design is desk work. The
cost-aware alternative needs E3 first.
*Caution found 2026-10-16 (B2):* a relative bar gets **more** prone to corpus
shift as the window lengthens, not less - benign p99 tightens while a corpus's
true offset does not, so a self-calibrated bar fires on 53.8% of Dolly windows
at 32 requests where the absolute bar fires on none. Since a change of traffic
has no latency cost attached, the adaptive form has to be calibrated over
traffic that already contains the variation it will meet, which is a much
stronger requirement than "calibrate on your own traffic".

**P3. The target is unknown.**
`eplb_updator.py` and `policy_factory.py` are marked for removal in favour of
vLLM core's EPLB (`vllm` issue #22246 / PR #24069). A proposal against code
being deleted goes nowhere, and the gate probably belongs against core's
`eplb_state.py`, where the word "imbalance" does not appear.
*Closes it:* check the state of those two upstream. Desk, and it gates P1-P5.

**P4. Nothing is written in their terms.**
Our documents are findings documents. A proposal needs metric and knob names,
default-off, the rank-divergence hazard up front, and to be legible without our
calibration apparatus. Worth knowing whether anyone upstream has already asked
for imbalance-gated rearrangement.
*Closes it:* a short self-contained RFC, after P3. Desk.

**P5. A rank-consistency test, not a hardware observation.**
We verified empirically that all four ranks log the same reading. vLLM core
carries an explicit comment that a divergent flag hangs an `all_reduce`, so a
conditional on a gathered tensor will be asked for a test.
*Closes it:* a unit test over the gathered-load arithmetic. Desk.

## Order

D1 first (running), then D2 and D4/D5, which are the missing evaluation and the
missing method. E1 and E2 are one run. P1 and P3 are free and open the channel
upstream.
