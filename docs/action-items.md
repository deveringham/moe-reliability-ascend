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

**D2. Detection delay has never been measured on hardware.** *(plugin done)*
The step-counted `imbalance.bias_schedule` landed 2026-10-16: one server can now
change its skew at a known engine step, rank-consistently, with the bias vector
mutated in place so graph capture keeps working. What is left is the run and its
analysis - the screen records the step per window, so a delay comes out directly.
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
separates costly from free levels. Desk.

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
