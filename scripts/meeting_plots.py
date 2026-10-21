"""Plots for the 2026-10-19 status write-up: EPLB's cost, the screen's detection and overhead.

    uv run python scripts/meeting_plots.py

Each figure carries a title, a legend where there is more than one series, and a
caption under the axes saying where the numbers come from. Figures 1, 2, 5 and 6
are typed in from the published tables (docs/findings.md, docs/upstream-eplb.md,
docs/detection-findings.md); figure 3 reads docs/figures/tables/detection.csv and
figure 4 the detection-delay run's counting files, which must be present under
results/.
"""
import csv, os, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(REPO, "docs", "figures", "meeting-2026-10-19")
os.makedirs(OUT, exist_ok=True)

def save(name, caption, width=70):
    """Put the caption under the axes, leave it room, and write the figure."""
    fig = plt.gcf()
    text = textwrap.fill(caption, width)
    lines = text.count("\n") + 1
    bottom = (0.12 + 0.135 * lines) / fig.get_figheight()
    fig.text(0.02, 0.02, text, fontsize=7.5, va="bottom", ha="left")
    plt.tight_layout(rect=(0, bottom, 1, 1))
    plt.savefig(os.path.join(OUT, name), dpi=120)
    plt.close()

# 1. DeepSeek: EPLB on vs gated, TPOT change against EPLB off (eplb-gated, docs/upstream-eplb.md)
x = [1.09, 1.73, 3.63]
plt.figure(figsize=(5.5, 4.6))
plt.errorbar(x, [1.86, 5.52, 4.63], yerr=[9.39, 3.80, 4.29], marker="o", capsize=3, label="EPLB on")
plt.errorbar([v + 0.05 for v in x], [0.00, 3.23, 1.54], yerr=[4.55, 0.42, 0.63], marker="o", capsize=3,
             label="EPLB gated at 2.43x")
plt.axhline(0, color="k", lw=0.5)
plt.title("DeepSeek: EPLB's cost, with and without the gate")
plt.xlabel("busiest NPU load (x)")
plt.ylabel("TPOT change vs EPLB off (%)")
plt.legend(fontsize=8, loc="lower right")
save("eplb_deepseek.png",
     "Mean of 4 paired repeats, bars ±1 sd; 300-token answers, 4 NPUs. None of these skews costs latency "
     "with answer length held, so anything above zero is overhead. The gate was shut at 1.09x and 1.73x and "
     "opened twice per run at 3.63x.")

# 2. EPLB on minus EPLB off, both models: 300 tokens, 4 paired repeats, +- sd of the pairs.
#    DeepSeek from eplb-gated, Mixtral from eplb-mixtral (scripts/eplb_false_positive.py on each run).
plt.figure(figsize=(5.5, 4.6))
plt.errorbar([1.093, 1.729, 3.634], [1.86, 5.52, 4.63], yerr=[9.39, 3.80, 4.29], marker="o", capsize=3,
             label="DeepSeek-V2-Lite (64 experts)")
plt.errorbar([1.112, 1.758, 2.618], [-0.18, 1.85, -0.99], yerr=[1.22, 1.88, 0.98], marker="o", capsize=3,
             label="Mixtral 8x7B (8 experts)")
plt.axhline(0, color="k", lw=0.5)
plt.title("EPLB Overhead vs. Model and Imbalance")
plt.xlabel("busiest NPU load (x)")
plt.ylabel("TPOT change vs EPLB off (%)")
plt.legend(fontsize=8, loc="lower right")
save("eplb_by_model.png",
     "Mean of 4 paired repeats, bars ±1 sd; 300-token answers, 4 NPUs. EPLB helps neither model. DeepSeek "
     "pays ~5% where Mixtral pays nothing measurable, likely because DeepSeek's decode is host-bound, so "
     "EPLB's per-step work adds directly to step time.")

# 2b. How far EPLB balances each model: its own busiest-NPU reading (per layer, averaged) at the
#     first and last plan, mean of the 4 eplb-on repeats per level, from the same two runs'
#     eplb_activity.hotness.
plt.figure(figsize=(5.5, 4.6))
plt.plot([1.118, 1.682, 3.490], [1.020, 1.030, 1.079], marker="o", label="DeepSeek-V2-Lite (64 experts)")
plt.plot([1.145, 1.751, 2.620], [1.073, 1.248, 1.581], marker="o", label="Mixtral 8x7B (8 experts)")
plt.plot([1, 3.6], [1, 3.6], color="k", lw=0.6, ls="--", label="no change")
plt.axhline(1, color="k", lw=0.6, ls=":", label="perfect balance")
plt.title("Imbalance Before vs. After EPLB")
plt.xlabel("busiest NPU load before EPLB (x)")
plt.ylabel("busiest NPU load after EPLB (x)")
plt.legend(fontsize=8, loc="upper left")
save("eplb_balance.png",
     "EPLB's own busiest-NPU reading at its first and last plan, mean of 4 repeats; 300-token answers, 4 "
     "NPUs, 8-12 plans per run. With 64 experts DeepSeek ends near 1.0x from any start; Mixtral's 8 experts "
     "over 4 NPUs (2 per NPU) leave no placement much better than 1.6x at the strongest skew.")

# 2c. What skew itself costs: TPOT against each run's least-skewed level, paired by round, +- sd.
#     Mixtral 100 tokens: calibration-sweep (3 repeats); Mixtral 300: eplb-mixtral, EPLB-off arm;
#     DeepSeek with every answer held at one length (ignore_eos): length-held-100, held-100 arm, and
#     fixed-length, EPLB-off arm (4 repeats each). Loads are each run's own validation readings.
plt.figure(figsize=(5.8, 4.8))
plt.errorbar([1.112, 1.133, 1.195, 1.273, 1.365, 1.758, 2.618], [0, -1.09, -0.25, 1.95, 0.16, 5.83, 14.47],
             yerr=[0, 1.34, 0.68, 1.91, 2.10, 1.47, 2.30], marker="o", capsize=3, color="C1",
             label="Mixtral, 100-token answers")
plt.errorbar([1.112, 1.758, 2.618], [0, -0.98, 2.95], yerr=[0, 1.73, 1.92], marker="s", capsize=3,
             color="C1", ls="--", label="Mixtral, 300-token answers")
plt.errorbar([1.04, 2.39, 3.53], [0, -2.45, -2.13], yerr=[0, 4.87, 3.91], marker="o", capsize=3,
             color="C0", label="DeepSeek, 100-token answers")
plt.errorbar([1.05, 2.48, 3.64], [0, 2.56, -0.27], yerr=[0, 1.89, 1.84], marker="s", capsize=3,
             color="C0", ls="--", label="DeepSeek, 300-token answers")
plt.axhline(0, color="k", lw=0.5)
plt.ylim(-9, 26)
plt.title("Latency Cost of Imbalance vs. Model and Workload")
plt.xlabel("busiest NPU load (x)")
plt.ylabel("TPOT change vs least-skewed level (%)")
plt.legend(fontsize=7.5, loc="upper left", handlelength=4)
save("skew_cost.png",
     "Injected skew, 4 NPUs, eager mode, responses held to fixed length. ±1 sd over 3-4 repeats. "
     "Mixtral starts paying at ~1.4x, while DeepSeek pays nothing measurable up to "
     "3.6x. Where imbalance starts to hurt depends on the model and the traffic, so the threshold has to "
     "be calibrated per deployment.",
     width=78)

# 3. Detection rate vs load, 8-request windows, 4 random layers, 10% of tokens
rows = list(csv.DictReader(open(f"{REPO}/docs/figures/tables/detection.csv")))
plt.figure(figsize=(5.5, 4.4))
for model, name in (("deepseek-v2", "DeepSeek"), ("mixtral", "Mixtral")):
    sel = [r for r in rows if r["group"] == "screen" and r["model"] == model and r["window_requests"] == "8"
           and r["layers"] == "4" and r["layer_choice"] == "random" and r["token_fraction"] == "0.1"]
    sel.sort(key=lambda r: float(r["busiest_rank"]))
    plt.plot([float(r["busiest_rank"]) for r in sel], [float(r["detection_rate"]) for r in sel], marker="o",
             label=name)
plt.axvline(1.42, color="k", lw=0.8, ls=":", label="Mixtral: skew starts costing (1.42x)")
plt.title("Detection rate vs. Imbalance Level")
plt.xlabel("busiest NPU load (x)")
plt.ylabel("fraction of windows flagged")
plt.legend(fontsize=8, loc="lower right")
save("detection_rate.png",
     "8-request windows, 4 random MoE layers, a tenth of their tokens. Alarm set to flag 1% of normal "
     "windows.")

# 4. Live screen reading over steps, bias switched on at step 1200 (detection-delay, Mixtral)
sys.path.insert(0, f"{REPO}/scripts")
from delay_eval import window_loads, usable
import numpy as np
run = f"{REPO}/results/20261016-143648_imbalance_mixtral_npu4_bs512_detection-delay"
# One threshold for every run, as a deployment would have: the 99.5th percentile of the
# pre-switch windows pooled over all six points, using the same first-60% calibration slice
# delay_eval.py sets its per-point alarms on (1.254x).
fit = []
for label in ("imbalance_0", "imbalance_0_r2", "imbalance_0.5", "imbalance_0.5_r2", "imbalance_1", "imbalance_1_r2"):
    loads, _ = usable(window_loads(f"{run}/{label}_expert_counts.jsonl"))
    pre = [l for s, l, _ in loads if s < 1200]
    fit += pre[:int(len(pre) * 0.6)]
threshold = float(np.quantile(fit, 0.995))
plt.figure(figsize=(6.5, 4.4))
for label, name in (("imbalance_0", "no skew"), ("imbalance_0.5", "moderate skew (1.35x)"),
                    ("imbalance_1", "strong skew (1.75x)")):
    loads, _ = usable(window_loads(f"{run}/{label}_expert_counts.jsonl"))
    plt.plot([s for s, _, _ in loads], [l for _, l, _ in loads], lw=0.8, label=name)
plt.axhline(threshold, color="k", lw=1, ls="--", label="threshold, set based on calibration")
plt.title("Live Screen with Injected Skew")
plt.xlabel("engine step")
plt.ylabel("screen reading (x)")
plt.ylim(1.05, 2.15)
plt.legend(fontsize=7.5, loc="upper left", ncol=2)
save("detection_delay.png",
     "One line per run, read live from the server on hardware. The screen fired in the first window after "
     "the switch in every skewed run.", width=85)

# 5. Overhead on Mixtral: sampled screen vs full capture (screen-counting)
plt.figure(figsize=(4.5, 4.4))
plt.bar([0, 1], [0.28, 6.38], yerr=[0.71, 0.61], capsize=3)
plt.xticks([0, 1], ["screen (~1% sampled)", "full capture"])
plt.title("What the screen costs to run")
plt.ylabel("end-to-end latency change (%)")
save("overhead.png",
     "Mixtral, no skew, 5 paired repeats, bars ±1 sd. Run-to-run noise is 0.44%.", width=55)

# 6. Standing cost while nothing is wrong, all four ways of watching for skew on one model, one
#    measure and one set of shuffled rounds (screen-vs-eplb, 2026-10-19). TPOT against the EPLB-off
#    arm of the same round, pooled over both free levels (1.05x and 1.72x), 9 paired rounds each.
plt.figure(figsize=(5.6, 4.6))
plt.bar([0, 1, 2, 3], [-0.47, 1.31, 2.76, 0.40], yerr=[4.89, 3.56, 3.74, 3.25], capsize=3,
        color=["C1", "C0", "C0", "C0"])
plt.xticks([0, 1, 2, 3], ["screen only", "EPLB idle", "EPLB gate shut", "EPLB on"])
plt.axhline(0, color="k", lw=0.5)
plt.title("Standing cost when skew costs nothing")
plt.ylabel("TPOT change vs EPLB off (%)")
save("standing_cost.png",
     "DeepSeek, 9 paired rounds per arm, bars ±1 sd. (A tenth round was re-run hours later after an "
     "unrelated server failure, so its arms no longer share conditions; including it moves every bar by "
     "under 0.7 points.) Repeating one arm varies by 2-3.5%, so this run separates none of them: it "
     "bounds the screen's standing cost near zero and leaves the EPLB arms unresolved. Pooled over "
     "all four runs (eplb-standing, 2026-10-20) plain EPLB costs +3.1% [+0.8, +5.4], the shut gate "
     "+2.1% [+0.0, +4.2] and an enabled-but-dormant EPLB +1.3% [-1.3, +3.8]: about a point a rung.",
     width=78)
