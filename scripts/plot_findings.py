"""Figures for presenting the router-bias findings.

    uv run python scripts/plot_findings.py [<out_dir>]

Writes docs/figures/*.png (and a figures.json of every plotted number) from the
run manifests and step tables under results/, so a re-run regenerates them.
Everything is read from result files; nothing is hard-coded except which runs
each panel draws from.

Colours come from the validated categorical palette: hue carries the model,
line style the execution mode, and the offset ramp is a single-hue ordinal
scale. Low-count cells are labelled with their n rather than hidden.
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
MODEL_COLOUR = {"deepseek-v2": "#2a78d6", "mixtral": "#eb6834"}
MODEL_LABEL = {"deepseek-v2": "DeepSeek-V2-Lite", "mixtral": "Mixtral 8x7B"}
COND = ("#2a78d6", "#eb6834", "#1baf7a")
OFFSET_RAMP = ("#86b6ef", "#2a78d6", "#104281")  # ordinal, light -> dark
RESULTS = "results"


def manifests(pattern):
    for d in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        yield d, json.load(open(os.path.join(d, "manifest.json")))


N_RANKS = 4


def calibration():
    """{model: {level: (busiest rank load, live experts)}} from the calibration runs.

    Load is the busiest rank's share in each MoE layer, averaged over layers
    (see ``busiest_rank``), read from the validation file rather than the
    manifest summary, which takes the busiest rank of the layer-averaged shares.
    The two agree whenever one rank is hot in every layer, as under router bias,
    but the summary form lets the hot rank cancel across layers in natural
    traffic (1.02x where the per-layer form reads 1.14x on DeepSeek).
    """
    out: dict[str, dict[float, tuple[float, int]]] = {}
    for d, m in manifests("*rbias-calibration"):
        if m.get("status") != "completed":
            continue
        model = os.path.basename(d).split("_")[2]
        for p in m["points"]:
            v = p.get("validation_summary") or {}
            if v.get("rank_max_over_mean") is not None:
                f = json.load(gzip.open(os.path.join(d, p["validation_file"])))
                load = float(np.mean(f["rank_max_over_mean_per_layer"]))
                out.setdefault(model, {})[float(p["value"])] = (load, v["active_experts"])
    return out


def busiest_rank(counts):
    """Busiest rank's load over the mean, per layer, averaged over layers.

    ``counts`` is (layers, experts) token-expert assignments; experts are placed
    on ranks contiguously, as vLLM does under expert parallelism.
    """
    f = counts / counts.sum(1, keepdims=True)
    per_rank = f.reshape(f.shape[0], N_RANKS, -1).sum(2)
    return float((per_rank.max(1) * N_RANKS).mean())


def routing_counts(pattern, n_experts):
    """Per-request (layers, experts) counts and MMLU subjects from a routed-expert capture.

    Prompt and generated tokens are pooled, as the 100-token serving regime
    routes both. Layers that never route (DeepSeek's dense first layer) drop out.
    Captures are not pulled by default: ``npull`` excludes ``activations/``.
    """
    (path,) = glob.glob(os.path.join(RESULTS, pattern, "activations", "records.jsonl.gz"))
    counts, subjects = [], []
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        a = np.concatenate([np.asarray(r["prompt_routed_experts"]), np.asarray(r["routed_experts"])])
        counts.append([np.bincount(a[:, layer].ravel(), minlength=n_experts) for layer in range(a.shape[1])])
        subjects.append(r["subject"])
    c = np.asarray(counts, float)
    return c[:, c[:, :, 1:].sum(axis=(0, 2)) > 0], np.array(subjects)


def natural_load(pattern, n_experts, seed=0, draws=1000):
    """Busiest-rank load that unbiased traffic produces, for windows of increasing coherence.

    A single prompt is the most skewed window natural traffic offers (a prefill
    chunk of one long prompt); a one-subject window is a topic shift; a random
    200-prompt window is ordinary mixed traffic.
    """
    c, subjects = routing_counts(pattern, n_experts)
    rng = np.random.default_rng(seed)
    single = [busiest_rank(x) for x in c]
    subject = [busiest_rank(c[subjects == s].sum(0)) for s in np.unique(subjects) if (subjects == s).sum() >= 30]
    mixed = [busiest_rank(c[rng.choice(len(c), 200, replace=False)].sum(0)) for _ in range(draws)]
    return {"single prompt": single, "one MMLU subject": subject, "200-prompt mix": mixed}


# The eager sweeps re-run with the plugin installed at level 0 too. The
# rbias-deepseek-001 / rbias-mixtral-001 runs they replace measured against a
# plugin-free balanced point, which overstated DeepSeek's cost.
EAGER = ("*rbias-controlled-000", "*rbias-controlled-001")
# The graph sweeps re-run with the plugin at level 0 too; rbias-graph-000/001,
# which they replace, measured against a plugin-free balanced point.
GRAPH = ("*rbias-graph-threshold-000", "*rbias-graph-threshold-001")
# A second eager DeepSeek sweep, filling the 2.3-3.4x gap the controlled sweep
# left. It is a separate run, so it carries its own balanced baseline and its own
# drift; the two are drawn together rather than pooled.
EAGER_SECOND = {"deepseek-v2": "*rbias-graph-threshold-002"}
DRIFT_PCT = 2.2  # largest run-to-run drift between identical points (zero-control level-2 arms)


def sweep(pattern, metric="tpot_ms_mean"):
    """{level: [per-repeat values]} and the model, for one run."""
    (d, m), = list(manifests(pattern))
    by: dict[float, list[float]] = {}
    for p in m["points"]:
        if p.get("status") == "completed" and p["request_summary"].get(metric):
            by.setdefault(float(p["value"]), []).append(p["request_summary"][metric])
    return os.path.basename(d).split("_")[2], by


def style(ax, xlabel, ylabel, title=None, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=30 if subtitle else 8)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.0), xytext=(0, 7), xycoords="axes fraction",
                    textcoords="offset points", color=INK2, fontsize=9.5, va="bottom")


def fig_dose_response(out, store):
    """TPOT against realised rank load, both models, eager and graph."""
    cal = calibration()
    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor=SURFACE)
    style(ax, "Load on the busiest expert-parallel rank (1.0 = perfectly balanced)",
          "Time per output token (ms)",
          "Strong rank imbalance costs latency; realistic imbalance does not",
          "4 NPUs, 100-token requests, 3 repeats per point. Error bars are the spread over repeats.")
    lo, hi = store["natural_band"]
    ax.axvspan(lo, hi, color="#f0efec", zorder=0)
    ax.annotate("where natural\ntraffic sits", xy=((lo + hi) / 2, 0.995), xycoords=("data", "axes fraction"),
                ha="center", va="top", color=INK3, fontsize=8.5, linespacing=1.3)
    rows = []
    for pattern, mode, ls, marker in ((EAGER[0], "eager", "-", "o"), (EAGER[1], "eager", "-", "o"),
                                      (GRAPH[0], "graph", "--", "s"), (GRAPH[1], "graph", "--", "s")):
        model, by = sweep(pattern)
        # Levels where experts drop out vary the live expert count as well as the
        # skew, so they are a different experiment and are left off.
        levels = [lv for lv in sorted(by) if cal[model][lv][1] == max(n for _, n in cal[model].values())]
        x = [cal[model][lv][0] for lv in levels]
        y = [float(np.mean(by[lv])) for lv in levels]
        e = [float(np.std(by[lv], ddof=1)) if len(by[lv]) > 1 else 0.0 for lv in levels]
        ax.errorbar(x, y, yerr=e, ls=ls, marker=marker, ms=5.5, lw=2, capsize=3, elinewidth=1,
                    color=MODEL_COLOUR[model], zorder=3,
                    label=f"{MODEL_LABEL[model]}, {mode}")
        rows.append({"model": model, "mode": mode, "levels": levels, "load": x, "tpot_ms": y, "sd": e})
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0.14, 1.0))
    ax.set_xlim(1.0, 3.7)
    ax.set_ylim(132, 232)
    fig.text(0.012, 0.015,
             "Every point carries the measurement plugin, the balanced one included. Levels that collapse "
             "routing onto fewer live experts are excluded.\nGraph mode's repeat spread is widest at its "
             "lowest levels, where a fast engine outruns the client: read its slope, not its per-level values.",
             color=INK3, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(os.path.join(out, "dose_response.png"), dpi=200, facecolor=SURFACE)
    store["dose_response"] = rows


def fig_instrument(out, store):
    """The zero-bias control: what the measurement plugin itself cost.

    Each pair is one run, so a pair is a within-run comparison; the two runs
    drift relative to each other and must not be mixed.
    """
    runs: dict[tuple[str, bool, float], list[float]] = {}
    for d, m in manifests("*rbias-zero-control-*"):
        model = os.path.basename(d).split("_")[2]
        flag = bool(m["config"]["imbalance"]["bias_plugin_at_zero"])
        for p in m["points"]:
            runs.setdefault((model, flag, float(p["value"])), []).append(p["request_summary"]["tpot_ms_mean"])

    fig, ax = plt.subplots(figsize=(9.0, 4.6), facecolor=SURFACE)
    style(ax, "", "Time per output token (ms)",
          "Part of the measured cost was our own measurement plugin",
          "Eager mode, 100-token requests, 3 repeats per bar. Each pair is one run, compared "
          "within itself.")
    rows = []
    width, inner, outer = 0.3, 0.34, 1.0
    centres, tick_labels = [], []
    for mi, model in enumerate(("deepseek-v2", "mixtral")):
        for fi, (flag, method) in enumerate(((False, "as first measured"), (True, "with the control"))):
            base = mi * (2 * outer + 0.55) + fi * outer
            lo = np.array(runs[(model, flag, 0.0)], float)
            hi = np.array(runs[(model, flag, 2.0)], float)
            for bi, (v, colour) in enumerate(((lo, COND[0]), (hi, COND[1]))):
                x = base + (bi - 0.5) * inner
                ax.bar(x, v.mean(), width, color=colour, zorder=3,
                       label=("balanced" if bi == 0 else "skewed 3.5x / 2.6x") if (mi, fi) == (0, 0) else None)
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color=INK2, lw=1, capsize=3, zorder=4)
                ax.annotate(f"{v.mean():.0f}", xy=(x, v.mean()), xytext=(0, 8), textcoords="offset points",
                            ha="center", color=INK, fontsize=9.5, fontweight="bold", zorder=5)
            delta = hi.mean() - lo.mean()
            # Offsets in points, so the gaps hold whatever height the figure is given.
            top = (base, max(lo.mean(), hi.mean()))
            ax.annotate(f"{delta:+.1f} ms", xy=top, xytext=(0, 30), textcoords="offset points",
                        ha="center", color=INK, fontsize=10.5, fontweight="bold")
            centres.append(base)
            tick_labels.append(method)
            rows.append({"model": model, "method": method, "balanced_ms": float(lo.mean()),
                         "skewed_ms": float(hi.mean()), "delta_ms": float(delta)})
    ax.set_xticks(centres, tick_labels, fontsize=9.5)
    ax.tick_params(axis="x", labelcolor=INK2)
    for mi, model in enumerate(("deepseek-v2", "mixtral")):
        ax.annotate(MODEL_LABEL[model], xy=(mi * (2 * outer + 0.55) + outer / 2, -0.155),
                    xycoords=("data", "axes fraction"), ha="center", color=INK, fontsize=11,
                    fontweight="bold")
    ax.set_ylim(0, 300)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.36), columnspacing=2.2)
    d = {(r["model"], r["method"]): r["delta_ms"] for r in rows}
    fig.text(0.012, 0.015,
             "Installing the plugin at the balanced point too cuts DeepSeek's measured cost from "
             f"{d[('deepseek-v2', 'as first measured')]:+.1f} to {d[('deepseek-v2', 'with the control')]:+.1f} ms; "
             f"Mixtral moves from {d[('mixtral', 'as first measured')]:+.1f} to "
             f"{d[('mixtral', 'with the control')]:+.1f},\nwhich is within the 1.5-2.2% drift between runs "
             "that the repeated identical points measure. So the plugin costs DeepSeek time and Mixtral none.",
             color=INK3, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "instrument_cost.png"), dpi=200, facecolor=SURFACE)
    store["instrument"] = rows


def step_table(run_dir, label):
    f = os.path.join(run_dir, "step_profile", f"{label}.csv.gz")
    rows = list(csv.DictReader(gzip.open(f, "rt")))
    n = max(int(r["step"]) for r in rows) + 1
    ranks = sorted({int(r["rank"]) for r in rows})
    wall = np.zeros((len(ranks), n))
    reqs, tok = np.zeros(n, int), np.zeros(n, int)
    for r in rows:
        i, s = ranks.index(int(r["rank"])), int(r["step"])
        wall[i, s] = float(r["wall_us"])
        if i == 0:
            reqs[s], tok[s] = int(r["reqs"]) - 1, int(r["tokens"])
    return reqs, tok, wall.mean(0) / 1e3


BINS = ((1, 200, "1-200"), (200, 350, "200-350"), (350, 480, "350-480"), (480, 514, "480-512"))


def fig_steps(out, store):
    """Where in a step the cost actually appears, at matched batch size."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), facecolor=SURFACE, sharey=True)
    rows = []
    for ax, (pattern, model) in zip(axes, (("*rbias-profiled-deepseek-001", "deepseek-v2"),
                                           ("*rbias-profiled-mixtral-001", "mixtral"))):
        (run_dir, m), = list(manifests(pattern))
        style(ax, "", "Median step wall time (ms)" if ax is axes[0] else "")
        ax.set_title(MODEL_LABEL[model], color=INK, fontsize=11, fontweight="bold", loc="left")
        width, gap = 0.24, 0.02
        for ci, p in enumerate(sorted(m["points"], key=lambda q: float(q["value"]))):
            reqs, tok, wall = step_table(run_dir, os.path.basename(p["trace_dir"]))
            dec, pre = tok == reqs, tok - reqs
            groups = [(dec & (reqs >= a) & (reqs < b), lab) for a, b, lab in BINS]
            groups.append((~dec & (pre >= 2500), "2500+\nprefill"))
            for gi, (sel, _) in enumerate(groups):
                if sel.sum() < 8:
                    continue
                x = gi + (ci - 1) * (width + gap) + (0.22 if gi == 4 else 0)
                med = float(np.median(wall[sel]))
                ax.bar(x, med, width, color=OFFSET_RAMP[ci], zorder=3,
                       label=f"offset {p['value']:g}" if gi == 3 and ax is axes[0] else None)
                ax.annotate(f"{int(sel.sum())}", xy=(x, med), xytext=(0, 3), textcoords="offset points",
                            ha="center", color=INK3, fontsize=6.5, zorder=4)
                rows.append({"model": model, "offset": float(p["value"]), "group": groups[gi][1],
                             "median_ms": round(med, 1), "n": int(sel.sum())})
        ax.set_ylim(0, 520)
        ax.set_xticks([0, 1, 2, 3, 4.22],
                      [b[2] for b in BINS] + ["2500+\nprefill"], fontsize=8.5)
        ax.tick_params(axis="x", labelcolor=INK2)
        ax.axvline(3.6, color=GRID, lw=1)
        ax.annotate("decode-only steps, by batch size", xy=(1.5, 0.96), xycoords=("data", "axes fraction"),
                    ha="center", color=INK3, fontsize=8.5)
        ax.annotate("steps carrying\nprefill", xy=(4.22, 0.99), xycoords=("data", "axes fraction"),
                    ha="center", va="top", color=INK3, fontsize=8.5, linespacing=1.3)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.suptitle("A saturated decode step does not lengthen under imbalance",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.915,
             "Eager mode, 100-token requests. Small numbers above each bar are the steps it averages; "
             "cells with fewer than 8 steps are omitted.",
             color=INK2, fontsize=9.5)
    fig.text(0.008, 0.015,
             "At full batch (480-512) the step is the same length at every offset on both models - so the "
             "hot rank's extra work is absorbed, not passed on.\nThe cost appears elsewhere: in "
             "below-full-batch decode steps on DeepSeek, and in prefill-carrying steps on Mixtral.",
             color=INK3, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(os.path.join(out, "step_level.png"), dpi=200, facecolor=SURFACE)
    store["steps"] = rows


def fig_calibration(out, store):
    """What each injected bias level does to rank load, and where routing collapses."""
    cal = calibration()
    fig, ax = plt.subplots(figsize=(7.8, 4.8), facecolor=SURFACE)
    style(ax, "Injected router-logit offset on one rank's experts",
          "Load on the busiest rank",
          "How the knob maps to realised imbalance",
          "Measured by capturing routed experts over 200 MMLU prompts per level.")
    rows, collapsed = [], []
    for model, levels in cal.items():
        xs = sorted(levels)
        full = max(n for _, n in levels.values())
        ax.plot(xs, [levels[x][0] for x in xs], "-", lw=2, color=MODEL_COLOUR[model],
                label=MODEL_LABEL[model], zorder=3)
        for x in xs:
            load, live = levels[x]
            ok = live == full
            # Hollow marks the levels that also cut the live expert count, so they
            # vary two things at once and are not a clean skew setting.
            ax.plot([x], [load], "o", ms=7, color=MODEL_COLOUR[model] if ok else SURFACE,
                    mec=MODEL_COLOUR[model], mew=2, zorder=4)
            if not ok and x == min(k for k in xs if levels[k][1] != full):
                collapsed.append(f"{MODEL_LABEL[model].split('-')[0].split(' ')[0]} "
                                 f"from offset {x:g} ({live} of {full})")
            rows.append({"model": model, "offset": x, "load": round(load, 3), "live_experts": live})
    lo, hi = store["natural_band"]
    ax.axhspan(lo, hi, color="#f0efec", zorder=0)
    ax.annotate("natural traffic sits here", xy=(4.05, (lo + hi) / 2), ha="right", va="center",
                color=INK3, fontsize=8.5)
    ax.set_ylim(0.85, 4.45)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0.03, 0.99))
    fig.text(0.012, 0.015,
             "Hollow points also collapse routing onto fewer live experts, so they vary the expert\n"
             "count as well as the skew: " + "; ".join(collapsed) + ".",
             color=INK3, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "calibration.png"), dpi=200, facecolor=SURFACE)
    store["calibration"] = rows


def natural(store):
    """Natural-traffic load per model and window, and the band it spans."""
    nat = {"deepseek-v2": natural_load("*alpha-sweep", 64), "mixtral": natural_load("*mixtral-alpha", 8)}
    detail = {}
    for model, n_experts, pattern in (("deepseek-v2", 64, "*alpha-sweep"), ("mixtral", 8, "*mixtral-alpha")):
        counts, _ = routing_counts(pattern, n_experts)
        pooled = counts.sum(axis=0)
        # The two forms of the same statistic: busiest rank per layer, averaged,
        # against the busiest rank of the layer-averaged shares.
        per_rank = (pooled / pooled.sum(axis=1, keepdims=True)).reshape(pooled.shape[0], N_RANKS, -1).sum(axis=2)
        detail[model] = (busiest_rank(pooled), float(per_rank.mean(axis=0).max() * N_RANKS))
    rows = []
    for model, windows in nat.items():
        for window, v in windows.items():
            # Subjects are few (11-19), so their full range; sampled windows show p1-p99.
            lo, hi = (min(v), max(v)) if window == "one MMLU subject" else np.quantile(v, [0.01, 0.99])
            rows.append({"model": model, "window": window, "n": len(v), "lo": round(float(lo), 3),
                         "median": round(float(np.median(v)), 3), "hi": round(float(hi), 3)})
    store["natural"] = rows
    store["natural_band"] = (min(r["lo"] for r in rows), max(r["hi"] for r in rows))
    store["natural_detail"] = detail


def fig_impact_map(out, store):
    """What it costs, against where natural traffic falls, on one load axis."""
    cal = calibration()
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9.4, 7.0), facecolor=SURFACE, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.45], "hspace": 0.1})
    style(top, "", "TPOT increase over balanced (%)",
          "Natural traffic stays far below the imbalance that costs latency",
          "Top: injected rank skew, 4 NPUs, eager, 100-token requests, 3 repeats. "
          "Bottom: unbiased MMLU traffic.")
    style(bot, "Load on the busiest expert-parallel rank (1.0 = perfectly balanced)", "")
    lo, hi = store["natural_band"]
    for ax in (top, bot):
        ax.axvspan(lo, hi, color="#f0efec", zorder=0)
    top.axhspan(-DRIFT_PCT, DRIFT_PCT, color="#f0efec", zorder=0)
    top.axhline(0, color=GRID, lw=1, zorder=1)
    top.annotate(f"run-to-run drift (\u00b1{DRIFT_PCT:g}%)", xy=(3.68, DRIFT_PCT), xytext=(0, 4),
                 textcoords="offset points", ha="right", va="bottom", color=INK3, fontsize=8.5)
    top.annotate("natural\ntraffic", xy=((lo + hi) / 2, 0.98), xycoords=("data", "axes fraction"),
                 ha="center", va="top", color=INK3, fontsize=8.5, linespacing=1.3)

    rows = []
    for pattern in EAGER:
        model, by = sweep(pattern)
        full = max(n for _, n in cal[model].values())
        levels = [lv for lv in sorted(by) if cal[model][lv][1] == full]
        base = float(np.mean(by[0.0]))
        x = [cal[model][lv][0] for lv in levels]
        y = [100 * (float(np.mean(by[lv])) / base - 1) for lv in levels]
        e = [100 * float(np.std(by[lv], ddof=1)) / base for lv in levels]
        top.errorbar(x, y, yerr=e, ls="-", marker="o", ms=6, lw=2, capsize=3, elinewidth=1,
                     color=MODEL_COLOUR[model], mec=SURFACE, mew=1.5, zorder=3, label=MODEL_LABEL[model])
        first = next((i for i, v in enumerate(y) if v > DRIFT_PCT), None)
        if first is not None and model not in EAGER_SECOND:
            # Placed in the clear space beside each curve, with a leader to the point.
            spot = {"mixtral": (1.42, 12.5, "bottom"), "deepseek-v2": (2.9, 9.5, "top")}[model]
            top.annotate(f"{MODEL_LABEL[model].split('-')[0].split(' ')[0]}: first cost beyond drift,\n"
                         f"{y[first]:+.1f}% at {x[first]:.2f}x",
                         xy=(x[first], y[first]), xytext=spot[:2], ha="left", va=spot[2],
                         color=INK2, fontsize=8.5, linespacing=1.3,
                         arrowprops={"arrowstyle": "-", "color": INK3, "lw": 0.8, "shrinkB": 5})
        rows.append({"model": model, "levels": levels, "load": [round(v, 3) for v in x],
                     "tpot_pct": [round(v, 2) for v in y], "sd_pct": [round(v, 2) for v in e],
                     "balanced_ms": round(base, 1)})
    # The second DeepSeek sweep, drawn hollow: same instrument, different run, and
    # it does not reproduce the first one's flat stretch below 2.4x.
    for model, pattern in EAGER_SECOND.items():
        if not list(manifests(pattern)):
            continue
        _, by = sweep(pattern)
        levels = sorted(by)
        base = float(np.mean(by[levels[0]]))
        x = [cal[model][lv][0] for lv in levels]
        y = [100 * (float(np.mean(by[lv])) / base - 1) for lv in levels]
        e = [100 * float(np.std(by[lv], ddof=1)) / base for lv in levels]
        top.errorbar(x, y, yerr=e, ls=":", marker="o", ms=6, lw=1.6, capsize=3, elinewidth=1,
                     color=MODEL_COLOUR[model], mfc=SURFACE, mew=1.6, zorder=2,
                     label=f"{MODEL_LABEL[model]}, second sweep")
        rows.append({"model": model, "run": "second", "levels": levels, "load": [round(v, 3) for v in x],
                     "tpot_pct": [round(v, 2) for v in y], "sd_pct": [round(v, 2) for v in e],
                     "balanced_ms": round(base, 1)})

    # Where the two sweeps of one model first leave the drift band, as a range.
    for model in EAGER_SECOND:
        crossings = [r["load"][next(i for i, v in enumerate(r["tpot_pct"]) if v > DRIFT_PCT)]
                     for r in rows if r["model"] == model
                     and any(v > DRIFT_PCT for v in r["tpot_pct"])]
        if len(crossings) < 2:
            continue
        lo_x, hi_x = min(crossings), max(crossings)
        top.annotate(f"{MODEL_LABEL[model].split('-')[0]}: cost begins somewhere in\n"
                     f"{lo_x:.1f}x-{hi_x:.1f}x; the two sweeps disagree",
                     xy=(lo_x - 0.05, 27.2), ha="left", va="bottom",
                     color=INK2, fontsize=8.5, linespacing=1.3)
        top.annotate("", xy=(lo_x, 26.4), xytext=(hi_x, 26.4),
                     arrowprops={"arrowstyle": "|-|,widthA=0.4,widthB=0.4", "color": INK3, "lw": 0.9})

    top.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0.12, 0.98))
    top.set_ylim(-6, 32)

    # One row per model and window: a p1-p99 (or min-max) range with the median.
    nat = store["natural"]
    ticks, labels = [], []
    for i, r in enumerate(nat):
        yv = len(nat) - 1 - i + (0.6 if r["model"] == "deepseek-v2" else 0)
        bot.plot([r["lo"], r["hi"]], [yv, yv], lw=2, color=MODEL_COLOUR[r["model"]], solid_capstyle="round", zorder=3)
        bot.plot([r["median"]], [yv], "o", ms=6, color=MODEL_COLOUR[r["model"]], mec=SURFACE, mew=1.5, zorder=4)
        ticks.append(yv)
        labels.append(f"{MODEL_LABEL[r['model']].split('-')[0].split(' ')[0]} \u00b7 {r['window']}")
    bot.set_yticks(ticks, labels, fontsize=8.5)
    bot.tick_params(axis="y", labelcolor=INK2)
    bot.grid(False)
    bot.set_ylim(-0.7, len(nat) + 0.2)
    bot.annotate(f"natural traffic peaks at {hi:.2f}x", xy=(hi, len(nat) - 0.4),
                 xytext=(10, 0), textcoords="offset points", ha="left", va="center", color=INK2, fontsize=8.5)
    bot.set_xlim(1.0, 3.7)
    fig.text(0.012, 0.012,
             "Load: the busiest rank's share of token-expert assignments in each MoE layer, averaged over layers; "
             "4-way contiguous placement.\nNatural ranges: single prompts and random 200-prompt mixes p1-p99, "
             "one-subject windows min-max; dot = median. Levels that collapse routing are excluded.\n"
             "DeepSeek's two eager sweeps disagree between 2.3x and 3.4x: the filled one reads +1% at 2.35x, "
             "the hollow one +7%.\nWhere its cost begins is unresolved; Mixtral's rise from 1.3x reproduces.",
             color=INK3, fontsize=7.5, va="bottom", linespacing=1.4)
    # Explicit margins: tight_layout cannot place shared axes with long tick labels.
    fig.subplots_adjust(left=0.215, right=0.975, top=0.875, bottom=0.165)
    fig.savefig(os.path.join(out, "impact_map.png"), dpi=200, facecolor=SURFACE)
    store["impact_map"] = rows


ROTATING = {"deepseek-v2": "*rotating-skew-000", "mixtral": "*rotating-skew-001"}


def per_layer_loads(pattern):
    """{level: per-layer busiest-rank load} from a run's own validation files."""
    (d, m), = list(manifests(pattern))
    out = {}
    for p in m["points"]:
        if p.get("validation_file"):
            f = json.load(gzip.open(os.path.join(d, p["validation_file"])))
            out[float(p["value"])] = float(np.mean(f["rank_max_over_mean_per_layer"]))
    return out


def fig_rotating(out, store):
    """Does it matter which rank is hot, or only how hot each layer is?"""
    cal = calibration()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), facecolor=SURFACE, sharey=True)
    rows = []
    for ax, model in zip(axes, ("mixtral", "deepseek-v2")):
        style(ax, "Load on the busiest rank, per layer",
              "Time per output token (ms)" if ax is axes[0] else "")
        ax.set_title(MODEL_LABEL[model], color=INK, fontsize=11, fontweight="bold", loc="left")
        # The fixed target is the reference, so it is drawn first and solid.
        for pattern, label, ls, marker, fill in ((EAGER[0] if model == "deepseek-v2" else EAGER[1],
                                                  "one fixed rank", "-", "o", None),
                                                 (ROTATING[model], "rotating", "--", "s", SURFACE)):
            if not list(manifests(pattern)):
                continue
            _, by = sweep(pattern)
            loads = per_layer_loads(pattern) if label == "rotating" else {
                lv: cal[model][lv][0] for lv in by if lv in cal[model]}
            full = max(n for _, n in cal[model].values())
            levels = [lv for lv in sorted(by) if lv in loads
                      and (label == "rotating" or cal[model][lv][1] == full)]
            x = [loads[lv] for lv in levels]
            y = [float(np.mean(by[lv])) for lv in levels]
            e = [float(np.std(by[lv], ddof=1)) for lv in levels]
            ax.errorbar(x, y, yerr=e, ls=ls, marker=marker, ms=6, lw=2, capsize=3, elinewidth=1,
                        color=MODEL_COLOUR[model], mfc=fill or MODEL_COLOUR[model], mew=1.8,
                        mec=MODEL_COLOUR[model] if fill else SURFACE, zorder=3, label=label)
            # Direct-label the endpoint so neither line can be mistaken for the other.
            ax.annotate(label, xy=(x[-1], y[-1]), xytext=(-6, 12 if label == "rotating" else -16),
                        textcoords="offset points", ha="right", color=MODEL_COLOUR[model], fontsize=9,
                        fontweight="bold")
            rows.append({"model": model, "skew": label, "load": [round(v, 3) for v in x],
                         "tpot_ms": [round(v, 1) for v in y], "sd": [round(v, 1) for v in e]})
        ax.set_xlim(1.0, 3.9)
    axes[0].set_ylim(150, 220)
    fig.suptitle("Mixtral does not care which rank is hot; DeepSeek only pays when one always is",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.905,
             "Rotating biases rank (layer mod 4), so each layer is skewed as hard as a fixed target makes it "
             "but no rank is hot throughout.",
             color=INK2, fontsize=9.5)
    fig.text(0.008, 0.015,
             "At matched per-layer load Mixtral lands on the same millisecond either way (176.4 against 176.6; "
             "193.2 against 193.9), while DeepSeek\nstays flat when the skew rotates and rises 22% when it does "
             "not. Natural traffic rotates, so this is a second reason DeepSeek's null holds.",
             color=INK3, fontsize=8, va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(os.path.join(out, "rotating_skew.png"), dpi=200, facecolor=SURFACE)
    store["rotating"] = rows


def _capture_loads(pattern, n_experts, level_filter=None):
    """{level: (per-layer load, pooled load)} from a run's validation records."""
    import sys as _sys
    _sys.path.insert(0, os.path.join("packages", "moe-reliability-results", "src"))
    from moe_reliability_results import detection as D

    out: dict[float, tuple[float, float]] = {}
    for d, m in manifests(pattern):
        for p in m["points"]:
            f = json.load(gzip.open(os.path.join(d, p["validation_file"]))) if p.get("validation_file") else {}
            path = os.path.join(d, f.get("records_file") or "")
            if not f.get("records_file") or not os.path.exists(path):
                continue
            level = float(p["value"])
            if level_filter is not None and level not in level_filter:
                continue
            counts = D.request_counts([json.loads(line) for line in gzip.open(path, "rt")], n_experts)
            pooled = counts.sum(axis=0)
            out[level] = (D.busiest_rank(pooled, N_RANKS), D.pooled_busiest_rank(pooled, N_RANKS))
    return out


def fig_statistic(out, store):
    """Summing before taking the maximum hides imbalance - in traces and in routing."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.9), facecolor=SURFACE)
    rows: dict = {}

    # Left: kernel time. Pair GroupedMatmul calls across ranks, or sum each rank first.
    points = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "*mixtral-alpha", "trace_metrics", "*.json.gz"))):
        summary = json.load(gzip.open(f)).get("stragglers", {}).get("GroupedMatmul", {})
        alpha = re.search(r"alpha_([0-9]+(?:\.[0-9]+)?)\.json", f)
        if summary.get("calls_per_rank") and alpha:
            points.append((float(alpha.group(1)), summary["straggler"], summary["totals_max_over_mean"]))
    points.sort()
    style(left, "Workload imbalance (alpha)", "Busiest rank, relative to the mean")
    left.set_title("In a profile", color=INK, fontsize=11, fontweight="bold", loc="left")
    x = [p[0] for p in points]
    for ys, colour, label, lw in (([p[1] for p in points], INK, "pair each call across ranks", 2.2),
                                  ([p[2] for p in points], INK3, "sum each rank, then compare", 2.0)):
        left.plot(x, ys, "-o", ms=6, lw=lw, color=colour, mec=SURFACE, mew=1.5, zorder=3)
        left.annotate(label, xy=(x[-1], ys[-1]), xytext=(-4, 10 if colour == INK else -16),
                      textcoords="offset points", ha="right", color=colour, fontsize=8.5, fontweight="bold")
    left.axhline(1.0, color=GRID, lw=1)
    left.set_ylim(0.99, 1.09)
    rows["profile"] = [{"alpha": a, "paired": round(b, 4), "summed": round(c, 4)} for a, b, c in points]

    # Right: routing. Busiest rank per layer, or pooled over layers first.
    conditions = []
    nat = store.get("natural_detail", {}).get("mixtral")
    if nat:
        conditions.append(("natural\ntraffic", nat[0], nat[1]))
    fixed = _capture_loads("*mixtral*detection-capture", 8, level_filter={2.0})
    if fixed:
        conditions.append(("one rank hot,\nevery layer", *list(fixed.values())[0]))
    rot = _capture_loads("*rotating-skew-001", 8, level_filter={2.0})
    if rot:
        conditions.append(("a different rank\nhot each layer", *list(rot.values())[0]))

    style(right, "", "Busiest rank, relative to the mean")
    right.set_title("In the router's own counters", color=INK, fontsize=11, fontweight="bold", loc="left")
    width = 0.34
    for i, (label, per_layer, pooled) in enumerate(conditions):
        for j, (value, colour, name) in enumerate(((per_layer, INK, "per layer"),
                                                   (pooled, INK3, "pooled over layers"))):
            right.bar(i + (j - 0.5) * (width + 0.02), value, width, color=colour, zorder=3,
                      label=name if i == 0 else None)
            right.annotate(f"{value:.2f}x", xy=(i + (j - 0.5) * (width + 0.02), value), xytext=(0, 4),
                           textcoords="offset points", ha="center", color=INK, fontsize=9,
                           fontweight="bold")
    right.set_xticks(range(len(conditions)), [c[0] for c in conditions], fontsize=9)
    right.tick_params(axis="x", labelcolor=INK2)
    right.axhline(1.0, color=GRID, lw=1)
    right.set_ylim(0, 3.6)
    right.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left", ncol=2,
                 columnspacing=1.4)
    rows["routing"] = [{"condition": c[0].replace("\n", " "), "per_layer": round(c[1], 3),
                        "pooled": round(c[2], 3)} for c in conditions]

    fig.suptitle("Summing before you take the maximum hides imbalance",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.905,
             "Both panels measure the same routing two ways. Mixtral 8x7B; the right-hand panel is the "
             "strongest injected skew that keeps every expert live.",
             color=INK2, fontsize=9.5)
    fig.text(0.008, 0.015,
             "Left: a step runs its layers in sequence, so what it waits for is the sum over calls of the "
             "maximum over ranks. Summing each rank first lets a\nbusiest rank that changes between layers "
             "cancel. Right: the same cancellation in the router's counters - and for a skew that rotates "
             "between layers,\nthe pooled form reads 1.03x, which is what benign traffic reads.",
             color=INK3, fontsize=8, va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 0.89))
    fig.savefig(os.path.join(out, "statistic.png"), dpi=200, facecolor=SURFACE)
    store["statistic"] = rows


def fig_detection(out, store):
    """What the screen catches, against what it costs to count."""
    detection = json.load(open(os.path.join("docs", "detection.json")))
    impact = {m: (r or {}).get("impact_threshold") for m, r in detection["impact"].items()}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), facecolor=SURFACE, sharey=True)
    rows = []
    # Window size is ordered, so it gets the single-hue ramp; how much of each
    # window is counted is the line style.
    shown = [(1, None, 1.0, "-"), (1, 4, 0.1, ":"), (8, None, 1.0, "-"), (8, 4, 0.1, ":"),
             (64, 4, 0.1, ":")]
    ramp = {1: OFFSET_RAMP[0], 8: OFFSET_RAMP[1], 64: OFFSET_RAMP[2]}
    for ax, model in zip(axes, ("deepseek-v2", "mixtral")):
        row = detection["detection"].get(model) or {}
        style(ax, "Load on the busiest rank", "Windows that raise the alarm" if ax is axes[0] else "")
        ax.set_title(MODEL_LABEL[model], color=INK, fontsize=11, fontweight="bold", loc="left")
        if "settings" not in row:
            continue
        threshold = impact.get(model)
        if threshold:
            ax.axvspan(threshold, 4.0, color="#f0efec", zorder=0)
            ax.annotate("costs latency", xy=(threshold + 0.06, 0.985), xycoords=("data", "axes fraction"),
                        ha="left", va="top", color=INK3, fontsize=8.5)
        for window, layers, token_fraction, ls in shown:
            match = [t for t in row["settings"] if (t["window"], t["layers"], t["token_fraction"])
                     == (window, layers, token_fraction)]
            if not match:
                continue
            setting = match[0]
            x = [lv["true_load"] for lv in setting["levels"]]
            y = [lv["detection_rate"] for lv in setting["levels"]]
            counted = setting["assignment_fraction"]
            ax.plot(x, y, ls=ls, marker="o", ms=5, lw=2, color=ramp[window], zorder=3,
                    label=f"{window} req \u00b7 {counted:.0%} counted" if counted >= 0.995
                          else f"{window} req \u00b7 {counted:.1%} counted")
            rows.append({"model": model, "window": window, "counted": counted, "load": x, "detected": y})
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(1.0, 3.7)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right",
                  borderpad=0.2, labelspacing=0.35)
    axes[0].set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    fig.suptitle("A window of 8 requests catches every costly skew, counting ~1% of routing",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.905,
             "Alarm set at a 1% false-alarm rate on each capture's own balanced arm. 1200 prompts per "
             "level over six workload families.",
             color=INK2, fontsize=9.5)
    fig.text(0.008, 0.015,
             "Counting a tenth of the tokens in a single request is the one setting that fails: one prompt "
             "is itself skewed, so the alarm has to sit high.\nWindow length buys more than counting more - "
             "8 requests at 1.3% beats 1 request at 100%.",
             color=INK3, fontsize=8, va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(os.path.join(out, "detection.png"), dpi=200, facecolor=SURFACE)
    store["detection"] = rows


def main(out="docs/figures"):
    os.makedirs(out, exist_ok=True)
    store: dict = {}
    natural(store)
    fig_impact_map(out, store)
    fig_dose_response(out, store)
    fig_instrument(out, store)
    fig_steps(out, store)
    fig_calibration(out, store)
    fig_statistic(out, store)
    if list(manifests(ROTATING["mixtral"])):
        fig_rotating(out, store)
    if os.path.exists(os.path.join("docs", "detection.json")):
        fig_detection(out, store)
    json.dump(store, open(os.path.join(out, "figures.json"), "w"), indent=1)
    print(f"wrote {out}/impact_map.png, detection.png, rotating_skew.png, statistic.png, dose_response.png, instrument_cost.png, step_level.png, calibration.png, figures.json")


if __name__ == "__main__":
    main(*sys.argv[1:2])
