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


def calibration():
    """{model: {level: (busiest rank load, live experts)}} from the calibration runs."""
    out: dict[str, dict[float, tuple[float, int]]] = {}
    for d, m in manifests("*rbias-calibration"):
        if m.get("status") != "completed":
            continue
        model = os.path.basename(d).split("_")[2]
        for p in m["points"]:
            v = p.get("validation_summary") or {}
            if v.get("rank_max_over_mean") is not None:
                out.setdefault(model, {})[float(p["value"])] = (v["rank_max_over_mean"], v["active_experts"])
    return out


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
    ax.axvspan(1.0, 1.15, color="#f0efec", zorder=0)
    ax.annotate("where natural\ntraffic sits", xy=(1.075, 0.995), xycoords=("data", "axes fraction"),
                ha="center", va="top", color=INK3, fontsize=8.5, linespacing=1.3)
    rows = []
    for pattern, mode, ls, marker in (("*rbias-deepseek-001", "eager", "-", "o"),
                                      ("*rbias-mixtral-001", "eager", "-", "o"),
                                      ("*rbias-graph-000", "graph", "--", "s"),
                                      ("*rbias-graph-001", "graph", "--", "s")):
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
    ax.set_xlim(0.9, 3.7)
    ax.set_ylim(132, 232)
    fig.text(0.012, 0.015,
             "Eager DeepSeek is an upper bound: its balanced point ran without the measurement plugin "
             "(see the instrument figure).\nLevels that collapse routing onto fewer live experts are excluded.",
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
    ax.set_ylim(0, 265)
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
    ax.axhspan(1.0, 1.15, color="#f0efec", zorder=0)
    ax.annotate("natural traffic sits here", xy=(4.05, 1.08), ha="right", va="center",
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


def main(out="docs/figures"):
    os.makedirs(out, exist_ok=True)
    store: dict = {}
    fig_dose_response(out, store)
    fig_instrument(out, store)
    fig_steps(out, store)
    fig_calibration(out, store)
    json.dump(store, open(os.path.join(out, "figures.json"), "w"), indent=1)
    print(f"wrote {out}/dose_response.png, instrument_cost.png, step_level.png, calibration.png, figures.json")


if __name__ == "__main__":
    main(*sys.argv[1:2])
