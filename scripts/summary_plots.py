"""Bare plots of the summary tables: a title, axes, the data, nothing else.

    uv run python scripts/summary_plots.py [<dir>]

Reads the CSVs that scripts/summary_tables.py writes and draws them into the
same directory, so a plot and the numbers behind it stay together. No captions
and no callouts: these are for a slide the speaker talks over.
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
MODEL = {"deepseek-v2": BLUE, "mixtral": ORANGE}
LABEL = {"deepseek-v2": "DeepSeek-V2-Lite", "mixtral": "Mixtral 8x7B"}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def axes(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}")


def plot_cost_vs_load(table, out):
    """The core picture: what a given imbalance costs, by model and by shape."""
    # Style per shape of skew. The fine sweep is a second run of the same mixtral
    # deployment as "one rank hot": kept separate rather than merged, because one
    # line through both runs' points reads as a wiggle where it is replication.
    style = {"one rank hot": ("-", "o", True),
             "one rank hot, fine levels": (":", "^", True),
             "rotating": ("--", "s", False)}
    series = {}
    for r in table:
        if r["group"] not in ("skew strength", "skew shape"):
            continue
        shape = "rotating" if r["group"] == "skew shape" else (
            "one rank hot, fine levels" if "fine" in r["condition"] else "one rank hot")
        series.setdefault((r["model"], shape), []).append(
            (num(r["busiest_rank"]), num(r["cost_pct"]), num(r["cost_sd_pct"], 0.0)))
    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor=SURFACE)
    axes(ax, "Load on the busiest rank (1.0 = balanced)", "Change in TPOT (%)",
         "Cost of imbalance, by model and shape")
    ax.axhline(0, color=GRID, lw=1)
    for (model, shape), pts in sorted(series.items()):
        pts.sort()
        ls, marker, filled = style[shape]
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                    ls=ls, marker=marker, ms=6, lw=2, capsize=3, elinewidth=1,
                    color=MODEL[model], mec=SURFACE, mew=1.3,
                    mfc=MODEL[model] if filled else SURFACE,
                    label=f"{LABEL[model]}, {shape}")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    save(fig, out, "cost_vs_load.png")


def plot_batch(table, out):
    """Slope of cost against load, within the natural range, per decode batch."""
    by_batch = {}
    for r in table:
        if r["group"] != "decode batch, natural strength":
            continue
        by_batch.setdefault(int(num(r["batch_achieved"], 0)), []).append(
            (num(r["busiest_rank"]), num(r["tpot_ms"])))
    points = []
    for batch, pts in sorted(by_batch.items()):
        if len(pts) < 3:
            continue
        pts.sort()
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        design = np.c_[np.ones_like(x), x]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        se = np.sqrt(np.diag((resid @ resid / max(len(y) - 2, 1))
                             * np.linalg.inv(design.T @ design)))[1]
        points.append((batch, float(coef[1]), float(se)))
    if not points:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor=SURFACE)
    axes(ax, "Decode batch (sequences in flight)", "Change in TPOT per +1x of load (ms)",
         "At natural imbalance, only a large batch pays")
    ax.axhline(0, color=INK3, lw=1.2)
    ax.errorbar([p[0] for p in points], [p[1] for p in points], yerr=[p[2] for p in points],
                ls="-", marker="o", ms=7, lw=2, capsize=4, elinewidth=1, color=ORANGE,
                mec=SURFACE, mew=1.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks([p[0] for p in points], [str(p[0]) for p in points])
    ax.minorticks_off()
    save(fig, out, "cost_vs_batch.png")


def plot_detection(table, out):
    """What the recommended screen catches, against how skewed the traffic is."""
    # One setting only: eight requests, four layers, a tenth of the tokens. Two
    # settings plotted together read as a fold-back in the line.
    want = [r for r in table if r["group"] == "screen" and r["window_requests"] == "8"
            and r["layers"] == "4" and r["token_fraction"] == "0.1"]
    choices = {r.get("layer_choice") for r in want}
    if "random" in choices:
        want = [r for r in want if r.get("layer_choice") == "random"]
    if not want:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor=SURFACE)
    axes(ax, "Load on the busiest rank", "Share of windows that alarm",
         "What the screen catches (8 requests, ~1% of routing)")
    for model in sorted({r["model"] for r in want}):
        pts = sorted((num(r["busiest_rank"]), num(r["detection_rate"]),
                      r.get("costs_latency") == "True") for r in want if r["model"] == model)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=6, lw=2, color=MODEL[model],
                mec=SURFACE, mew=1.3, label=LABEL[model])
        costly = [p for p in pts if p[2]]
        if costly:
            ax.plot([p[0] for p in costly], [p[1] for p in costly], "o", ms=11, mfc="none",
                    mec=MODEL[model], mew=2, label=f"{LABEL[model]}: costs latency")
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    save(fig, out, "detection_rate.png")


def plot_layer_sampling(table, out):
    """Fixed layers against random ones, on a skew confined to a few layers."""
    want = [r for r in table if r["group"] == "layer sampling"]
    if not want:
        return
    want.sort(key=lambda r: (-int(num(r["layers"], 0)), r["layer_choice"]))
    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor=SURFACE)
    axes(ax, "", "Share of windows that alarm",
         "A skew in 6 of 26 layers: fixed layers miss it")
    labels = [f"{r['layers']} layers\n{r['layer_choice']}" for r in want]
    values = [num(r["detection_rate"]) for r in want]
    ax.bar(range(len(values)), values, 0.6,
           color=[GRID if v < 0.1 else BLUE for v in values], zorder=3)
    for i, (v, r) in enumerate(zip(values, want)):
        ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.tick_params(axis="x", labelcolor=INK2)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    save(fig, out, "layer_sampling.png")


def plot_margin(table, out):
    """What each alarm margin buys, on the deployments calibrated so far.

    Margin on the x-axis rather than a false-alarm/warning trade-off curve: the
    false alarm rate is zero at every margin but the last, so the trade-off form
    stacks four of five points on one coordinate.
    """
    want = [r for r in table if r.get("margin") not in (None, "")]
    if not want:
        return
    by_run = {}
    for r in want:
        by_run.setdefault(r["run"], []).append(
            (num(r["margin"]), num(r["false_alarm_rate"]), num(r["warns_before_cost"])))
    fig, axs = plt.subplots(1, len(by_run), figsize=(8.0, 5.0), facecolor=SURFACE, squeeze=False)
    for ax, (run, pts) in zip(axs[0], sorted(by_run.items())):
        pts.sort()
        model = "mixtral" if "mixtral" in run else "deepseek-v2"
        batch = run.split("_bs")[-1].split("_")[0] if "_bs" in run else "?"
        axes(ax, "Alarm margin (0 = at the cost, 1 = at benign traffic)", "Share of windows",
             f"{LABEL[model]}, batch {batch}: what a margin buys")
        x = [p[0] for p in pts]
        ax.plot(x, [p[2] for p in pts], "-o", ms=7, lw=2, color=BLUE, mec=SURFACE, mew=1.4,
                label="skew caught while still harmless")
        ax.plot(x, [p[1] for p in pts], "-s", ms=7, lw=2, color=ORANGE, mec=SURFACE, mew=1.4,
                label="false alarms on benign traffic")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    save(fig, out, "alarm_margin.png")


def main(out="docs/figures/tables"):
    print("writing plots:")
    regimes = rows(os.path.join(out, "regimes.csv"))
    detection = rows(os.path.join(out, "detection.csv"))
    calibration = rows(os.path.join(out, "calibration.csv"))
    plot_cost_vs_load(regimes, out)
    plot_batch(regimes, out)
    plot_detection(detection, out)
    plot_layer_sampling(detection, out)
    plot_margin(calibration, out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
