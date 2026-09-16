###
# hta_plot.py
#
# Visualization of holistic trace analysis (HTA) results.
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

__all__ = [
    "hta_frames",
    "plot_idle_comparison",
    "plot_temporal_stacked",
    "plot_idle_category_stacked",
    "plot_per_rank_load",
    "plot_load_imbalance",
    "plot_overlap",
    "plot_kernel_types",
    "plot_throughput",
    "comparison_table",
    "HTA_RC_PARAMS",
]

IMB_LABEL = {0: "balanced", 100: "imbalanced"}
IMB_COLOR = {0: "#2a9d8f", 100: "#e76f51"}          # teal  vs  terracotta
IMB_ORDER = [0, 100]
KTYPE_ORDER = ["COMPUTATION", "COMMUNICATION", "MEMORY"]
KTYPE_COLOR = {"COMPUTATION": "#264653",
               "COMMUNICATION": "#e9c46a",
               "MEMORY": "#8ab17d"}
IDLE_ORDER = ["host_wait", "kernel_wait", "other"]
IDLE_COLOR = {"host_wait": "#457b9d",
              "kernel_wait": "#f4a261",
              "other": "#e76f51"}
TB_ORDER = ["compute", "non_compute", "idle"]
TB_COLOR = {"compute": "#2a9d8f", "non_compute": "#e9c46a", "idle": "#e76f51"}

HTA_RC_PARAMS = {
    "figure.dpi": 110, "savefig.dpi": 140, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
}

def hta_frames(store, **filters):
    tables = {"rank": [], "idle_categories": [], "kernel_types": [], "runs": []}
    for run in store.runs(**filters):
        frames = run.hta_frames()
        if not frames:
            continue
        for name in tables:
            df = frames.get(name)
            if df is not None and not df.empty:
                df = df.copy()
                df.insert(0, "run_id", run.id)
                tables[name].append(df)
    out = []
    for name in ("rank", "idle_categories", "kernel_types", "runs"):
        out.append(pd.concat(tables[name], ignore_index=True) if tables[name] else pd.DataFrame())
    return tuple(out)

def _conditions(df):
    c = (df[["model", "batch"]].drop_duplicates()
           .sort_values(["model", "batch"]))
    return [tuple(x) for x in c.to_numpy()]


def _clabel(model, batch):
    return f"{model}\nb{batch}"


def _grouped_bars(ax, conds, value_fn, ylabel, title,
                  err_fn=None, pct=True):
    x = np.arange(len(conds))
    w = 0.38
    for j, imb in enumerate(IMB_ORDER):
        vals = [value_fn(m, b, imb) for (m, b) in conds]
        err = None
        if err_fn is not None:
            lo, hi = zip(*[err_fn(m, b, imb) for (m, b) in conds])
            err = [list(lo), list(hi)]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, yerr=err, capsize=4,
                      color=IMB_COLOR[imb], label=IMB_LABEL[imb],
                      alpha=0.92, edgecolor="white", linewidth=0.6)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.0f}" if pct else f"{v:.2f}",
                            (xi, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([_clabel(m, b) for (m, b) in conds])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(title="run")
    return ax



def plot_idle_comparison(run_df, rank_df, save=None):
    conds = _conditions(run_df)

    def mean_v(m, b, imb):
        s = rank_df[(rank_df.model == m) & (rank_df.batch == b) & (rank_df.imbalance == imb)]
        return s.idle_pctg.mean() if not s.empty else np.nan

    def err_v(m, b, imb):
        s = rank_df[(rank_df.model == m) & (rank_df.batch == b) & (rank_df.imbalance == imb)]
        if s.empty:
            return (0, 0)
        mu = s.idle_pctg.mean()
        return (max(mu - s.idle_pctg.min(), 0), max(s.idle_pctg.max() - mu, 0))

    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(conds)), 5))
    _grouped_bars(ax, conds, mean_v, "NPU idle time",
                  "NPU idle time: balanced vs imbalanced\n(whiskers = min..max across ranks)",
                  err_fn=err_v, pct=True)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_temporal_stacked(rank_df, save=None):
    conds = _conditions(rank_df)
    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(conds)), 5.2))
    for j, imb in enumerate(IMB_ORDER):
        bottoms = np.zeros(len(conds))
        comps = {
            "compute": [_mean(rank_df, m, b, imb, "compute_pctg") for (m, b) in conds],
            "non_compute": [_mean(rank_df, m, b, imb, "non_compute_pctg") for (m, b) in conds],
            "idle": [_mean(rank_df, m, b, imb, "idle_pctg") for (m, b) in conds],
        }
        for seg in TB_ORDER:
            vals = np.array(comps[seg], dtype=float)
            ax.bar(x + (j - 0.5) * w, vals, w, bottom=bottoms,
                   color=TB_COLOR[seg], edgecolor="white", linewidth=0.5,
                   label=seg if j == 0 else None)
            bottoms += np.nan_to_num(vals)
        for xi in x:
            ax.annotate(IMB_LABEL[imb][:3], (xi + (j - 0.5) * w, 101),
                        ha="center", fontsize=7, color=IMB_COLOR[imb])
    ax.set_xticks(x)
    ax.set_xticklabels([_clabel(m, b) for (m, b) in conds])
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("share of NPU kernel time (mean over ranks)")
    ax.set_title("Temporal breakdown: balanced (left) vs imbalanced (right)",
                 fontweight="bold")
    ax.legend(title="time class", ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def _mean(rank_df, m, b, imb, col):
    s = rank_df[(rank_df.model == m) & (rank_df.batch == b) & (rank_df.imbalance == imb)]
    return float(s[col].mean()) if not s.empty else np.nan


def plot_idle_category_stacked(idle_cat_df, save=None):
    if idle_cat_df.empty:
        print("no idle-category data")
        return None
    conds = _conditions(idle_cat_df)
    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(conds)), 5.2))
    for j, imb in enumerate(IMB_ORDER):
        # total idle per category per condition, normalised to % of that run's idle
        bottoms = np.zeros(len(conds))
        pct = {cat: [] for cat in IDLE_ORDER}
        for (m, b) in conds:
            s = idle_cat_df[(idle_cat_df.model == m) & (idle_cat_df.batch == b)
                            & (idle_cat_df.imbalance == imb)]
            tot = s["idle_time"].sum()
            for cat in IDLE_ORDER:
                v = s[s.idle_category == cat]["idle_time"].sum()
                pct[cat].append(100.0 * v / tot if tot else np.nan)
        for cat in IDLE_ORDER:
            vals = np.array(pct[cat], dtype=float)
            ax.bar(x + (j - 0.5) * w, vals, w, bottom=bottoms,
                   color=IDLE_COLOR[cat], edgecolor="white", linewidth=0.5,
                   label=cat if j == 0 else None)
            bottoms += np.nan_to_num(vals)
        for xi in x:
            ax.annotate(IMB_LABEL[imb][:3], (xi + (j - 0.5) * w, 101),
                        ha="center", fontsize=7, color=IMB_COLOR[imb])
    ax.set_xticks(x)
    ax.set_xticklabels([_clabel(m, b) for (m, b) in conds])
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("share of idle time")
    ax.set_title("Idle-time breakdown: balanced (left) vs imbalanced (right)",
                 fontweight="bold")
    ax.legend(title="idle cause", ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_per_rank_load(rank_df, metric="busy_us", save=None):
    conds = _conditions(rank_df)
    n = len(conds)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             squeeze=False)
    ylabel = {"busy_us": "busy time (compute+comm, a.u.)",
              "idle_pctg": "idle %",
              "compute_us": "compute time (a.u.)"}.get(metric, metric)
    for k, (m, b) in enumerate(conds):
        ax = axes[k // ncol][k % ncol]
        for imb in IMB_ORDER:
            s = (rank_df[(rank_df.model == m) & (rank_df.batch == b)
                         & (rank_df.imbalance == imb)]
                 .sort_values("rank"))
            if s.empty:
                continue
            ax.plot(s["rank"], s[metric], marker="o", color=IMB_COLOR[imb],
                    label=IMB_LABEL[imb])
            ax.fill_between(s["rank"], s[metric], alpha=0.10, color=IMB_COLOR[imb])
        ax.set_title(f"{m}  b{b}", fontsize=10, fontweight="bold")
        ax.set_xlabel("rank (NPU)")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.margins(x=0.05)
        if k == 0:
            ax.legend(fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Per-rank {ylabel}: flat = balanced, spiky = imbalanced",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_load_imbalance(run_df, save=None):
    conds = _conditions(run_df)

    def v(m, b, imb):
        s = run_df[(run_df.model == m) & (run_df.batch == b) & (run_df.imbalance == imb)]
        return float(s.busy_cov.iloc[0]) if not s.empty else np.nan

    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(conds)), 5))
    _grouped_bars(ax, conds, v,
                  "coeff. of variation of per-rank busy time",
                  "Load imbalance across ranks (std/mean of busy time)\nhigher = more skewed",
                  pct=False)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_overlap(run_df, save=None):
    conds = _conditions(run_df)

    def v(m, b, imb):
        s = run_df[(run_df.model == m) & (run_df.batch == b) & (run_df.imbalance == imb)]
        return float(s.overlap_pctg_mean.iloc[0]) if not s.empty else np.nan

    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(conds)), 5))
    _grouped_bars(ax, conds, v, "comm/comp overlap",
                  "Communication–computation overlap (mean over ranks)\nhigher = comm better hidden",
                  pct=True)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_kernel_types(kern_df, save=None):
    if kern_df.empty:
        print("no kernel-type data")
        return None
    conds = _conditions(kern_df)
    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(conds)), 5.2))
    for j, imb in enumerate(IMB_ORDER):
        bottoms = np.zeros(len(conds))
        for kt in KTYPE_ORDER:
            vals = []
            for (m, b) in conds:
                s = kern_df[(kern_df.model == m) & (kern_df.batch == b)
                            & (kern_df.imbalance == imb) & (kern_df.kernel_type == kt)]
                vals.append(float(s.percentage.iloc[0]) if not s.empty else 0.0)
            vals = np.array(vals, dtype=float)
            ax.bar(x + (j - 0.5) * w, vals, w, bottom=bottoms,
                   color=KTYPE_COLOR[kt], edgecolor="white", linewidth=0.5,
                   label=kt if j == 0 else None)
            bottoms += np.nan_to_num(vals)
        for xi in x:
            ax.annotate(IMB_LABEL[imb][:3], (xi + (j - 0.5) * w, bottoms[list(x).index(xi)] + 1),
                        ha="center", fontsize=7, color=IMB_COLOR[imb])
    ax.set_xticks(x)
    ax.set_xticklabels([_clabel(m, b) for (m, b) in conds])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("share of NPU kernel time")
    ax.set_title("NPU kernel-type breakdown: balanced (left) vs imbalanced (right)",
                 fontweight="bold")
    ax.legend(title="kernel type", ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_throughput(run_df, save=None):
    conds = _conditions(run_df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(11, 2.4 * len(conds)), 5))

    def thr(m, b, imb):
        s = run_df[(run_df.model == m) & (run_df.batch == b) & (run_df.imbalance == imb)]
        return float(s.throughput_rel.iloc[0]) if not s.empty else np.nan

    def util(m, b, imb):
        s = run_df[(run_df.model == m) & (run_df.batch == b) & (run_df.imbalance == imb)]
        return float(s.compute_pctg_mean.iloc[0]) if not s.empty else np.nan

    _grouped_bars(ax1, conds, thr, "relative throughput",
                  "Throughput proxy (balanced = 100%)\nfrom slowest-rank busy time; higher = faster",
                  pct=True)
    ax1.axhline(100, color="grey", ls="--", lw=1)
    _grouped_bars(ax2, conds, util, "mean compute %",
                  "Effective NPU utilisation (mean compute share)\nhigher = less wasted",
                  pct=True)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def comparison_table(run_df):
    metrics = ["idle_pctg_mean", "idle_pctg_max", "compute_pctg_mean",
               "non_compute_pctg_mean", "overlap_pctg_mean", "busy_cov",
               "bottleneck_busy_us", "throughput_rel"]
    rows = []
    for (m, b), g in run_df.groupby(["model", "batch"]):
        bal = g[g.imbalance == 0]
        imb = g[g.imbalance == 100]
        if bal.empty or imb.empty:
            continue
        row = {"model": m, "batch": b}
        for mt in metrics:
            bv, iv = float(bal[mt].iloc[0]), float(imb[mt].iloc[0])
            row[f"{mt}__bal"] = round(bv, 2)
            row[f"{mt}__imb"] = round(iv, 2)
            row[f"{mt}__delta"] = round(iv - bv, 2)
        rows.append(row)
    return pd.DataFrame(rows)
