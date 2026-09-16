###
# plots.py
#
# Visualization of results.
# All functions return a matplotlib.figure.Figure.
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from . import schema

__all__ = [
    "plot_workload_sweep",
    "plot_workload_sweep_cvs",
    "plot_latency_sweep",
    "plot_latency_comparison",
    "plot_trace_sweep",
    "plot_kernel_histograms",
    "plot_rank_kernel_means",
    "plot_expert_load",
    "moe_imbalance_overview_inputs",
    "plot_moe_imbalance_overview",
    "plot_run",
    "save_figures",
]

_STYLE = "seaborn-v0_8-whitegrid"


def _uniform_bins(lo: float, hi: float, n_bins: int = 100) -> np.ndarray | int:
    bin_width = (hi - lo) / n_bins
    if not np.isfinite(bin_width) or bin_width <= 0:
        return n_bins
    return np.arange(lo, hi + bin_width, bin_width)


def plot_workload_sweep(workload_set: Mapping[str, Any], title: str | None = None):
    if title is None:
        title = f"Synthetic Workload Generation, Max Allowed Prompt Repeats={workload_set.get('max_repeats')}"
    return plot_workload_sweep_cvs(workload_set["workloads"], workload_set["target_alphas"],
                                   np.asarray(workload_set["cv_nat"], dtype=float), title)


def plot_workload_sweep_cvs(workloads, alphas, cv_nat, title):
    with plt.style.context(_STYLE):
        target_ls = list(workloads.keys())
        target_cvs_list = list(workloads[target_ls[0]].keys())
        obtained_cvs_list = {l: np.stack([np.asarray(workloads[l][cvs]['obtained_cvs']) for cvs in target_cvs_list]) for l in target_ls} # (n_alpha, n_layers)
        n_alphas = len(alphas)
        alphas = np.asarray(alphas, dtype=float)
        cv_nat = np.asarray(cv_nat, dtype=float)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Color maps for distinguishing lengths and CVs
        l_colors = plt.cm.plasma(np.linspace(0, 1, len(target_ls)))

        # Plot 1: CV Median + spread across layers (effective alpha) vs target alpha
        ax = axes[0]
        l = target_ls[-1] # Use the largest workload length
        effective_alphas = obtained_cvs_list[l] / np.stack([cv_nat]*n_alphas) # (n_alpha, n_layers)

        lo, mid, hi = np.percentile(effective_alphas, [10, 50, 90], axis=1)
        ax.fill_between(alphas, lo, hi, alpha=0.25, color='C0', label='layers 10–90%')
        ax.plot(alphas, mid, 'o-', color='C0', label='median layer')
        lim = (alphas.min() * 0.95, alphas.max() * 1.05)
        ax.plot(lim, lim, 'k--', alpha=0.5, label='ideal')

        ax.set_xlim(lim)
        ax.set_title(f'Effective Alpha (MCV Median/Spread over Layers) vs. Target Alpha\nWorkload Length: {l}')
        ax.set_xlabel('Target Alpha (CV Target Scaling)')
        ax.set_ylabel('')
        ax.set_xticks(alphas)
        ax.legend()

        # Plot 2: MAE by CV (alpha) (Grouped by workload length)
        ax = axes[1]
        for i, l in enumerate(target_ls):
            x = alphas
            y = [workloads[l][cvs]['mae'] for cvs in target_cvs_list]
            ax.plot(x, y, marker='s', label=f'Workload length: {l}', color=l_colors[i])

        ax.set_title('Workload Length vs. Mean Absolute Error')
        ax.set_xlabel('Target Alpha (CV Target Scaling)')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_xticks(alphas)
        ax.legend()

        # Plot 3: Percent Unique Prompts by Alpha
        ax = axes[2]
        for i, l in enumerate(target_ls):
            x = alphas
            y = [workloads[l][cvs]['percent_unique_prompts']*100 for cvs in target_cvs_list]
            ax.plot(x, y, marker='^', label=f'Workload length: {l}', color=l_colors[i])

        ax.set_title('Alpha vs. Unique Prompts')
        ax.set_xlabel('Target Alpha (CV Target Scaling)')
        ax.set_ylabel('% Unique Prompts')
        ax.set_xticks(alphas)
        ax.legend()

        fig.suptitle(title)
        fig.tight_layout()
    return fig

def plot_latency_sweep(results_dict: Mapping[Any, Sequence[Mapping[str, Any]] | None],
                       sweep_name: str = "Alpha", axis_label: str = "Load Imbalance (Parameterized by Alpha)"):
    # Remove any failed runs
    results_keys = list(results_dict.keys()) # e.g. alphas for synthetic workloads
    results = {k: results_dict[k] for k in results_keys if results_dict[k] is not None}
    results_keys = list(results.keys()) # update

    # Get metrics: TTFT, TPOT
    ttfts = {k: [r['ttft']*1000 for r in results[k]] for k in results_keys}
    tpots = {k: [r['tpot']*1000 for r in results[k]] for k in results_keys}
    all_ttfts = sum(list(ttfts.values()), [])
    all_tpots = sum(list(tpots.values()), [])
    min_ttft = min(all_ttfts)
    min_tpot = min(all_tpots)
    max_ttft = max(all_ttfts)
    max_tpot = max(all_tpots)
    avg_ttfts = {k: np.mean(ttfts[k]) for k in results_keys}
    avg_tpots = {k: np.mean(tpots[k]) for k in results_keys}

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        alpha_colors = plt.cm.plasma(np.linspace(0, 1, len(results_keys)))

        # Plot 1: TTFT
        ax = axes[0][0]
        bins_ttft = _uniform_bins(min_ttft, max_ttft)

        # Hist for each sweep value
        for i, k in enumerate(results_keys):
            ax.hist(ttfts[k], bins=bins_ttft,
                          color=alpha_colors[i],
                          label=f'{sweep_name.lower()} = {k}')

        ax.set_title(f'TTFT vs. {sweep_name} (Imbalance)')
        ax.set_xlabel('TTFT(ms)')
        ax.set_ylabel('Frequency')
        ax.legend()

        # Plot 2: TPOT
        ax = axes[0][1]
        bins_tpot = _uniform_bins(min_tpot, max_tpot)

        for i, k in enumerate(results_keys):
            ax.hist(tpots[k], bins=bins_tpot,
                          color=alpha_colors[i],
                          label=f'{sweep_name.lower()} = {k}')

        ax.set_title(f'TPOT vs. {sweep_name} (Imbalance)')
        ax.set_xlabel('TPOT(ms)')
        ax.set_ylabel('Frequency')
        ax.legend()

        # Plot 3: Avg TPOT vs. sweep value
        ax = axes[1][0]
        x = results_keys
        y = [avg_tpots[k] for k in results_keys]
        ax.plot(x, y, 'o-', color='b')
        ax.set_title(f'Average TPOT vs. {sweep_name} (Imbalance)')
        ax.set_xlabel(axis_label)
        ax.set_ylabel('TPOT(ms)')

        # Plot 4: Avg TTFT vs. sweep value
        ax = axes[1][1]
        x = results_keys
        y = [avg_ttfts[k] for k in results_keys]
        ax.plot(x, y, 'o-', color='b')
        ax.set_title(f'Average TTFT vs. {sweep_name} (Imbalance)')
        ax.set_xlabel(axis_label)
        ax.set_ylabel('TTFT(ms)')

        fig.suptitle('Performance Metrics from vLLM')
        fig.tight_layout()
    return fig


def plot_latency_comparison(results_balanced, results_imbalanced,
                            labels: tuple[str, str] = ("baseline model", "imbalanced model")):

    # Get metrics: TTFT, TPOT
    ttfts_balanced = [r['ttft']*1000 for r in results_balanced]
    ttfts_imbalanced = [r['ttft']*1000 for r in results_imbalanced]
    tpots_balanced = [r['tpot']*1000 for r in results_balanced]
    tpots_imbalanced = [r['tpot']*1000 for r in results_imbalanced]
    all_ttfts = ttfts_balanced + ttfts_imbalanced
    all_tpots = tpots_balanced + tpots_imbalanced
    min_ttft = min(all_ttfts)
    min_tpot = min(all_tpots)
    max_ttft = max(all_ttfts)
    max_tpot = max(all_tpots)

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(2, figsize=(10, 10))

        # Plot 1: TTFT
        ax = axes[0]
        bins_ttft = _uniform_bins(min_ttft, max_ttft)
        ax.hist(ttfts_balanced, bins=bins_ttft, label=labels[0])
        ax.hist(ttfts_imbalanced, bins=bins_ttft, label=labels[1])
        ax.set_title('TTFT vs. Forced Imbalance')
        ax.set_xlabel('TTFT(ms)')
        ax.set_ylabel('Frequency')
        ax.legend()

        # Plot 2: TPOT
        ax = axes[1]
        bins_tpot = _uniform_bins(min_tpot, max_tpot)
        ax.hist(tpots_balanced, bins=bins_tpot, label=labels[0])
        ax.hist(tpots_imbalanced, bins=bins_tpot, label=labels[1])
        ax.set_title('TPOT vs. Forced Imbalance')
        ax.set_xlabel('TPOT(ms)')
        ax.set_ylabel('Frequency')
        ax.legend()
        fig.suptitle('Performance Metrics from vLLM')
        fig.tight_layout()
    return fig


def plot_trace_sweep(trace_results: Mapping[Any, Mapping[str, Any]],
                     sweep_name: str = "Alpha", axis_label: str = "Load Imbalance (Parameterized by Alpha)"):
    x = list(trace_results.keys())
    rows = [trace_results[k] for k in x]
    y1 = [r["max_over_mean"] for r in rows]
    y2 = [r["mean_over_ranks_us"] for r in rows]

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot 1: max over mean
        ax = axes[0]
        ax.plot(x, y1, 'o-', color='b')
        ax.set_title(f'Max over Mean Fused MoE Kernel Time over Ranks vs. {sweep_name}')
        ax.set_xlabel(axis_label)
        ax.set_ylabel('Max / Mean')

        # Plot 2: mean over ranks
        ax = axes[1]
        ax.plot(x, y2, 'o-', color='b')
        ax.set_title(f'Mean Fused MoE Kernel Time over Ranks vs. {sweep_name}')
        ax.set_xlabel(axis_label)
        ax.set_ylabel('Mean Kernel Time (us)')

        fig.tight_layout()
    return fig


def plot_kernel_histograms(results_balanced, results_imbalanced,
                           labels: tuple[str, str] = ("baseline model", "imbalanced model"),
                           xlim: tuple[float, float] | None = None):
    with plt.style.context(_STYLE):
        fig = plt.figure(figsize=(8, 5))

        calls = results_balanced['calls_per_rank']
        durs_balanced = sum(results_balanced['per_rank_durs'], [])
        durs_imbalanced = sum(results_imbalanced['per_rank_durs'], [])
        durs_total = durs_balanced + durs_imbalanced

        total_dur_balanced = results_balanced['total_over_ranks_ms']
        total_dur_imbalanced = results_imbalanced['total_over_ranks_ms']

        # Create uniform bins
        bins = _uniform_bins(min(durs_total), max(durs_total), n_bins=1000)

        # Hist for each
        plt.hist(durs_balanced, bins=bins,
                      label=f'{labels[0]} (total time: {total_dur_balanced:.2f}ms, max: {max(durs_balanced):.2f}us)', color="#4C72B0")
        plt.hist(durs_imbalanced, bins=bins,
                      label=f'{labels[1]} (total time: {total_dur_imbalanced:.2f}ms, max: {max(durs_imbalanced):.2f}us)', color="#C44E52")

        if xlim is not None:
            plt.xlim(xlim)
        plt.title(f'MoE Kernel Times vs. Forced Imbalance\nTotal Calls: {calls}')
        plt.xlabel('Kernel Time (us)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
    return fig


def plot_rank_kernel_means(baseline, imbalanced, use_dom: bool = False,
                           labels: tuple[str, str] = ("baseline", "imbalanced")):
    key = "dom_per_rank_mean_us" if use_dom else "per_rank_mean_us"
    runs = [(labels[0], baseline, "#4C72B0"),
            (labels[1], imbalanced, "#C44E52")]

    ranks = baseline["ranks"]
    x = np.arange(len(ranks))
    w = 0.38

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    for i, (label, d, color) in enumerate(runs):
        vals = np.asarray(d[key], dtype=float)
        mu = vals.mean()
        off = (i - 0.5) * w

        # Left: raw means + dashed line at the run mean
        axL.bar(x + off, vals, w, label=label, color=color)
        axL.axhline(mu, ls="--", lw=1, color=color, alpha=0.7)

        # Right: normalized to the run's own mean -> pure shape/skew
        axR.bar(x + off, vals / mu, w,
                label=f"{label}  (max/mean={vals.max()/mu:.2f})", color=color)

    axL.set(title=f"Per-rank MoE kernel mean time {'(dominant grid)' if use_dom else ''}".strip(),
            xlabel="rank", ylabel="mean per-call kernel time (us)")
    axL.set_xticks(x); axL.set_xticklabels(ranks)
    axL.legend(frameon=False, fontsize=9)

    axR.axhline(1.0, ls=":", lw=1, color="gray")
    axR.set(title="Same, normalized",
            xlabel="rank", ylabel="mean / run-mean")
    axR.set_xticks(x); axR.set_xticklabels(ranks)
    axR.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    return fig


def plot_expert_load(validation: Mapping[str, Any]):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    n_experts = validation["n_experts"]
    expert_ids = np.array(range(n_experts))
    freqs = np.asarray(validation["frequencies"], dtype=float)
    tokens = validation["n_assignments"]
    expected_freq = 1 / n_experts
    ax.bar(expert_ids, freqs, label=f"tokens: {tokens}", alpha=0.7)
    ax.set_title(f"Expert Activation Frequency (Load), router {validation.get('router_id', 0)}")
    ax.set_xlabel(f"Expert Index (0-{n_experts-1})")
    ax.set_ylabel("Activation Frequency")
    ax.axhline(y=expected_freq, color='red', linestyle='--', linewidth=2, label="expected frequency (1/n_experts)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

C_BASE, C_IMB, C_REST = "#4C72B0", "#C44E52", "#DDDDDD"

def _c(imb):
    return C_BASE if imb == 0 else C_IMB


def _pair_positions(model, keys_m):
    batches = sorted({bs for (_, bs, _) in keys_m})
    pos, ticks, labs = {}, [], []
    for bi, bs in enumerate(batches):
        for j, imb in enumerate((0, 100)):
            pos[(model, bs, imb)] = bi * 2.5 + j
        ticks.append(bi * 2.5 + 0.5)
        labs.append(f"{bs}")
    return pos, ticks, labs


def _style_violin(vp, colors):
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.6); body.set_edgecolor("k")
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        if k in vp:
            vp[k].set_color("k"); vp[k].set_linewidth(1.0)


def _fit_ylim(ax, arrays, pad=0.03):
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    m = (hi - lo) * pad or 1.0
    ax.set_ylim(lo - m, hi + m)


def _empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10,
            color="gray", style="italic", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


def _plot_one_model(model, K, T, outdir=None):
    km = sorted([k for k in K if k[0] == model], key=lambda k: (k[1], k[2]))
    tm = sorted([k for k in T if k[0] == model], key=lambda k: (k[1], k[2]))
    
    fig1, ax1 = plt.subplots(figsize=(6, 2))
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    fig3, ax3 = plt.subplots(figsize=(6, 2))
    fig4, ax4 = plt.subplots(figsize=(6, 2))
    fig5, ax5 = plt.subplots(figsize=(6, 2))
    figs = {
        "A_per_rank_moe_time": fig1,
        "B_moe_pct_step": fig2,
        "C_call_durations": fig3,
        "D_tpot": fig4,
        "E_ttft": fig5,
    }
    #for _f in figs.values():
    #    _f.suptitle(f"MoE routing-imbalance overview: {model}", fontsize=16, fontweight="bold")
 
    # -- Panel A: per-rank MoE time, grouped by batch size ---------------
    if km:
        batches = sorted({bs for (_, bs, _) in km})
        nrank = len(K[km[0]]["per_rank_moe_us"])
        w = 0.185
        arm_span = nrank * w + 0.12          # one (batch,arm) cluster + gap
        batch_span = 2 * arm_span + 0.5      # both arms + gap between batches
        ticks, labs = [], []
        for bi, bs in enumerate(batches):
            for ai, imb in enumerate((0, 100)):
                k = (model, bs, imb)
                if k not in K:
                    continue
                base = bi * batch_span + ai * arm_span
                pr = [np.sum(r) / 1000 for r in K[k]["per_rank_moe_us"]]
                xr = base + np.arange(nrank) * w
                ax1.bar(xr, pr, w, color=_c(imb), edgecolor="k", linewidth=.4)
                ratio = max(pr) / min(pr) if min(pr) > 0 else float("nan")
                ax1.text(base + (nrank - 1) * w / 2, max(pr) * 1.02,
                         f"{ratio:.2f}×", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ticks.append(bi * batch_span + arm_span - 0.06)
            labs.append(f"{bs}")
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labs)
        #ax1.set_xlabel("batch size")
        ax1.set_ylabel("total MoE kernel time\n over ranks (ms)")
        ax1.margins(y=0.12)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    else:
        _empty(ax1, f"kernel trace not available for {model}")
    #ax1.set_title("(A) Per-rank MoE compute time (bars = ranks 0–3)\n label = max-to-mean ratio across ranks")
 
    # -- Panel B: MoE kernel time as % of decode step --------------------
    kt = [k for k in km if k in T]
    if kt:
        pos, ticks, labs = _pair_positions(model, kt)
        for k in kt:
            d = K[k]; n = len(d["ranks"]); s = d["steps"]
            moe_part = (d["moe_total_over_ranks_ms"] / n / s)
            total = T[k]["tpot"].mean()
            pct = 100.0 * moe_part / total 
            xp = pos[k]
            ax2.bar(xp, moe_part, width=0.9, color=_c(k[2]), edgecolor="k", linewidth=.3)
            ax2.bar(xp, total - moe_part, width=0.9, bottom=moe_part, color=C_REST, edgecolor="k", linewidth=.3)
            ax2.text(xp, moe_part + 1.5, f"{pct:.0f}%", ha="center", fontsize=9, fontweight="bold")
        upper = max([T[k]["tpot"].mean() for k in km])
        ax2.set_ylim(0, upper+1.5)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labs)
        #ax2.set_xlabel("batch size")
        ax2.set_ylabel("step time (ms)")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        _empty(ax2, f"kernel trace not available for {model}")
    #ax2.set_title("(B) MoE kernel time per decode step\n grey = non-compute overhead")
 
    # -- Panel C: MoE call-duration distributions ------------------------
    if km:
        rng = np.random.default_rng(0)
        pos, ticks, labs = _pair_positions(model, km)
        vd, cols, xs = [], [], []
        for k in km:
            allc = np.concatenate([np.asarray(r, float) for r in K[k]["per_rank_moe_us"]])
            if len(allc) > 15000:
                allc = rng.choice(allc, 15000, replace=False)
            vd.append(allc); cols.append(_c(k[2])); xs.append(pos[k])
        _style_violin(ax3.violinplot(vd, positions=xs, showmedians=True, showmeans=True, widths=0.9), cols)
        _fit_ylim(ax3, vd)
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(labs)
        ax3.set_xlabel("batch size")
        ax3.set_ylabel("per-call duration (µs)")
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    else:
        _empty(ax3, f"kernel trace not available for {model}")
    #ax3.set_title("(C) MoE expert-GEMM call durations")
    
    # -- Panels D & E: TPOT / TTFT distributions -------------------------
    def dist(ax, field, title, ylab):
        if not tm:
            _empty(ax, f"throughput data not available for {model}"); ax.set_title(title); return
        pos, ticks, labs = _pair_positions(model, tm)
        xs, vd, cols = [], [], []
        for k in tm:
            xs.append(pos[k]); vd.append(T[k][field]); cols.append(_c(k[2]))
        _style_violin(ax.violinplot(vd, positions=xs, showmedians=True, showmeans=True, widths=0.9), cols)
        _fit_ylim(ax, vd)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs)
        ax.set_xlabel("batch size")
        ax.set_ylabel(ylab)
        #ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
 
    dist(ax4, "tpot", "(D) End-to-end TPOT distributions", "TPOT (ms)")
    dist(ax5, "ttft", "(E) End-to-end TTFT distributions", "TTFT (ms)")
 
    for _f in figs.values():
        _f.legend(handles=[Patch(facecolor=C_BASE, alpha=.6, edgecolor="k", label="baseline"),
                           Patch(facecolor=C_IMB, alpha=.6, edgecolor="k", label="imbalanced")],
                  loc="upper right", fontsize=10, framealpha=0.9)
 
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        for name, _f in figs.items():
            _f.savefig(os.path.join(outdir, f"{model}_{name}.pdf"),
                       format='pdf', dpi=150, bbox_inches="tight")
    return figs




def moe_imbalance_overview_inputs(store, **filters) -> tuple[dict, dict]:
    
    K: dict[tuple, dict] = {}
    T: dict[tuple, dict] = {}
    for run in store.runs(experiment=schema.EXPERIMENT_FORCED_IMBALANCE, **filters):
        model = run.config["model"]["model_name"]
        bs = run.config["server"]["batch_size"]
        for p in run.points:
            key = (model, bs, p["value"])
            summary = run.trace_summary(p["label"]) if p.get("trace_metrics_file") else None
            if summary:
                K[key] = {
                    "per_rank_moe_us": summary["per_rank_durs"],
                    "moe_total_over_ranks_ms": summary["total_over_ranks_ms"],
                    "ranks": summary["ranks"],
                    "steps": summary["steps"],
                }
            records = run.request_records(p["label"]) if p.get("metrics_file") else None
            if records:
                T[key] = dict(
                    tpot=np.array([r["tpot"] for r in records]) * 1000.0,
                    ttft=np.array([r["ttft"] for r in records]) * 1000.0,
                )
    return K, T


def plot_moe_imbalance_overview(K: Mapping, T: Mapping, outdir: str | os.PathLike | None = None):
    
    models = sorted({m for (m, _, _) in list(K) + list(T)},
                    key=lambda m: (m != "deepseek", m))
    return {m: _plot_one_model(m, K, T, outdir=outdir) for m in models}

def plot_run(run) -> dict[str, Any]:
    
    figs: dict[str, Any] = {}
    param = run.sweep_parameter
    sweep_name = "Alpha" if param == "alpha" else "Imbalance Level"
    axis_label = schema.SWEEP_AXIS_LABELS.get(param, param)

    if run.experiment == schema.EXPERIMENT_SYNTHETIC_WORKLOADS:
        for max_repeats in run.available_workloads():
            try:
                figs[f"workloads_repeats{max_repeats}"] = plot_workload_sweep(run.workloads(max_repeats))
            except (KeyError, FileNotFoundError):
                pass

    results = run.results_by_point()
    results = {k: v for k, v in results.items() if v and all(r.get("ttft") is not None for r in v)}
    if results:
        figs["latency_sweep"] = plot_latency_sweep(results, sweep_name=sweep_name, axis_label=axis_label)
        if run.experiment == schema.EXPERIMENT_FORCED_IMBALANCE and len(results) >= 2:
            keys = list(results)
            figs["latency_comparison"] = plot_latency_comparison(
                results[keys[0]], results[keys[-1]],
                labels=(f"imbalance level {keys[0]}", f"imbalance level {keys[-1]}"))

    traces = {k: v for k, v in run.trace_summaries().items() if v}
    if traces:
        figs["kernel_sweep"] = plot_trace_sweep(traces, sweep_name=sweep_name, axis_label=axis_label)
        if len(traces) >= 2:
            keys = list(traces)
            lo, hi = traces[keys[0]], traces[keys[-1]]
            labels = (f"{param} {keys[0]}", f"{param} {keys[-1]}")
            if lo.get("per_rank_durs") and hi.get("per_rank_durs"):
                figs["kernel_histograms"] = plot_kernel_histograms(lo, hi, labels=labels)
            if lo.get("ranks") == hi.get("ranks"):
                figs["kernel_rank_means"] = plot_rank_kernel_means(lo, hi, labels=labels)

    for p in run.points:
        if p.get("validation_file"):
            figs[f"expert_load_{p['label']}"] = plot_expert_load(run.validation(p["label"]))

    return figs


def save_figures(figs: Mapping[str, Any], outdir: str | os.PathLike, fmt: str = "png",
                 dpi: int = 150, close: bool = True) -> list[Path]:
    
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fig in figs.items():
        if isinstance(fig, Mapping):
            written.extend(save_figures({f"{name}_{k}": v for k, v in fig.items()}, outdir, fmt, dpi, close))
            continue
        path = outdir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
        if close:
            plt.close(fig)
    return written
