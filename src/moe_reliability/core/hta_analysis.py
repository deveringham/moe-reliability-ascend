###
# hta_analysis.py
#
# Holistic Trace Analysis (HTA) of vLLM worker traces: temporal breakdown,
# idle-time causes, communication/computation overlap and kernel-type shares
# per rank, aggregated into tidy tables.
#
# Requires the optional `hta` extra (HolisticTraceAnalysis).
# Dylan Everingham
###

import glob, gzip, json, os
import numpy as np
import pandas as pd

IMB_LABEL = {0: "balanced", 100: "imbalanced"}


def worker_trace_files(trace_dir):
    """Return {rank: basename} for GPU worker traces only (skip CPU/frontend traces)."""
    paths = sorted(set(
        glob.glob(os.path.join(trace_dir, "*.json.gz")) +
        glob.glob(os.path.join(trace_dir, "*.json"))
    ))
    mapping = {}
    for p in paths:
        try:
            with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
                d = json.load(f)
        except Exception as e:
            print(f"skip (unreadable): {os.path.basename(p)} -> {e}")
            continue
        rank = (d.get("distributedInfo") or {}).get("rank")
        has_gpu = bool(d.get("deviceProperties"))   # worker traces have GPU device props
        if rank is None or not has_gpu:
            print(f"skip (non-worker/no-rank): {os.path.basename(p)}")
            continue
        mapping[int(rank)] = os.path.basename(p)
    return mapping


def _safe(call, tag, what):
    try:
        return call()
    except Exception as e:                                    # noqa: BLE001
        print(f"  ! {what} failed for {tag}: {e}")
        return None


def extract_metrics(results):
    """Call the HTA analyzers once per run and return tidy frames.

    Returns
    -------
    rank_df     : per (model, batch, imbalance, rank) temporal + overlap metrics
    idle_cat_df : per (model, batch, imbalance, rank, idle_category) idle time
    kern_df     : per (model, batch, imbalance, kernel_type) time share
    run_df      : per (model, batch, imbalance) aggregated / derived metrics
    """
    rank_rows, idle_rows, kern_rows = [], [], []

    for (model, batch, imb), ta in sorted(results.items()):
        tag = f"{model} b{batch} {IMB_LABEL.get(imb, imb)}"
        print(f"extracting: {tag}")

        # 1) temporal breakdown --------------------------------------------- #
        tb = _safe(lambda: ta.get_temporal_breakdown(visualize=False),
                   tag, "temporal_breakdown")
        if tb is None or tb.empty:
            print(f"  ! no temporal breakdown for {tag} -> run skipped")
            continue
        ranks = sorted(int(r) for r in tb["rank"].tolist())

        # 2) comm / comp overlap -------------------------------------------- #
        ov = _safe(lambda: ta.get_comm_comp_overlap(visualize=False),
                   tag, "comm_comp_overlap")
        ov_map = {}
        if ov is not None and not ov.empty:
            ov_map = dict(zip(ov["rank"].astype(int),
                              ov["comp_comm_overlap_pctg"].astype(float)))

        for _, r in tb.iterrows():
            rk = int(r["rank"])
            comp = float(r["compute_time(us)"])
            ncomp = float(r["non_compute_time(us)"])
            rank_rows.append(dict(
                model=model, batch=batch, imbalance=imb, rank=rk,
                idle_us=float(r["idle_time(us)"]),
                compute_us=comp, non_compute_us=ncomp,
                kernel_us=float(r["kernel_time(us)"]),
                busy_us=comp + ncomp,
                idle_pctg=float(r["idle_time_pctg"]),
                compute_pctg=float(r["compute_time_pctg"]),
                non_compute_pctg=float(r["non_compute_time_pctg"]),
                overlap_pctg=ov_map.get(rk, np.nan),
            ))

        # 3) idle-time breakdown (host / kernel / other) over ALL ranks ----- #
        ib = _safe(lambda: ta.get_idle_time_breakdown(ranks=ranks,
                                                      visualize=False),
                   tag, "idle_time_breakdown")
        if ib is not None:
            idf = ib[0] if isinstance(ib, tuple) else ib
            if idf is not None and not idf.empty:
                g = (idf.groupby(["rank", "idle_category"], as_index=False)
                        ["idle_time"].sum())
                for _, r in g.iterrows():
                    idle_rows.append(dict(
                        model=model, batch=batch, imbalance=imb,
                        rank=int(r["rank"]),
                        idle_category=str(r["idle_category"]),
                        idle_time=float(r["idle_time"]),
                    ))

        # 4) kernel-type breakdown ------------------------------------------ #
        kb = _safe(lambda: ta.get_gpu_kernel_breakdown(visualize=False),
                   tag, "gpu_kernel_breakdown")
        if kb is not None:
            ktdf = kb[0] if isinstance(kb, tuple) else kb
            if ktdf is not None and not ktdf.empty:
                for _, r in ktdf.iterrows():
                    kern_rows.append(dict(
                        model=model, batch=batch, imbalance=imb,
                        kernel_type=str(r["kernel_type"]),
                        sum=float(r["sum"]),
                        percentage=float(r["percentage"]),
                    ))

    rank_df = pd.DataFrame(rank_rows)
    idle_cat_df = pd.DataFrame(idle_rows)
    kern_df = pd.DataFrame(kern_rows)

    # ---- per-run aggregates / derived metrics ----------------------------- #
    run_rows = []
    for (model, batch, imb), g in rank_df.groupby(["model", "batch", "imbalance"]):
        busy = g["busy_us"].to_numpy(dtype=float)
        mean_busy = busy.mean() if busy.size else np.nan
        cov = float(busy.std() / mean_busy) if mean_busy else np.nan       # load skew
        spread = float((busy.max() - busy.min()) / busy.max()) if busy.max() else np.nan
        run_rows.append(dict(
            model=model, batch=batch, imbalance=imb, n_ranks=int(len(g)),
            idle_pctg_mean=float(g["idle_pctg"].mean()),
            idle_pctg_max=float(g["idle_pctg"].max()),
            idle_pctg_min=float(g["idle_pctg"].min()),
            compute_pctg_mean=float(g["compute_pctg"].mean()),
            non_compute_pctg_mean=float(g["non_compute_pctg"].mean()),
            overlap_pctg_mean=float(g["overlap_pctg"].mean()),
            busy_cov=cov,                       # std/mean of per-rank busy time
            busy_spread=spread,                 # (max-min)/max of per-rank busy time
            bottleneck_busy_us=float(busy.max()),   # slowest rank paces the step
        ))
    run_df = pd.DataFrame(run_rows)

    # relative throughput proxy: same #samples per run, so throughput ~ 1/bottleneck.
    # normalise within each (model, batch) so the balanced run = 100%.
    run_df["throughput_rel"] = np.nan
    for (model, batch), g in run_df.groupby(["model", "batch"]):
        base = g.loc[g.imbalance == 0, "bottleneck_busy_us"]
        if not base.empty and base.iloc[0] > 0:
            b = float(base.iloc[0])
            run_df.loc[g.index, "throughput_rel"] = 100.0 * b / g["bottleneck_busy_us"]
    return rank_df, idle_cat_df, kern_df, run_df


def load_trace_analysis(trace_dir):
    """Build an HTA TraceAnalysis for a trace directory (imports HTA lazily)."""
    from hta.trace_analysis import TraceAnalysis
    tf = worker_trace_files(trace_dir)
    print(f"{trace_dir} -> {tf}")
    return TraceAnalysis(trace_dir=trace_dir, trace_files=tf)   # explicit files, not whole dir
