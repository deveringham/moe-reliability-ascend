###
# trace_analysis.py
#
# Extraction of fused-MoE kernel metrics from PyTorch profiler traces
# written by the vLLM workers (one trace file per rank).
# Dylan Everingham
###

import os, re, glob, gzip, json
from collections import defaultdict

KERNEL = "fused_moe_kernel"


def moe_per_rank(trace_dir):
    out = {}
    for fp in glob.glob(os.path.join(trace_dir, "*rank*.pt.trace.json.gz")):
        print(f'reading directory: {fp}')
        rank = int(re.search(r"rank(\d+)", os.path.basename(fp)).group(1))
        with gzip.open(fp, "rt") as f:
            events = json.load(f)["traceEvents"]

        by_grid = defaultdict(list)
        steps = 0
        for e in events:
            if e.get("ph") != "X":
                continue
            name = e.get("name") or ""
            # 1 AllGather per forward pass -- used as a step counter
            if "ncclDevKernel_AllGather" in name:
                steps += 1
            if (e.get("cat") or "").lower() == "kernel" and KERNEL in name:
                g = (e.get("args") or {}).get("grid")
                by_grid[tuple(g) if isinstance(g, list) else None].append(
                    e.get("dur", 0))

        durs = [d for v in by_grid.values() for d in v]
        out[rank] = {
            "total_us": sum(durs),
            "calls": len(durs),
            "durs": durs,
            "mean_us": sum(durs) / len(durs) if durs else 0.0,
            "steps": steps,
            # per grid shape: (n_calls, mean_us). Grid shape encodes batch
            # size, so comparing within a shape holds batch size constant.
            "by_grid": {g: (len(v), sum(v) / len(v)) for g, v in by_grid.items()},
        }
    return dict(sorted(out.items()))


def summarize(trace_dir):
    per = moe_per_rank(trace_dir)
    if not per:
        raise SystemExit(f"no rank traces found in {trace_dir}")
    ranks = list(per)

    means = [per[r]["mean_us"] for r in ranks]
    mu = sum(means) / len(means)

    # Grid-controlled variant: restrict to the grid shape with the most calls
    # that every rank shares. Removes batch-size mix as a confound between runs.
    shared = set.intersection(*[set(per[r]["by_grid"]) for r in ranks])
    shared.discard(None)
    dom, dom_means, dom_mu = None, None, None
    if shared:
        dom = max(shared, key=lambda g: per[ranks[0]]["by_grid"][g][0])
        dom_means = [per[r]["by_grid"][dom][1] for r in ranks]
        dom_mu = sum(dom_means) / len(dom_means)

    return {
        "trace_dir": trace_dir,
        "ranks": ranks,
        "per_rank_mean_us": means,
        "per_rank_durs": [per[r]['durs'] for r in ranks],
        "mean_over_ranks_us": mu,
        "max_over_mean": max(means) / mu if mu else 0.0,
        "max_over_min": max(means) / min(means) if min(means) else 0.0,
        "hottest_rank": ranks[means.index(max(means))],
        "total_over_ranks_ms": sum(per[r]["total_us"] for r in ranks) / 1000.0,
        "calls_per_rank": per[ranks[0]]["calls"],
        "steps": per[ranks[0]]["steps"],
        # grid-controlled
        "dominant_grid": dom,
        "dom_per_rank_mean_us": dom_means,
        "dom_mean_over_ranks_us": dom_mu,
        "dom_max_over_mean": (max(dom_means) / dom_mu) if dom_means else None,
    }
