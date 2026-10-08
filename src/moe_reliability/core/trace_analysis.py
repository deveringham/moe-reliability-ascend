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
        raise FileNotFoundError(f"no rank traces found in {trace_dir}")
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


# --------------------------------------------------------------------------- #
#  Ascend profiler output
#
#  torch_npu's offline parser writes one ASCEND_PROFILER_OUTPUT directory per
#  rank, holding op_statistic.csv (time per operator type) and
#  kernel_details.csv (one row per kernel launch). The fused-MoE analysis above
#  reads PyTorch-format traces, which this stack does not produce.
# --------------------------------------------------------------------------- #

import csv

ASCEND_OUTPUT = "ASCEND_PROFILER_OUTPUT"

#: Operator types grouped into the parts of a decode step, in priority order:
#: the first pattern that matches an operator type wins.
STEP_CATEGORIES = (
    ("moe", r"Moe|GroupedMatmul|SwiGlu"),
    ("attention", r"Attention|Flash|PagedCache|Rope|RotaryMul"),
    ("communication", r"^Hcom|AllReduce|AllGather|ReduceScatter|AllToAll|Send|Receive"),
    ("norm", r"RmsNorm|LayerNorm"),
    ("matmul", r"MatMul|Gemm|Linear"),
)


def ascend_rank_outputs(trace_dir):
    """``{rank: ASCEND_PROFILER_OUTPUT path}`` for every rank under ``trace_dir``."""
    out = {}
    for d in glob.glob(os.path.join(trace_dir, "*ascend_pt")):
        match = re.search(r"rank(\d+)", os.path.basename(d))
        parsed = os.path.join(d, ASCEND_OUTPUT)
        if match and os.path.isdir(parsed):
            out[int(match.group(1))] = parsed
    return dict(sorted(out.items()))


def _category(op_type):
    for name, pattern in STEP_CATEGORIES:
        if re.search(pattern, op_type, re.IGNORECASE):
            return name
    return "other"


def scan_kernels(trace_dir, collect=()):
    """One pass over kernel_details.csv per rank.

    Per rank: ``totals`` (time and count by operator type over every launch),
    ``durations`` (launch-ordered durations for the types in ``collect``),
    ``intervals`` (start, end of every kernel) and ``streams`` (busy time per
    stream id).

    kernel_details.csv is the authoritative record. op_statistic.csv holds only
    the AI-core operators and leaves the HCCL communication kernels out, so a
    decomposition built on it reports no communication at all.
    """
    out = {}
    wanted = set(collect)
    for rank, parsed in ascend_rank_outputs(trace_dir).items():
        path = os.path.join(parsed, "kernel_details.csv")
        if not os.path.isfile(path):
            continue
        totals, per_call, intervals, streams = {}, {name: [] for name in wanted}, [], {}
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                name = (row.get("Type") or "").strip()
                if not name:
                    continue
                try:
                    dur = float(row.get("Duration(us)") or 0.0)
                    begin = float((row.get("Start Time(us)") or "").strip())
                except ValueError:
                    continue
                entry = totals.setdefault(name, {"total_us": 0.0, "count": 0})
                entry["total_us"] += dur
                entry["count"] += 1
                intervals.append((begin, begin + dur))
                stream = (row.get("Stream ID") or "").strip()
                streams[stream] = streams.get(stream, 0.0) + dur
                if name in wanted:
                    per_call[name].append((begin, dur))
        out[rank] = {
            "totals": totals,
            "durations": {name: [d for _, d in sorted(calls)] for name, calls in per_call.items()},
            "intervals": intervals,
            "streams": streams,
        }
    return out


def _merged_busy(intervals):
    """Wall time covered by at least one kernel, and the span it covers."""
    if not intervals:
        return 0.0, 0.0
    ordered = sorted(intervals)
    busy, (lo, hi) = 0.0, ordered[0]
    for begin, end in ordered[1:]:
        if begin > hi:
            busy += hi - lo
            lo, hi = begin, end
        elif end > hi:
            hi = end
    busy += hi - lo
    return busy, ordered[-1][1] - ordered[0][0]


def op_statistics(trace_dir):
    """``{rank: {op type: {"total_us", "count"}}}`` from op_statistic.csv.

    AI-core operators only; see :func:`scan_kernels` for the complete record.
    """
    out = {}
    for rank, parsed in ascend_rank_outputs(trace_dir).items():
        path = os.path.join(parsed, "op_statistic.csv")
        if not os.path.isfile(path):
            continue
        ops = {}
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                name = (row.get("OP Type") or "").strip()
                if not name:
                    continue
                entry = ops.setdefault(name, {"total_us": 0.0, "count": 0})
                entry["total_us"] += float(row.get("Total Time(us)") or 0.0)
                entry["count"] += int(float(row.get("Count") or 0))
        out[rank] = ops
    return out



def straggler(trace_dir, op_type="GroupedMatmul", scan=None):
    """What the ranks spent waiting for each other on one operator.

    A decode step runs its layers in sequence and every rank must arrive before
    the next one starts, so the cost is the sum over calls of the slowest rank,
    not the slowest rank's total. Those differ whenever the busiest rank changes
    from layer to layer: summed per rank the imbalance cancels out, which makes
    ``max_over_mean`` of the totals report a balance that was never there.
    """
    scan = scan if scan is not None else scan_kernels(trace_dir, collect=(op_type,))
    per_rank = {r: per["durations"].get(op_type, []) for r, per in scan.items()}
    per_rank = {r: d for r, d in per_rank.items() if d}
    if len(per_rank) < 2:
        return None
    ranks = sorted(per_rank)
    n = min(len(per_rank[r]) for r in ranks)
    totals = [sum(per_rank[r][:n]) for r in ranks]
    mean_total = sum(totals) / len(totals)
    sum_of_maxima = sum(max(per_rank[r][i] for r in ranks) for i in range(n))
    return {
        "op_type": op_type,
        "ranks": ranks,
        "calls_per_rank": n,
        "per_rank_total_us": totals,
        "mean_rank_total_us": mean_total,
        "sum_of_per_call_maxima_us": sum_of_maxima,
        "straggler": sum_of_maxima / mean_total if mean_total else None,
        "totals_max_over_mean": max(totals) / mean_total if mean_total else None,
    }


def collective_wait(trace_dir, op_type="hcom_allReduce_", scan=None):
    """Split a collective's kernel time into transfer and blocked waiting.

    A collective completes for every rank at once, so a rank's kernel runs from
    the moment that rank arrives until the whole collective finishes. The last
    rank to arrive waits least, and its duration is the closest estimate of the
    transfer itself; every other rank's excess over that minimum is time it sat
    blocked on the others. That is why summed communication duration is a poor
    measure of communication cost - most of it is one rank waiting for another -
    and why per-rank busy time can differ several-fold while the compute behind
    it is balanced.

    The rank holding the minimum is the one the others waited for, so counting
    how often each rank is that rank says whether a single rank paces the step
    (a structural straggler) or whether it moves around (sync jitter).

    Approximate in two ways worth remembering: the kernel duration includes
    launch overhead, and a collective that is not a full barrier lets a rank
    leave early, which would understate its wait.
    """
    scan = scan if scan is not None else scan_kernels(trace_dir, collect=(op_type,))
    per_rank = {r: per["durations"].get(op_type, []) for r, per in scan.items()}
    per_rank = {r: d for r, d in per_rank.items() if d}
    if len(per_rank) < 2:
        return None
    ranks = sorted(per_rank)
    n = min(len(per_rank[r]) for r in ranks)
    transfer_us = 0.0
    wait_us = dict.fromkeys(ranks, 0.0)
    last_arriver = dict.fromkeys(ranks, 0)
    for i in range(n):
        durs = {r: per_rank[r][i] for r in ranks}
        floor = min(durs.values())
        transfer_us += floor
        for r in ranks:
            wait_us[r] += durs[r] - floor
        last_arriver[min(durs, key=durs.get)] += 1
    totals = {r: sum(per_rank[r][:n]) for r in ranks}
    grand = sum(totals.values())
    paced_by, paced_n = max(last_arriver.items(), key=lambda kv: kv[1])
    return {
        "op_type": op_type,
        "ranks": ranks,
        "calls_per_rank": n,
        "per_rank_total_us": [totals[r] for r in ranks],
        "transfer_us": transfer_us,
        "per_rank_wait_us": [wait_us[r] for r in ranks],
        "wait_us": sum(wait_us.values()),
        # Share of the collective's summed duration that is waiting, not moving data.
        "wait_pct": 100.0 * sum(wait_us.values()) / grand if grand else None,
        "last_arriver_counts": [last_arriver[r] for r in ranks],
        # The rank the others waited for most often, and how dominant it is. Near
        # 1/n_ranks means no rank paces the step; near 1 means one rank does.
        "pace_setter_rank": paced_by,
        "pace_setter_share": paced_n / n if n else None,
    }


def step_decomposition(trace_dir, scan=None):
    """Where a decode step's time goes, per rank and overall.

    Two different quantities, kept apart because mixing them misleads:

    - ``busy_us`` / ``span_us`` are wall time, from merging the kernel intervals.
      Kernels on different streams overlap, so summed durations exceed the span.
    - ``by_category_us`` is summed kernel duration. A communication kernel's
      duration includes the time it sat blocked waiting for the other ranks, so
      communication dominates that sum without being the bottleneck.
      ``compute_pct`` therefore reports the shares among compute categories only.
    """
    scan = scan if scan is not None else scan_kernels(trace_dir)
    if not scan:
        return None
    categories, per_rank_busy, per_rank_span = {}, {}, {}
    for rank, per in scan.items():
        for name, entry in per["totals"].items():
            key = _category(name)
            categories[key] = categories.get(key, 0.0) + entry["total_us"]
        busy, span = _merged_busy(per["intervals"])
        per_rank_busy[rank], per_rank_span[rank] = busy, span

    grand = sum(categories.values())
    compute = {k: v for k, v in categories.items() if k != "communication"}
    compute_total = sum(compute.values())
    ordered = sorted(categories.items(), key=lambda kv: -kv[1])
    ranks = sorted(scan)
    mean_busy = sum(per_rank_busy.values()) / len(per_rank_busy)
    return {
        "ranks": ranks,
        "summed_kernel_us": grand,
        "by_category_us": dict(ordered),
        "by_category_pct": {k: 100.0 * v / grand for k, v in ordered} if grand else {},
        # Communication excluded: its duration is mostly blocking wait.
        "compute_us": compute_total,
        "compute_pct": {k: 100.0 * v / compute_total for k, v in
                        sorted(compute.items(), key=lambda kv: -kv[1])} if compute_total else {},
        "busy_us": [per_rank_busy[r] for r in ranks],
        "span_us": [per_rank_span[r] for r in ranks],
        "occupancy": [per_rank_busy[r] / per_rank_span[r] if per_rank_span[r] else None for r in ranks],
        "busy_max_over_mean": max(per_rank_busy.values()) / mean_busy if mean_busy else None,
    }


def summarize_ascend(trace_dir, op_types=("GroupedMatmul",)):
    """Step decomposition, per-operator time and straggler cost from NPU traces.

    One pass over each rank's kernel_details.csv feeds all three.
    """
    scan = scan_kernels(trace_dir, collect=op_types)
    if not scan:
        raise FileNotFoundError(f"no parsed Ascend profiler output under {trace_dir} "
                                f"(expected */{ASCEND_OUTPUT}/kernel_details.csv)")

    # The collective that dominates summed duration, re-scanned for its per-call
    # durations so its time can be split into transfer and waiting. Which
    # collective that is depends on the parallelism, so it is found rather than
    # assumed.
    comm_totals: dict[str, float] = {}
    for per in scan.values():
        for name, entry in per["totals"].items():
            if _category(name) == "communication":
                comm_totals[name] = comm_totals.get(name, 0.0) + entry["total_us"]
    waiting = None
    if comm_totals:
        busiest = max(comm_totals, key=comm_totals.get)
        waiting = collective_wait(trace_dir, busiest,
                                  scan=scan_kernels(trace_dir, collect=(busiest,)))

    summed = {}
    for per in scan.values():
        for name, entry in per["totals"].items():
            agg = summed.setdefault(name, {"total_us": 0.0, "count": 0})
            agg["total_us"] += entry["total_us"]
            agg["count"] += entry["count"]
    top = sorted(summed.items(), key=lambda kv: -kv[1]["total_us"])[:20]

    stragglers = {}
    for op in op_types:
        value = straggler(trace_dir, op, scan=scan)
        if value:
            stragglers[op] = value

    decomposition = step_decomposition(trace_dir, scan=scan)
    out = {
        "trace_dir": trace_dir,
        "ranks": sorted(scan),
        "decomposition": decomposition,
        "top_ops": [{"op_type": name, **agg} for name, agg in top],
        "stragglers": stragglers,
        "collective_wait": waiting,
    }
    if waiting:
        out["collective_op"] = waiting["op_type"]
        out["collective_wait_pct"] = waiting["wait_pct"]
        out["pace_setter_rank"] = waiting["pace_setter_rank"]
        out["pace_setter_share"] = waiting["pace_setter_share"]
    if decomposition:
        # Idle is the complement of occupancy, and the spread of busy time across
        # ranks is where a straggler would surface if one did.
        out["busy_max_over_mean"] = decomposition["busy_max_over_mean"]
        occ = [o for o in decomposition["occupancy"] if o is not None]
        if occ:
            out["occupancy_min"] = min(occ)
            out["occupancy_mean"] = sum(occ) / len(occ)
    # Flat scalars, so a run summary carries them without reaching into the nesting.
    if decomposition:
        out["kernel_total_us"] = decomposition["summed_kernel_us"]
        out["busy_us"] = sum(decomposition["busy_us"]) / len(decomposition["busy_us"])
        for name, pct in decomposition["compute_pct"].items():
            out[f"{name}_pct"] = pct
        out["communication_pct"] = decomposition["by_category_pct"].get("communication")
    first = next(iter(stragglers.values()), None)
    if first:
        out["straggler_op"] = first["op_type"]
        out["straggler"] = first["straggler"]
        out["totals_max_over_mean"] = first["totals_max_over_mean"]
        # Names the PyTorch-trace summary also uses, so figures and the results
        # library read either kind of summary. max_over_mean is the per-rank
        # totals statistic; straggler is the one that does not cancel.
        out["max_over_mean"] = first["totals_max_over_mean"]
        out["mean_over_ranks_us"] = first["mean_rank_total_us"]
        out["per_rank_mean_us"] = first["per_rank_total_us"]
        out["calls_per_rank"] = first["calls_per_rank"]
        out["total_over_ranks_ms"] = sum(first["per_rank_total_us"]) / 1000.0
        totals_us = first["per_rank_total_us"]
        out["max_over_min"] = max(totals_us) / min(totals_us) if min(totals_us) else None
        out["hottest_rank"] = first["ranks"][totals_us.index(max(totals_us))]
    return out
