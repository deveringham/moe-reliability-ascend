###
# cli.py
#
# moe-reliability-results command line interface.
# Commands:
# list      list stored runs (same as moe-reliability list)
# show      show details of a run (same as moe-reliability show)
# summary   get summary table (one row per sweep point)
# requests  get per-request metrics
# plot      render standard figures for a run
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .store import ResultsStore, default_results_dir

_DEFAULT_SUMMARY_COLUMNS = [
    "run_id", "experiment", "model_name", "n_npus", "batch_size", "sweep_value", "point_status",
    "ttft_ms_mean", "tpot_ms_mean", "tpot_ms_p99", "trace_max_over_mean",
]


def _parse_filters(items: list[str] | None) -> dict:
    filters = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"invalid filter {item!r}; expected KEY=VALUE")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        filters[key.strip()] = value
    return filters


def _print_frame(df: pd.DataFrame) -> None:
    if df.empty:
        print("(no results)")
        return
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 200):
        print(df.to_string(index=False))


def cmd_list(args: argparse.Namespace) -> int:
    store = ResultsStore(args.results_dir)
    df = store.configurations(**_parse_filters(args.filter))
    if not df.empty:
        cols = [c for c in ["run_id", "experiment", "status", "created_at", "model_name", "n_npus",
                            "batch_size", "n_points"] if c in df.columns]
        df = df[cols]
    _print_frame(df)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run = ResultsStore(args.results_dir).get(args.run)
    m = run.manifest
    print(f"run:         {run.id}")
    print(f"experiment:  {run.experiment}")
    print(f"status:      {run.status}")
    print(f"created:     {run.created_at}")
    print(f"path:        {run.path}")
    print("stages:")
    for name, st in run.stages.items():
        err = f"  error: {st['error'].splitlines()[-1]}" if st.get("error") else ""
        print(f"  {name:<12} {st.get('status', '?'):<12}{err}")
    print("configuration:")
    for section, values in run.config.items():
        print(f"  [{section}]")
        for k, v in values.items():
            print(f"    {k} = {v!r}")
    print("points:")
    summary = run.summary()
    cols = [c for c in ["label", "point_status", "n_requests", "ttft_ms_mean", "tpot_ms_mean",
                        "tpot_ms_p99", "trace_max_over_mean", "workload_mae"] if c in summary.columns]
    _print_frame(summary[cols] if not summary.empty else summary)
    if m.get("figures"):
        print("figures:")
        for f in m["figures"]:
            print(f"  {f}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    store = ResultsStore(args.results_dir)
    filters = _parse_filters(args.filter)
    df = store.query(args.query, **filters) if args.query else store.summary(**filters)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".json":
            df.to_json(out, orient="records", indent=2)
        else:
            df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows to {out}")
        return 0
    if args.columns:
        cols = [c.strip() for c in args.columns.split(",")]
    elif args.all_columns:
        cols = list(df.columns)
    else:
        cols = [c for c in _DEFAULT_SUMMARY_COLUMNS if c in df.columns]
    _print_frame(df[cols] if not df.empty else df)
    return 0


def cmd_requests(args: argparse.Namespace) -> int:
    store = ResultsStore(args.results_dir)
    df = store.requests(**_parse_filters(args.filter))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        df.to_json(out, orient="records")
    else:
        df.to_csv(out, index=False)
    print(f"wrote {len(df)} request rows to {out}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from . import plots

    store = ResultsStore(args.results_dir)
    run = store.get(args.run)
    out = Path(args.output) if args.output else run.path / "figures"
    figs = plots.plot_run(run)
    written = plots.save_figures(figs, out, fmt=args.format)
    for p in written:
        print(p)
    if not written:
        print("no figures could be rendered for this run (no metrics stored yet)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moe-results", description="Browse, query and plot stored MoE experiment results.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--results-dir", default=str(default_results_dir()),
                        help="results directory (default: $MOE_RESULTS_DIR or ./results)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list runs")
    p.add_argument("--filter", "-f", action="append", metavar="KEY=VALUE", help="filter runs (repeatable)")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show run details")
    p.add_argument("run", help="run id, unique id prefix, or run directory")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("summary", help="summary table (one row per sweep point)")
    p.add_argument("--filter", "-f", action="append", metavar="KEY=VALUE", help="filter runs (repeatable)")
    p.add_argument("--query", "-q", help="pandas query expression, e.g. \"tpot_ms_mean > 40\"")
    p.add_argument("--columns", "-c", help="comma-separated columns to print")
    p.add_argument("--all-columns", action="store_true", help="print all columns")
    p.add_argument("--output", "-o", help="write the table to a .csv or .json file instead of printing")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("requests", help="export per-request measurements")
    p.add_argument("--filter", "-f", action="append", metavar="KEY=VALUE", help="filter runs (repeatable)")
    p.add_argument("--output", "-o", required=True, help=".csv or .json output file")
    p.set_defaults(func=cmd_requests)

    p = sub.add_parser("plot", help="render the standard figures of a run")
    p.add_argument("run", help="run id, unique id prefix, or run directory")
    p.add_argument("--output", "-o", help="output directory (default: <run>/figures)")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    p.set_defaults(func=cmd_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
