###
# cli.py
#
# moe-reliability command line interface.
# Commands:
# init      write a commented configuration template
# validate  check a configuration and print the resolved run plan
# run       start a run from a configuration
# resume    continue an interrupted or partially failed run
# analyze   re-run trace analysis, HTA and figure rendering from a run
# grid      run every configuration of a parameter grid
# doctor    check the Ascend NPU environment
# list      list stored runs (same as moe-reliability-results list)
# show      show details of a run (same as moe-reliability-results show)
#
# Dylan Everingham
# 16.09.2026
###

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from moe_results import schema
from moe_results.store import default_results_dir

from . import __version__
from .config import TEMPLATES, ConfigError, ExperimentConfig, reference_markdown, render_template
from .logs import log, tee_output
from .runs import RunContext, RunError, make_run_id, utcnow

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_ENVIRONMENT = 3
EXIT_INTERRUPTED = 130

def _overrides(args: argparse.Namespace) -> list[str]:
    items = list(getattr(args, "set", None) or [])
    if getattr(args, "results_dir", None):
        items.append(f'output.results_dir="{args.results_dir}"')
    return items


def _print_plan(cfg: ExperimentConfig, run_id: str | None = None) -> None:
    from .pipelines import pipeline_for

    pipeline = pipeline_for(cfg.experiment_type)
    print(f"experiment:   {cfg.experiment_type}")
    if run_id:
        print(f"run id:       {run_id}")
    print(f"model:        {cfg.model.model_id} ({cfg.model.model_name})")
    devices = f" (devices {cfg.hardware.visible_devices})" if cfg.hardware.visible_devices else ""
    print(f"deployment:   {cfg.hardware.n_npus} x Ascend NPU{devices}, "
          f"batch size {cfg.server.batch_size}, max model len {cfg.server.max_model_len}, "
          f"expert parallel {cfg.server.enable_expert_parallel}, EPLB {cfg.server.enable_eplb}")
    print(f"results:      {Path(cfg.output.results_dir).resolve()}")
    print("stages:       " + " -> ".join(pipeline.STAGES))
    for line in pipeline.plan(cfg):
        print(f"  - {line}")


# Activate Ascend environment
def _environment(cfg: ExperimentConfig, require_devices: bool = True) -> dict[str, Any]:
    from .environment import collect_provenance, configure_environment

    runtime = configure_environment(cfg.environment, visible_devices=cfg.hardware.visible_devices,
                                    n_npus=cfg.hardware.n_npus, require_devices=require_devices)
    return collect_provenance(runtime)


def _execute(ctx: RunContext, cfg: ExperimentConfig, retry_failed: bool, log_file: bool, analyze_only: bool = False) -> int:
    from .pipelines import analyze, run_pipeline

    with tee_output(ctx.abspath(schema.LOG_FILE), enabled=log_file) as log_path:
        if log_path is not None and ctx.manifest.get("log_file") != schema.LOG_FILE:
            ctx.manifest["log_file"] = schema.LOG_FILE
            ctx.save()
        log(f"run {ctx.run_id} ({ctx.path})")
        try:
            status = analyze(ctx, cfg) if analyze_only else run_pipeline(ctx, cfg, retry_failed=retry_failed)
        except KeyboardInterrupt:
            log(f"interrupted - resume with: moe-reliability resume {ctx.run_id}")
            return EXIT_INTERRUPTED
        except Exception as exc:  # noqa: BLE001
            log(f"run failed: {exc!r}")
            log(f"details: {ctx.path / schema.MANIFEST_FILE}; resume with: moe-reliability resume {ctx.run_id}")
            return EXIT_FAILED
        log(f"run {ctx.run_id} finished with status '{status}'")
    return EXIT_OK if status == schema.STATUS_COMPLETED else EXIT_FAILED


# Commands

def cmd_init(args: argparse.Namespace) -> int:
    text = render_template(args.experiment)
    if args.output in (None, "-"):
        sys.stdout.write(text)
        return EXIT_OK
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"error: {out} exists (use --force to overwrite)", file=sys.stderr)
        return EXIT_USAGE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig.load(args.config, _overrides(args))
    print(f"{args.config}: valid")
    _print_plan(cfg, make_run_id(cfg))
    if args.show:
        print()
        print(cfg.to_toml(header="Resolved configuration"))
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig.load(args.config, _overrides(args))
    if args.dry_run:
        _print_plan(cfg, args.run_id or make_run_id(cfg))
        return EXIT_OK
    provenance = _environment(cfg)
    ctx = RunContext.create(cfg, run_id=args.run_id, environment=provenance)
    _print_plan(cfg, ctx.run_id)
    return _execute(ctx, cfg, retry_failed=False, log_file=not args.no_log_file)


def cmd_resume(args: argparse.Namespace) -> int:
    ctx = RunContext.open(args.run, args.results_dir or default_results_dir())
    cfg = ctx.config()
    if ctx.manifest.get("status") == schema.STATUS_COMPLETED and not args.retry_failed:
        print(f"run {ctx.run_id} is already completed")
        return EXIT_OK
    _environment(cfg)
    ctx.manifest.setdefault("resumed_at", []).append(utcnow())
    ctx.save()
    return _execute(ctx, cfg, retry_failed=args.retry_failed, log_file=not args.no_log_file)


def cmd_analyze(args: argparse.Namespace) -> int:
    ctx = RunContext.open(args.run, args.results_dir or default_results_dir())
    cfg = ctx.config()
    _environment(cfg, require_devices=False)  # post-processing does not use the NPUs
    return _execute(ctx, cfg, retry_failed=False, log_file=not args.no_log_file, analyze_only=True)


def cmd_grid(args: argparse.Namespace) -> int:
    from .grid import find_run_by_name, load_grid, workload_share_key
    from .pipelines import pipeline_for

    grid = load_grid(args.grid, _overrides(args))
    n_conf = grid.n_configurations()
    print(f"grid '{grid.name}': {grid.n_runs} runs, "
          f"{n_conf if n_conf is not None else 'unknown number of'} infrastructure configurations (sweep points)")

    if args.write_configs:
        outdir = Path(args.write_configs)
        outdir.mkdir(parents=True, exist_ok=True)
        for e in grid.entries:
            (outdir / f"{e.name}.toml").write_text(
                e.config.to_toml(header=f"Grid '{grid.name}' entry {e.index}: {e.assignments}"), encoding="utf-8")
        print(f"wrote {grid.n_runs} configurations to {outdir}")

    if args.dry_run or args.write_configs:
        for e in grid.entries:
            values = pipeline_for(e.config.experiment_type).sweep_values(e.config)
            n = len(values) if values is not None else "?"
            print(f"  {e.name}: {n} points  " + ", ".join(f"{k}={v}" for k, v in e.assignments.items()))
        return EXIT_OK

    exit_code = EXIT_OK
    shared_workloads: dict[tuple, str] = {}
    for e in grid.entries:
        cfg = e.config
        results_dir = cfg.output.results_dir
        print(f"\n=== grid entry {e.index + 1}/{grid.n_runs}: {e.name} {e.assignments}")
        existing = find_run_by_name(results_dir, cfg.experiment_type, e.name)

        key = workload_share_key(cfg) if grid.share_workloads else None
        if existing is not None:
            if key is not None and existing.manifest.get("workloads"):
                shared_workloads.setdefault(key, existing.id)
            if existing.status == schema.STATUS_COMPLETED and not args.retry_failed:
                print(f"already completed as run {existing.id} - skipping")
                continue
            ctx = RunContext.open(existing.path)
            run_cfg = ctx.config()
            print(f"resuming run {ctx.run_id}")
        else:
            if key is not None and key in shared_workloads:
                cfg = ExperimentConfig.from_dict(
                    cfg.to_dict(), [f'workloads.reuse_workloads_from="{shared_workloads[key]}"'])
                print(f"reusing workloads of run {shared_workloads[key]}")
            run_cfg = cfg
            ctx = None

        provenance = _environment(run_cfg)
        if ctx is None:
            ctx = RunContext.create(run_cfg, environment=provenance)
            ctx.manifest["grid"] = {"name": grid.name, "index": e.index, "assignments": e.assignments,
                                    "file": str(grid.path) if grid.path else None}
            ctx.save()
        code = _execute(ctx, run_cfg, retry_failed=args.retry_failed, log_file=not args.no_log_file)
        if key is not None and ctx.manifest.get("workloads"):
            shared_workloads.setdefault(key, ctx.run_id)
        if code == EXIT_INTERRUPTED:
            return code
        if code != EXIT_OK:
            exit_code = EXIT_FAILED
            if not args.continue_on_error:
                print("stopping grid (use --continue-on-error to run the remaining entries)")
                return exit_code
    return exit_code


def cmd_doctor(args: argparse.Namespace) -> int:
    from .environment import diagnose

    n_npus, visible = args.n_npus, args.visible_devices or ""
    if args.config:
        cfg = ExperimentConfig.load(args.config, _overrides(args))
        n_npus = n_npus or cfg.hardware.n_npus
        visible = visible or cfg.hardware.visible_devices
    results = diagnose(n_npus=n_npus, visible_devices=visible)
    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        detail = detail.replace("\n", "\n" + " " * (width + 11))
        print(f"  [{'ok' if ok else 'FAIL':>4}] {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return EXIT_OK if not failed else EXIT_ENVIRONMENT


def cmd_reference(args: argparse.Namespace) -> int:
    sys.stdout.write(reference_markdown())
    return EXIT_OK


def _delegate_results_cli(command: str, args: argparse.Namespace) -> int:
    from moe_results.cli import main as results_main

    argv = ["--results-dir", str(args.results_dir or default_results_dir()), command]
    if command == "show":
        argv.append(args.run)
    for f in args.filter or []:
        argv += ["--filter", f]
    return results_main(argv)


# Parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe-reliability",
        description="Run MoE inference reliability experiments on Ascend deployments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add_set(p: argparse.ArgumentParser) -> None:
        p.add_argument("--set", "-s", action="append", metavar="SECTION.KEY=VALUE",
                       help="override a configuration value (TOML syntax, repeatable)")

    def add_results_dir(p: argparse.ArgumentParser, help_text: str) -> None:
        p.add_argument("--results-dir", help=help_text)

    def add_log(p: argparse.ArgumentParser) -> None:
        p.add_argument("--no-log-file", action="store_true",
                       help=f"do not mirror console output into <run>/{schema.LOG_FILE}")

    p = sub.add_parser("init", help="write a commented configuration template")
    p.add_argument("experiment", choices=list(TEMPLATES), help="experiment type")
    p.add_argument("--output", "-o", help="output file (default: stdout)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("validate", help="validate a configuration and print the run plan")
    p.add_argument("config", help="TOML configuration file")
    add_set(p)
    add_results_dir(p, "override output.results_dir")
    p.add_argument("--show", action="store_true", help="also print the fully resolved configuration")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="start a run from a configuration")
    p.add_argument("config", help="TOML configuration file")
    add_set(p)
    add_results_dir(p, "override output.results_dir")
    p.add_argument("--run-id", help="explicit run id (default: timestamp_type_model_npus_batch[_name])")
    p.add_argument("--dry-run", action="store_true", help="validate and print the plan without running")
    add_log(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="continue an interrupted or partially failed run")
    p.add_argument("run", help="run id, unique run id prefix, or run directory")
    add_results_dir(p, "results directory (default: $MOE_RESULTS_DIR or ./results)")
    p.add_argument("--retry-failed", action="store_true", help="benchmark failed sweep points again")
    add_log(p)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("analyze", help="re-run trace analysis, HTA and figures of a run")
    p.add_argument("run", help="run id, unique run id prefix, or run directory")
    add_results_dir(p, "results directory (default: $MOE_RESULTS_DIR or ./results)")
    add_log(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("grid", help="run every configuration of a parameter grid")
    p.add_argument("grid", help="grid TOML file")
    add_set(p)
    add_results_dir(p, "override output.results_dir of every configuration")
    p.add_argument("--dry-run", action="store_true", help="list the expanded configurations without running")
    p.add_argument("--write-configs", metavar="DIR", help="write the expanded configurations to DIR and exit")
    p.add_argument("--continue-on-error", action="store_true", help="keep going when a run fails")
    p.add_argument("--retry-failed", action="store_true", help="re-benchmark failed points of existing runs")
    add_log(p)
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("doctor", help="check the Ascend NPU environment (CANN, torch_npu, NPUs, validated versions)")
    p.add_argument("config", nargs="?", help="optional configuration whose NPU settings are checked")
    add_set(p)
    p.add_argument("--n-npus", type=int, help="number of NPUs that must be visible")
    p.add_argument("--visible-devices", help="NPU ids to expose (ASCEND_RT_VISIBLE_DEVICES)")
    p.set_defaults(func=cmd_doctor, results_dir=None)

    p = sub.add_parser("list", help="list stored runs")
    add_results_dir(p, "results directory (default: $MOE_RESULTS_DIR or ./results)")
    p.add_argument("--filter", "-f", action="append", metavar="KEY=VALUE", help="filter runs (repeatable)")
    p.set_defaults(func=lambda a: _delegate_results_cli("list", a))

    p = sub.add_parser("show", help="show details of a run")
    p.add_argument("run", help="run id, unique run id prefix, or run directory")
    add_results_dir(p, "results directory (default: $MOE_RESULTS_DIR or ./results)")
    p.set_defaults(func=lambda a: _delegate_results_cli("show", a), filter=None)

    p = sub.add_parser("reference", help="print the configuration reference (Markdown)")
    p.set_defaults(func=cmd_reference)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .environment import AscendEnvironmentError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, RunError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except AscendEnvironmentError as exc:
        print(f"error: Ascend environment: {exc}\n(run `moe-reliability doctor` for a full report)", file=sys.stderr)
        return EXIT_ENVIRONMENT


if __name__ == "__main__":
    sys.exit(main())

