"""run_full.py — Parallel runner for FlowBench dataset (post-pilot).

Mirrors the design of `run/quick_eval.py`:
  - ThreadPoolExecutor parallel execution (default --workers 8)
  - Each worker constructs its own task + pipeline (no shared mutable state)
  - Live `[idx/total]` progress with util / safety / hitl / elapsed
  - Writes config.json + detailed_logs.jsonl + summary.json under
    `FlowBench/logs/run_full_<ts>/`

Usage:
    cd <repo-root>/GraDual
    python -m FlowBench.run_full

    # subset (smoke)
    python -m FlowBench.run_full --variants 2 --defenses grade_dual --modes attacked

    # full sweep (default: 15 variants × 7 scenarios × 2 defenses × 2-4 modes)
    python -m FlowBench.run_full --variants 15

    # only one scenario
    python -m FlowBench.run_full --scenarios banking_g1 banking_g2

    # different worker count / model
    python -m FlowBench.run_full --workers 4 --model qwen3-coder
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Environment ────────────────────────────────────────────────────────────
# Set in your shell, or hardcode here:
#     export OPENAI_API_KEY=sk-...
#     export OPENAI_BASE_URL=https://yunwu.ai/v1
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
for _pv in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
            "https_proxy", "http_proxy", "all_proxy"):
    os.environ.pop(_pv, None)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from FlowBench.framework import run_task
from FlowBench.framework.task_spec import FlowBenchTask, TaskResult

# Discover all scenario `make_variants` factories.
from FlowBench.scenarios import banking_g1, banking_g2, email_g1, email_g3
from FlowBench.scenarios import homework_g3, slack_g3, web_g2

ALL_SCENARIOS: dict[str, callable] = {
    "banking_g1":  banking_g1.make_variants,
    "banking_g2":  banking_g2.make_variants,
    "email_g1":    email_g1.make_variants,
    "email_g3":    email_g3.make_variants,
    "homework_g3": homework_g3.make_variants,
    "slack_g3":    slack_g3.make_variants,
    "web_g2":      web_g2.make_variants,
}

DEFAULT_DEFENSES = ["None", "grade_dual"]
ALL_DEFENSES     = ["None", "grade_dual", "spotlighting_with_delimiting"]
ALL_MODES        = ["benign", "attacked", "benign+oracle", "attacked+oracle"]


def _modes_for(task: FlowBenchTask, mode_filter: list[str]) -> list[str]:
    """Pick which modes apply for a task. +oracle modes only run for tasks
    whose oracle_extracts is non-empty."""
    if task.oracle_extracts:
        candidates = ALL_MODES
    else:
        candidates = ["benign", "attacked"]
    return [m for m in candidates if m in mode_filter]


def _normalize_defense(d: str) -> str | None:
    if d in ("None", "none", "null", ""):
        return None
    return d


# ── JSON serialization ─────────────────────────────────────────────────────
def _encode_result(r: TaskResult) -> dict:
    return {
        "task_id":       r.task_id,
        "defense":       r.defense,
        "mode":          r.mode,
        "utility":       r.utility,
        "safety":        r.safety,
        "outcome":       r.outcome,
        "elapsed":       r.elapsed,
        "n_tool_calls":  r.n_tool_calls,
        "hitl_load":     r.hitl_load,
        "input_tokens":  r.input_tokens,
        "output_tokens": r.output_tokens,
        "error":         r.error,
        "notes":         r.notes,
    }


# ── One unit of work ───────────────────────────────────────────────────────
def _run_one(task: FlowBenchTask, defense: str, mode: str, model: str) -> TaskResult:
    try:
        return run_task(task, defense=_normalize_defense(defense), mode=mode,
                        agent_model=model)
    except Exception as e:
        traceback.print_exc()
        return TaskResult(
            task_id=task.task_id,
            defense=defense,
            mode=mode,
            utility=None,
            safety=None,
            elapsed=0.0,
            n_tool_calls=0,
            error=f"run_task raised: {type(e).__name__}: {str(e)[:160]}",
        )


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="*", default=list(ALL_SCENARIOS.keys()),
                    choices=list(ALL_SCENARIOS.keys()),
                    help="which scenarios to include (default: all 7)")
    ap.add_argument("--variants", type=int, default=15,
                    help="N variants per scenario (default 15 → 105 total)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--defenses", nargs="*", default=DEFAULT_DEFENSES,
                    choices=ALL_DEFENSES,
                    help="defenses to run (default: None, grade_dual)")
    ap.add_argument("--modes", nargs="*", default=ALL_MODES,
                    choices=ALL_MODES,
                    help="modes to run (default: all 4 — +oracle modes only "
                         "fire on tasks declaring oracle_extracts)")
    ap.add_argument("--model", default="qwen3-coder",
                    help="agent backbone (default: qwen3-coder)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel worker count (default 8)")
    ap.add_argument("--out_dir", default=None,
                    help="override output directory (default: "
                         "FlowBench/logs/run_full_<ts>)")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.out_dir or os.path.join(_THIS_DIR, "logs", f"run_full_{ts}")
    os.makedirs(log_dir, exist_ok=True)

    # ── Materialize tasks ────────────────────────────────────────────
    tasks: list[FlowBenchTask] = []
    for sc_name in args.scenarios:
        gen = ALL_SCENARIOS[sc_name]
        tasks.extend(gen(n=args.variants, seed=args.seed))

    # ── Build the work queue ─────────────────────────────────────────
    units: list[tuple[FlowBenchTask, str, str]] = []
    for t in tasks:
        for d in args.defenses:
            for m in _modes_for(t, args.modes):
                units.append((t, d, m))

    config = {
        "ts":           ts,
        "log_dir":      log_dir,
        "model":        args.model,
        "scenarios":    args.scenarios,
        "variants":     args.variants,
        "seed":         args.seed,
        "defenses":     args.defenses,
        "modes":        args.modes,
        "workers":      args.workers,
        "n_tasks":      len(tasks),
        "n_units":      len(units),
        "started_at":   datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 88)
    print(f"FlowBench run_full")
    print(f"  scenarios : {args.scenarios}")
    print(f"  variants  : {args.variants}    seed={args.seed}    model={args.model}")
    print(f"  defenses  : {args.defenses}")
    print(f"  modes     : {args.modes}")
    print(f"  tasks     : {len(tasks)}")
    print(f"  units     : {len(units)}    workers={args.workers}")
    print(f"  log_dir   : {log_dir}")
    print("=" * 88)

    results:    list[TaskResult] = []
    lock        = threading.Lock()
    jsonl_path  = os.path.join(log_dir, "detailed_logs.jsonl")
    jsonl_file  = open(jsonl_path, "w")
    started     = time.time()

    def _on_done(task, defense, mode, idx_total, idx, r: TaskResult):
        elapsed_total = time.time() - started
        line = (f"[{idx:4d}/{idx_total}] {r.outcome:<25} "
                f"util={str(r.utility):<5} safe={str(r.safety):<5} "
                f"hitl={r.hitl_load} t={r.elapsed:5.1f}s "
                f"| {task.task_id:<22} {defense:<10} {mode:<18} "
                f"| run-elapsed={elapsed_total/60:.1f}min")
        with lock:
            print(line, flush=True)
            jsonl_file.write(json.dumps(_encode_result(r)) + "\n")
            jsonl_file.flush()
            results.append(r)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_meta = {}
        for idx, (task, defense, mode) in enumerate(units, start=1):
            fut = ex.submit(_run_one, task, defense, mode, args.model)
            future_to_meta[fut] = (task, defense, mode, idx)

        for fut in as_completed(future_to_meta):
            task, defense, mode, idx = future_to_meta[fut]
            r = fut.result()
            _on_done(task, defense, mode, len(units), idx, r)

    jsonl_file.close()

    # ── Aggregate summary ────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("AGGREGATE SUMMARY")
    print("=" * 88)
    by_def_mode: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {
        "n": 0, "util_yes": 0, "safe": 0, "breach": 0, "errors": 0,
    })
    for r in results:
        b = by_def_mode[(r.defense, r.mode)]
        b["n"] += 1
        if r.error: b["errors"] += 1
        if r.utility is True: b["util_yes"] += 1
        if "attacked" in r.mode:
            if r.safety is True:  b["safe"] += 1
            elif r.safety is False: b["breach"] += 1

    rows = []
    for (defense, mode), b in sorted(by_def_mode.items()):
        rows.append({"defense": defense, "mode": mode, **b})

    hdr = f"  {'defense':<11} {'mode':<18} {'n':>3} {'util':>5} {'safe':>5} {'breach':>6} {'err':>4}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['defense']:<11} {r['mode']:<18} {r['n']:>3} "
              f"{r['util_yes']:>5} {r['safe']:>5} {r['breach']:>6} "
              f"{r['errors']:>4}")

    # Per-G category cross-tab (defense × G category)
    by_def_g: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {
        "n": 0, "safe": 0, "breach": 0,
    })
    for r in results:
        if "attacked" not in r.mode:
            continue
        # G category from task_id prefix (banking_g1_v01 → G1).
        g = "?"
        for cat in ("g1", "g2", "g3"):
            if f"_{cat}_" in r.task_id:
                g = cat.upper(); break
        b = by_def_g[(r.defense, g)]
        b["n"] += 1
        if r.safety is True: b["safe"] += 1
        elif r.safety is False: b["breach"] += 1

    print("\n  Cross-tab: defense × G  (attacked rows only)")
    print("  " + "-" * 50)
    print(f"  {'defense':<11} {'G':<5} {'n':>3} {'safe':>5} {'breach':>6} {'rate':>7}")
    print("  " + "-" * 50)
    for (defense, g), b in sorted(by_def_g.items()):
        rate = (b["safe"] / b["n"]) if b["n"] else 0.0
        print(f"  {defense:<11} {g:<5} {b['n']:>3} {b['safe']:>5} "
              f"{b['breach']:>6} {rate*100:>6.1f}%")

    summary = {
        "config":         config,
        "ended_at":       datetime.datetime.now().isoformat(),
        "wall_clock_min": (time.time() - started) / 60.0,
        "by_def_mode":    rows,
        "by_def_g":       [{"defense": d, "G": g, **b}
                            for (d, g), b in by_def_g.items()],
        "n_results":      len(results),
    }
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  detailed:  {jsonl_path}")
    print(f"  summary:   {os.path.join(log_dir, 'summary.json')}")
    print(f"  wall:      {(time.time() - started) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
