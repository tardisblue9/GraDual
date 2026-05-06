"""
quick_eval.py — Parallel quick evaluation for IPIGuard defenses.

Randomly samples a percentage of (user_task, injection_task) pairs from each suite
(default 20%), runs them in parallel through specified defense(s), and produces
a summary report.

Usage:
    cd <repo-root>

    # Default: 20% sample, all suites, under_attack, grade_dual, 8 workers
    python run/quick_eval.py

    # Specific suites and defenses
    python run/quick_eval.py --suites travel slack --defenses grade_dual ipiguard None --sample_pct 0.2

    # Benign mode
    python run/quick_eval.py --mode benign --defenses grade_dual

    # Adjust parallelism
    python run/quick_eval.py --workers 4

    # Full run (100%)
    python run/quick_eval.py --sample_pct 1.0 --suites travel
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Environment setup ─────────────────────────────────────────────────────────
# Configure your OpenAI-compatible endpoint here, or set them in your shell.
# Example for qwen3-coder via yunwu.ai (an OpenAI-compatible proxy):
#     export OPENAI_API_KEY=sk-...
#     export OPENAI_BASE_URL=https://yunwu.ai/v1
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
for _pv in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
            "https_proxy", "http_proxy", "all_proxy"):
    os.environ.pop(_pv, None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from agentdojo.task_suite.load_suites import get_suite
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.endorsement_oracle import make_gt_oracle
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.models import ModelsEnum
from agentdojo.functions_runtime import FunctionCall
from data_module import initialize_dataset


# ── JSON Encoder for FunctionCall objects ────────────────────────────────────
class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, FunctionCall):
            return obj.model_dump()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


def _msgs_to_json(messages) -> list:
    out = []
    for m in messages:
        entry = {"role": m.get("role", "?"), "content": m.get("content") or ""}
        tcs = m.get("tool_calls") or []
        if tcs:
            entry["tool_calls"] = [
                {"function": tc.function, "args": tc.args, "id": tc.id}
                for tc in tcs
            ]
        if m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]
        out.append(entry)
    return out


# ── Per-case worker (thread-safe; builds its own pipeline) ───────────────────
def _run_case(
    idx: int,
    total: int,
    suite_name: str,
    defense_name: str,
    uid: int,
    iid: int | None,
    mode: str,
    model_name: str,
    attack_name: str,
    benchmark_version: str,
    log_dir: str,
    log_lock: threading.Lock,
) -> dict:
    """Run one (suite, defense, uid, iid?) case. Returns a result dict.

    Each worker constructs its own suite + pipeline so threads don't share
    mutable LLM-client state. Slight overhead per call but clean isolation
    (pipeline construction is dominated by graph-build LLM call later, not
    by the from_config() call itself).
    """
    t0 = time.time()
    label = f"u{uid}+i{iid}" if iid is not None else f"u{uid}"
    short = f"{suite_name}/{label} [{defense_name}]"

    try:
        suite = get_suite(benchmark_version, suite_name)
        # Match single_task_suite.py: register utility tools so grade_dual etc.
        # have access to calculate / table_query.
        try:
            from agentdojo.agent_pipeline.utility_tools import calculate, table_query
            from agentdojo.functions_runtime import make_function
            for tool in (make_function(calculate), make_function(table_query)):
                if tool not in suite.tools:
                    suite.tools = list(suite.tools) + [tool]
        except Exception:
            pass

        defense = None if defense_name == "None" else defense_name
        pipeline = AgentPipeline.from_config(PipelineConfig(
            llm=ModelsEnum(model_name),
            defense=defense,
            system_message_name=None,
            system_message=None,
        ))

        user_task = suite.get_user_task_by_id(f"user_task_{uid}")

        if mode == "under_attack" and iid is not None:
            injection_task = suite.get_injection_task_by_id(f"injection_task_{iid}")
            attacker = load_attack(attack_name, suite, pipeline)
            injections = attacker.attack(user_task, injection_task)
        else:
            injection_task = None
            injections = {k: "" for k in suite.get_injection_vector_defaults().keys()}

        extra_args = {"input_tokens": 0, "output_tokens": 0}
        # Plan C v7: register the GT-aware endorsement oracle. The pipeline's
        # request_endorsement short-circuit calls this callable to make the
        # binary allow/deny decision. None-valued in benign mode is fine —
        # make_gt_oracle handles that case (no contamination check; only
        # delegation-phrase check).
        extra_args["endorsement_oracle"] = make_gt_oracle(
            user_task=user_task,
            injection_task=injection_task,
            injections=injections,
        )
        utility, security, messages, token_info = suite.run_task_with_pipeline_msg(
            agent_pipeline=pipeline,
            user_task=user_task,
            injection_task=injection_task,
            injections=injections,
            extra_args=extra_args,
        )
        elapsed = time.time() - t0

        # Pull HITL accounting added by the endorsement protocol.
        hitl_load = int(extra_args.get("hitl_load", 0))
        endorsement_log = list(extra_args.get("endorsement_log", []))

        u_str = "YES" if utility else "NO "
        # Note: AgentDojo semantic — security=True means injection succeeded → UNSAFE.
        s_str = "—   " if mode == "benign" else ("SAFE  " if not security else "UNSAFE")
        h_str = f"H{hitl_load}" if hitl_load else "H0"
        print(f"[{idx:3d}/{total}] {u_str} | {s_str} | {h_str} | "
              f"{elapsed:5.1f}s | {short}")

        result = {
            "idx": idx,
            "suite": suite_name,
            "defense": defense_name,
            "user_task_id": uid,
            "injection_task_id": iid,
            "user_task_prompt": user_task.PROMPT[:200],
            "mode": mode,
            "utility": bool(utility),
            "security": bool(security),
            "hitl_load": hitl_load,
            "endorsement_log": endorsement_log,
            "elapsed": round(elapsed, 1),
            "input_tokens": token_info.get("input_tokens", 0),
            "output_tokens": token_info.get("output_tokens", 0),
            "error": None,
        }

        # Persist per-case messages (under lock, append-only JSONL)
        log_entry = {**result, "messages": _msgs_to_json(messages)}
        with log_lock:
            with open(os.path.join(log_dir, "detailed_logs.jsonl"), "a") as f:
                json.dump(log_entry, f, cls=_Encoder, ensure_ascii=False)
                f.write("\n")

        return result

    except Exception as e:
        elapsed = time.time() - t0
        msg = repr(e)[:120]
        print(f"[{idx:3d}/{total}] ERROR {short}: {msg}")
        result = {
            "idx": idx,
            "suite": suite_name,
            "defense": defense_name,
            "user_task_id": uid,
            "injection_task_id": iid,
            "user_task_prompt": "",
            "mode": mode,
            "utility": False,
            "security": True,  # conservative — treat as breach on error so it stands out
            "hitl_load": 0,
            "endorsement_log": [],
            "elapsed": round(elapsed, 1),
            "input_tokens": 0,
            "output_tokens": 0,
            "error": msg,
        }
        with log_lock:
            with open(os.path.join(log_dir, "detailed_logs.jsonl"), "a") as f:
                json.dump({**result, "messages": []}, f, cls=_Encoder, ensure_ascii=False)
                f.write("\n")
        return result


def main():
    parser = argparse.ArgumentParser(description="Parallel quick eval for IPIGuard defenses")
    parser.add_argument("--suites", nargs="+",
                        default=["travel", "slack", "banking", "workspace"],
                        help="Suites to evaluate (default: all four)")
    parser.add_argument("--defenses", nargs="+", default=["grade_dual"],
                        help="Defense names (e.g. grade_dual ipiguard None)")
    parser.add_argument("--model", default="qwen3-coder", help="Agent model name")
    parser.add_argument("--attack", default="important_instructions", help="Attack name")
    parser.add_argument("--mode", default="under_attack",
                        choices=["under_attack", "benign"], help="Evaluation mode")
    parser.add_argument("--sample_pct", type=float, default=0.2,
                        help="Fraction of cases to sample (0.0-1.0, default 0.2 = 20%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--benchmark_version", default="v1.1.2", help="Benchmark version")
    parser.add_argument("--workers", type=int, default=8,
                        help="ThreadPool max workers (default: 8)")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir (default: auto under evaluation_results/)")
    args = parser.parse_args()

    # ── Setup output ──────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "evaluation_results",
            f"quick_eval_parallel_{args.mode}_{'+'.join(args.defenses)}_{ts}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    config = vars(args).copy()
    config["timestamp"] = ts
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Quick Eval (PARALLEL): {args.mode} | model={args.model} | "
          f"attack={args.attack}")
    print(f"  Suites: {args.suites} | Defenses: {args.defenses}")
    print(f"  Sample: {args.sample_pct*100:.0f}% | Seed: {args.seed} | "
          f"Workers: {args.workers}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}\n")

    random.seed(args.seed)

    # ── Sample cases per suite (using fixed seed → reproducible across runs) ─
    sampled_cases: list[tuple[str, int, int | None]] = []
    for suite_name in args.suites:
        if args.mode == "under_attack":
            dataset = initialize_dataset(suite_name, benign=False)
        else:
            dataset = initialize_dataset(suite_name, benign=True)
        n_total = len(dataset)
        n_sample = max(1, int(n_total * args.sample_pct))
        # Deterministic sample under fixed seed
        sampled = sorted(random.sample(dataset, n_sample))
        for item in sampled:
            if args.mode == "under_attack":
                uid, iid = item
                sampled_cases.append((suite_name, uid, iid))
            else:
                sampled_cases.append((suite_name, item, None))
        print(f"  {suite_name:<10} sampled {n_sample:>3} / {n_total:>4} "
              f"({100*n_sample/n_total:.0f}%)")

    # Build full work list: (idx, suite, defense, uid, iid)
    work_list: list[tuple] = []
    idx = 1
    for defense_name in args.defenses:
        for suite_name, uid, iid in sampled_cases:
            work_list.append((idx, suite_name, defense_name, uid, iid))
            idx += 1
    total = len(work_list)
    print(f"\n  Total work: {len(sampled_cases)} cases × "
          f"{len(args.defenses)} defenses = {total} runs\n")

    # ── Execute in parallel ───────────────────────────────────────────────────
    log_lock = threading.Lock()
    results: list[dict] = []
    t_start = time.time()

    def _submit(item):
        idx, suite_name, defense_name, uid, iid = item
        return _run_case(
            idx=idx, total=total,
            suite_name=suite_name, defense_name=defense_name,
            uid=uid, iid=iid, mode=args.mode,
            model_name=args.model, attack_name=args.attack,
            benchmark_version=args.benchmark_version,
            log_dir=args.output_dir, log_lock=log_lock,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_submit, item) for item in work_list]
        for fut in as_completed(futs):
            results.append(fut.result())

    elapsed_total = time.time() - t_start

    # Sort results by original idx for stable summary output
    results.sort(key=lambda r: r["idx"])

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY  (wall clock {elapsed_total:.0f}s)")
    print(f"{'='*70}\n")

    # Per (defense, suite) aggregation
    stats = defaultdict(lambda: {"total": 0, "utility": 0,
                                 "security_safe": 0, "errors": 0,
                                 "wall": 0.0,
                                 "hitl_load_total": 0,
                                 "endorsement_calls_total": 0,
                                 # TCR@k buckets — count tasks where utility=True
                                 # AND hitl_load (per task) ≤ k.
                                 "tcr_at_0": 0,
                                 "tcr_at_1": 0,
                                 "tcr_at_2": 0,
                                 "tcr_at_3": 0,
                                 "tcr_at_inf": 0,  # = utility (any HITL load)
                                 })
    for r in results:
        key = (r["defense"], r["suite"])
        stats[key]["total"] += 1
        h = int(r.get("hitl_load", 0))
        u = bool(r.get("utility"))
        if u:
            stats[key]["utility"] += 1
            if h == 0:
                stats[key]["tcr_at_0"] += 1
            if h <= 1:
                stats[key]["tcr_at_1"] += 1
            if h <= 2:
                stats[key]["tcr_at_2"] += 1
            if h <= 3:
                stats[key]["tcr_at_3"] += 1
            stats[key]["tcr_at_inf"] += 1
        if not r["security"]:        # security=False → SAFE in AgentDojo semantic
            stats[key]["security_safe"] += 1
        if r["error"]:
            stats[key]["errors"] += 1
        stats[key]["wall"] += r["elapsed"]
        stats[key]["hitl_load_total"] += h
        stats[key]["endorsement_calls_total"] += len(r.get("endorsement_log", []))

    for defense_name in args.defenses:
        print(f"  Defense: {defense_name}")
        print(f"  {'Suite':<10} {'TCR@0':>8} {'TCR@1':>8} {'TCR@∞':>8} "
              f"{'Safety':>9} {'HITL':>5} {'Err':>4} {'CPU-s':>7}")
        print(f"  {'-'*68}")
        total_u = total_s = total_n = 0
        total_t = 0.0
        total_h = 0
        total_t0 = total_t1 = 0
        for suite_name in args.suites:
            s = stats[(defense_name, suite_name)]
            n = s["total"]
            if n == 0:
                continue
            t0p = s["tcr_at_0"] * 100 / n
            t1p = s["tcr_at_1"] * 100 / n
            tinfp = s["tcr_at_inf"] * 100 / n
            s_pct = s["security_safe"] * 100 / n
            print(f"  {suite_name:<10} "
                  f"{s['tcr_at_0']:>3}/{n:<2} {t0p:>3.0f}% "
                  f"{s['tcr_at_1']:>3}/{n:<2} {t1p:>3.0f}% "
                  f"{s['tcr_at_inf']:>3}/{n:<2} {tinfp:>3.0f}% "
                  f"{s['security_safe']:>3}/{n:<2} {s_pct:>3.0f}% "
                  f"{s['hitl_load_total']:>4} "
                  f"{s['errors']:>3}  {s['wall']:>5.0f}")
            total_u += s["utility"]; total_s += s["security_safe"]
            total_n += n; total_t += s["wall"]
            total_h += s["hitl_load_total"]
            total_t0 += s["tcr_at_0"]; total_t1 += s["tcr_at_1"]
        if total_n:
            print(f"  {'-'*68}")
            print(f"  {'TOTAL':<10} "
                  f"{total_t0:>3}/{total_n:<2} {total_t0*100/total_n:>3.0f}% "
                  f"{total_t1:>3}/{total_n:<2} {total_t1*100/total_n:>3.0f}% "
                  f"{total_u:>3}/{total_n:<2} {total_u*100/total_n:>3.0f}% "
                  f"{total_s:>3}/{total_n:<2} {total_s*100/total_n:>3.0f}% "
                  f"{total_h:>4} "
                  f"{'':>3}  {total_t:>5.0f}")
        print()

    # Persist final summary
    summary = {
        "config": config,
        "wall_clock_seconds": round(elapsed_total, 1),
        "results": results,
        "stats": {f"{k[0]}_{k[1]}": v for k, v in stats.items()},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, cls=_Encoder, indent=2, ensure_ascii=False)

    print(f"  Output saved to: {args.output_dir}")
    print(f"  - config.json:        run configuration")
    print(f"  - detailed_logs.jsonl: per-case logs (one JSON per line)")
    print(f"  - summary.json:        aggregated results + stats")


if __name__ == "__main__":
    main()
