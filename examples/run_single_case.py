"""Run a single (suite, user_task, injection_task) case through one or more
defenses, end-to-end.

Usage:
    cd <repo-root>/GraDual

    # Default: workspace user_task_3 + injection_task_1, grade_dual defense
    python examples/run_single_case.py

    # Specific case + multiple defenses
    python examples/run_single_case.py --suite travel --uid 0 --iid 0 \
        --defenses None spotlighting_with_delimiting grade_dual

Prerequisites: set OPENAI_API_KEY and OPENAI_BASE_URL in your shell, or hardcode
them at the top of `run/quick_eval.py`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ── Path bootstrap ────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentdojo.task_suite.load_suites import get_suite
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.endorsement_oracle import make_gt_oracle
from agentdojo.agent_pipeline.utility_tools import calculate, table_query
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import make_function
from agentdojo.models import ModelsEnum


def run_one(suite_name: str, uid: int, iid: int | None,
            defense: str | None, model: str, attack_name: str,
            benchmark_version: str = "v1.1.2") -> dict:
    suite = get_suite(benchmark_version, suite_name)

    # Make calculator + table_query available to every defense.
    for tool in (make_function(calculate), make_function(table_query)):
        if tool not in suite.tools:
            suite.tools = list(suite.tools) + [tool]

    pipeline = AgentPipeline.from_config(PipelineConfig(
        llm=ModelsEnum(model),
        defense=defense,
        system_message_name=None,
        system_message=None,
    ))
    user_task = suite.get_user_task_by_id(f"user_task_{uid}")

    if iid is not None:
        injection_task = suite.get_injection_task_by_id(f"injection_task_{iid}")
        attacker = load_attack(attack_name, suite, pipeline)
        injections = attacker.attack(user_task, injection_task)
    else:
        injection_task = None
        injections = {k: "" for k in suite.get_injection_vector_defaults().keys()}

    extra_args = {
        "input_tokens": 0,
        "output_tokens": 0,
        "endorsement_oracle": make_gt_oracle(
            user_task=user_task,
            injection_task=injection_task,
            injections=injections,
        ),
    }
    t0 = time.time()
    utility, security, _msgs, info = suite.run_task_with_pipeline_msg(
        agent_pipeline=pipeline,
        user_task=user_task,
        injection_task=injection_task,
        injections=injections,
        extra_args=extra_args,
    )
    elapsed = time.time() - t0

    return {
        "defense": defense or "None",
        "utility": bool(utility),
        # AgentDojo: security=True ⇒ injection succeeded ⇒ UNSAFE.
        "safe": (injection_task is None) or (not bool(security)),
        "elapsed": round(elapsed, 1),
        "input_tokens": info.get("input_tokens", 0),
        "output_tokens": info.get("output_tokens", 0),
        "hitl_load": int(extra_args.get("hitl_load", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--uid", type=int, default=3)
    parser.add_argument("--iid", type=int, default=1, help="Injection task id; pass -1 for benign mode")
    parser.add_argument("--defenses", nargs="+",
                        default=["None", "spotlighting_with_delimiting", "grade_dual"])
    parser.add_argument("--model", default="qwen3-coder")
    parser.add_argument("--attack", default="important_instructions")
    args = parser.parse_args()

    iid = None if args.iid < 0 else args.iid
    print(f"Suite={args.suite} uid={args.uid} iid={iid} model={args.model}\n")
    print(f"{'Defense':<32}{'Utility':>9}{'Safe':>7}{'HITL':>6}{'In tok':>10}{'Out tok':>10}{'Wall-s':>8}")
    print("-" * 82)
    for d in args.defenses:
        defense = None if d == "None" else d
        r = run_one(args.suite, args.uid, iid, defense, args.model, args.attack)
        print(f"{r['defense']:<32}{str(r['utility']):>9}{str(r['safe']):>7}"
              f"{r['hitl_load']:>6}{r['input_tokens']:>10,}{r['output_tokens']:>10,}{r['elapsed']:>8.1f}")


if __name__ == "__main__":
    main()
