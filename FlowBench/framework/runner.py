"""End-to-end runner for a single FlowBenchTask against one defense."""
from __future__ import annotations

import time
from typing import Any, Optional

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.models import ModelsEnum

from .task_spec import FlowBenchTask, TaskResult
from .oracle import augmented_query, hitl_cost


def _count_tool_calls(messages: list[dict]) -> int:
    n = 0
    for m in messages:
        tcs = m.get("tool_calls") or []
        n += len(tcs)
    return n


def run_task(task: FlowBenchTask,
             defense: Optional[str],
             mode: str,
             agent_model: str = "gpt-4o-2024-11-20") -> TaskResult:
    """Run `task` under `defense` in one of four modes:

        benign          — no injection, no oracle
        attacked        — injection planted, no oracle
        benign+oracle   — no injection, user query augmented with the
                          oracle's extracted trusted values (+hitl_load)
        attacked+oracle — injection planted AND user query augmented
                          (+hitl_load). Attacker's injected values should
                          be structurally overridden by the oracle hints.
    """
    assert mode in {"benign", "attacked", "benign+oracle", "attacked+oracle"}
    defense_label = defense if defense else "None"

    env, runtime = task.build_env()
    if "attacked" in mode:
        task.plant_injection(env)

    use_oracle = mode.endswith("+oracle")
    query = (augmented_query(task.user_query, task.oracle_extracts)
             if use_oracle else task.user_query)
    hitl = hitl_cost(task.oracle_extracts) if use_oracle else 0

    pipeline = AgentPipeline.from_config(PipelineConfig(
        llm=ModelsEnum(agent_model),
        defense=defense,
        system_message_name=None,
        system_message=None,
    ))

    extra_args = {"input_tokens": 0, "output_tokens": 0}

    t0 = time.time()
    try:
        _, _, post_env, messages, extra_args = pipeline.query(
            query, runtime, env, extra_args=extra_args,
        )
        elapsed = time.time() - t0
        err = None
    except Exception as e:
        elapsed = time.time() - t0
        post_env = env
        messages = []
        err = f"{type(e).__name__}: {str(e)[:200]}"

    utility: Optional[bool]
    safety:  Optional[bool]
    try:
        utility = bool(task.utility_grader(post_env, messages)) if not err else None
    except Exception as e:
        utility = None
        err = (err or "") + f" | utility_grader: {e}"
    try:
        safety  = bool(task.safety_grader(post_env, messages)) if not err else None
    except Exception as e:
        safety = None
        err = (err or "") + f" | safety_grader: {e}"

    return TaskResult(
        task_id=task.task_id,
        defense=defense_label,
        mode=mode,
        utility=utility,
        safety=safety,
        elapsed=elapsed,
        n_tool_calls=_count_tool_calls(messages),
        hitl_load=hitl,
        input_tokens=extra_args.get("input_tokens", 0),
        output_tokens=extra_args.get("output_tokens", 0),
        error=err,
        notes={},
    )
