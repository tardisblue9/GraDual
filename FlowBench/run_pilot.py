"""Run all 5 FlowBench pilot meta-examples under None + GraDual × benign/attacked.

Usage:
    cd <repo-root>/GraDual
    python -m FlowBench.run_pilot
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import traceback

# ── Environment ───────────────────────────────────────────────────────────
# Set these in your shell, or hardcode them here:
#     export OPENAI_API_KEY=sk-...
#     export OPENAI_BASE_URL=https://yunwu.ai/v1
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
for _pv in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
            "https_proxy", "http_proxy", "all_proxy"):
    os.environ.pop(_pv, None)

# Ensure local packages resolve when run from the GraDual root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# Default backbone: qwen3-coder. Override with env FLOWBENCH_MODEL.
# Supported values in src/agentdojo/models.py::ModelsEnum.
AGENT_MODEL = os.environ.get("FLOWBENCH_MODEL", "qwen3-coder")

DEFENSES = [None, "grade_dual"]

# Per-task modes. Tasks that declare oracle_extracts get the additional
# +oracle modes; otherwise we keep it to plain benign/attacked.
def _modes_for(task) -> list[str]:
    base = ["benign", "attacked"]
    if task.oracle_extracts:
        base += ["benign+oracle", "attacked+oracle"]
    return base

from FlowBench.framework import run_task
from FlowBench.framework.task_spec import TaskResult

from FlowBench.scenarios.banking_g1  import TASK as TASK_BANKING_G1
from FlowBench.scenarios.banking_g2  import TASK as TASK_BANKING_G2
from FlowBench.scenarios.email_g1    import TASK as TASK_EMAIL_G1
from FlowBench.scenarios.email_g3    import TASK as TASK_EMAIL_G3
from FlowBench.scenarios.homework_g3 import TASK as TASK_HOMEWORK_G3
from FlowBench.scenarios.slack_g3    import TASK as TASK_SLACK_G3
from FlowBench.scenarios.web_g2      import TASK as TASK_WEB_G2

PILOTS = [
    TASK_BANKING_G1,
    TASK_BANKING_G2,
    TASK_EMAIL_G1,
    TASK_EMAIL_G3,
    TASK_HOMEWORK_G3,
    TASK_SLACK_G3,
    TASK_WEB_G2,
]


def main() -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(_THIS_DIR, "logs", f"pilot_{ts}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"FlowBench pilot — model={AGENT_MODEL}  log_dir={log_dir}")
    print("=" * 72)

    results: list[TaskResult] = []
    for task in PILOTS:
        modes = _modes_for(task)
        print(f"\n### {task.task_id}  ({task.scenario}/{task.g_category})  modes={modes}")
        for defense in DEFENSES:
            for mode in modes:
                tag = f"{task.task_id} | defense={defense or 'None':<11} | {mode:<18}"
                try:
                    r = run_task(task, defense=defense, mode=mode,
                                 agent_model=AGENT_MODEL)
                except Exception as e:
                    traceback.print_exc()
                    r = TaskResult(
                        task_id=task.task_id,
                        defense=defense or "None",
                        mode=mode,
                        utility=None,
                        safety=None,
                        elapsed=0.0,
                        n_tool_calls=0,
                        error=f"run_task raised: {type(e).__name__}: {str(e)[:160]}",
                    )
                results.append(r)
                print(f"  {tag}  →  {r.outcome:<25}  "
                      f"(util={r.utility} safe={r.safety} "
                      f"calls={r.n_tool_calls} hitl={r.hitl_load} "
                      f"{r.elapsed:.1f}s)")
                if r.error:
                    print(f"      ⚠ {r.error}")

                fn = os.path.join(
                    log_dir,
                    f"{task.task_id}__{defense or 'None'}__{mode.replace('+', '_')}.json",
                )
                with open(fn, "w") as f:
                    json.dump(r.__dict__, f, indent=2, default=str)

    # ── Summary ──
    print("\n" + "=" * 100)
    print("PILOT SUMMARY")
    print("=" * 100)
    hdr = (f"{'task':<22} {'defense':<11} "
           f"{'benign':<22} {'attacked':<22} "
           f"{'benign+oracle':<22} {'attacked+oracle':<22}")
    print(hdr)
    print("-" * len(hdr))

    # Build a {(task_id, defense): {mode: result}} table
    tbl: dict[tuple[str, str], dict[str, TaskResult]] = {}
    for r in results:
        tbl.setdefault((r.task_id, r.defense), {})[r.mode] = r

    for (tid, dfn), by_mode in tbl.items():
        cells = []
        for mode in ("benign", "attacked", "benign+oracle", "attacked+oracle"):
            r = by_mode.get(mode)
            cells.append(r.outcome if r else "—")
        print(f"{tid:<22} {dfn:<11} "
              f"{cells[0]:<22} {cells[1]:<22} "
              f"{cells[2]:<22} {cells[3]:<22}")

    # Save the full result dump.
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2, default=str)
    print(f"\nFull log dir: {log_dir}")


if __name__ == "__main__":
    main()
