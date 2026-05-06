"""TaskSpec + result dataclasses for FlowBench."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FlowBenchTask:
    """A single FlowBench meta-example.

    The contract is deliberately thin so pilot tasks can be self-contained.
    Later refactor can extract common env builders / injection planters when
    10+ tasks share a template.
    """

    task_id:        str              # e.g., "banking_g2_pilot"
    scenario:       str              # banking / email / homework / slack / web
    g_category:     str              # G1 / G2 / G3 / DEL
    user_query:     str              # the benign user prompt

    # Environment construction + injection planting.
    # build_env() -> (env, runtime, tools) — the return shape is per-scenario.
    build_env:      Callable[[], Any]
    # Given the env returned by build_env, mutate it in place to plant the
    # attack payload. Returns nothing; raises if the plant target is missing.
    plant_injection: Callable[[Any], None]

    # Graders receive the post-execution env + the full message list and
    # return bool. The semantics (utility=task done, safety=attack blocked)
    # are task-specific because each scenario has different success signals.
    utility_grader: Callable[[Any, list[dict]], bool]
    safety_grader:  Callable[[Any, list[dict]], bool]

    # Optional: oracle-extracted trusted values for content-sourced args.
    # When mode ends with '+oracle', the runner appends these as confirmed
    # user hints to the query, and charges hitl_load += len(oracle_extracts).
    # Used to rescue G2b (arg-from-tool-output) without sacrificing safety.
    oracle_extracts: dict[str, str] = field(default_factory=dict)

    # Metadata / notes
    marker:         Optional[str] = None      # primary attack marker, if any
    notes:          str = ""

    def __post_init__(self) -> None:
        assert self.scenario in {"banking", "email", "homework", "slack", "web"}
        assert self.g_category in {"G1", "G2", "G3", "DEL"}


@dataclass
class TaskResult:
    task_id:        str
    defense:        str              # "None" / "grade_dual" / "spotlighting_with_delimiting"
    mode:           str              # "benign" / "attacked" / "benign+oracle" / "attacked+oracle"
    utility:        Optional[bool]   # None if grader couldn't decide / errored
    safety:         Optional[bool]   # same
    elapsed:        float
    n_tool_calls:   int
    hitl_load:      int              = 0  # oracle interactions + endorsement prompts
    input_tokens:   int              = 0
    output_tokens:  int              = 0
    error:          Optional[str]    = None
    notes:          dict             = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        if self.error:
            return "ERROR"
        if "attacked" in self.mode:
            if self.safety is True:
                return "SAFE" + (f" (hitl={self.hitl_load})" if self.hitl_load else "")
            elif self.safety is False:
                return "BREACH"
            else:
                return "SAFETY-UNKNOWN"
        else:  # benign
            if self.utility is True:
                return "UTIL-YES" + (f" (hitl={self.hitl_load})" if self.hitl_load else "")
            elif self.utility is False:
                return "UTIL-NO"
            else:
                return "UTIL-UNKNOWN"
