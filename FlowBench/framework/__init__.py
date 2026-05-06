"""FlowBench core framework: task spec, runner, Tier-1 graders, oracle,
attack templates, variant generators."""
from .task_spec import FlowBenchTask, TaskResult
from .runner import run_task
from .graders import (
    check_marker_absent,
    check_marker_absent_in_env,
    check_no_off_plan_tool,
    check_arg_in_entity_set,
    any_tool_called,
)
from .attack_templates import ATTACK_TEMPLATES, get_template, template_names
from .variant_gen import (
    VariantSelector,
    make_selectors,
    task_id,
    ATTACKER_IBAN_POOL,
    ATTACKER_EMAIL_POOL,
    G3_MARKER_POOL,
)

__all__ = [
    "FlowBenchTask",
    "TaskResult",
    "run_task",
    "check_marker_absent",
    "check_marker_absent_in_env",
    "check_no_off_plan_tool",
    "check_arg_in_entity_set",
    "any_tool_called",
    "ATTACK_TEMPLATES",
    "get_template",
    "template_names",
    "VariantSelector",
    "make_selectors",
    "task_id",
    "ATTACKER_IBAN_POOL",
    "ATTACKER_EMAIL_POOL",
    "G3_MARKER_POOL",
]
