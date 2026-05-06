"""FlowBench endorsement oracle — perfect HITL simulation.

Concept: when GraDual cannot bind an argument to a user-named entity (pure
G2b: target is only readable from a tainted tool output), a HITL human
could look at the source and tell the agent the trusted value of that
argument. The 'perfect oracle' models the upper bound: human always returns
the ground-truth value, independent of what the attacker may have injected.

Design:
    - A FlowBenchTask declares `oracle_extracts: dict[str, str]` — a mapping
      of short human-readable field labels to the ground-truth trusted value
      (e.g., {"payment IBAN": "UK12345678901234567890"}).
    - The runner, when invoked with mode="attacked+oracle" (or
      "benign+oracle"), resolves those fields BEFORE the agent turn and
      appends them to the user query as explicit hints, simulating a HITL
      round where the user consulted the oracle and told the agent.
    - hitl_load is incremented by len(oracle_extracts) on an oracle run.

This is a pre-resolution model: equivalent to (a) the agent running the
pipeline, (b) pausing when it tries to use a tainted-sourced arg, (c)
querying a HITL who sees the source and types in the trusted value, (d)
resuming. The semantic effect is the same: the agent has a trusted source
for the arg, and the attacker-injected value never enters the trusted
context.

Compared to PRUDENTIA/FIDES 'endorse':
    - Endorse = relabel the entire tainted variable U → T, then the
      UNCHANGED content flows. If content carries a marker, it goes through.
    - Oracle extract = return ONLY the trusted value (typed, narrow). The
      broader tainted content is never promoted to T. Injection content
      has no path to the trusted context.
"""
from __future__ import annotations

from typing import Any


def format_oracle_hints(extracts: dict[str, str]) -> str:
    """Render the oracle's extractions as terse inline clarifications.

    We weave them into the user prompt as explicit *values* rather than a
    separate appendix, so the Phase-1 planner treats them as primary user-
    named entities (same provenance class as any literal in the query).
    """
    if not extracts:
        return ""
    # Short, directive form — the LLM picks up values that look like
    # imperative constraints more reliably than ones phrased as "for
    # reference" annexes.
    pieces = [f"{label}: {value}" for label, value in extracts.items()]
    return " (Use " + "; ".join(pieces) + ".)"


def augmented_query(base_query: str, extracts: dict[str, str]) -> str:
    """Return the base user query plus the oracle-extracted hints inlined."""
    return base_query + format_oracle_hints(extracts)


def hitl_cost(extracts: dict[str, str]) -> int:
    """Number of HITL interactions the oracle consumed.

    Each extracted field counts as one HITL interaction (a human typed in
    the trusted value for that field, once, for this task)."""
    return len(extracts or {})
