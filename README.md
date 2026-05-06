# GraDual

---

## Layout

```
GraDual/
├── pyproject.toml
├── README.md
├── docs/
│   └── overhead_comparison.md         # measured wall-clock + token cost vs. baselines
├── src/agentdojo/                     # AgentDojo framework (slimmed)
│   ├── agent_pipeline/
│   │   ├── agent_pipeline.py          # only None / spotlight / grade_dual wired
│   │   ├── llms/grade_dual_llm.py     # the GraDual defense (single file, ~7.6 k LOC)
│   │   ├── llms/openai_llm.py
│   │   ├── llms/prompting_llm.py
│   │   ├── endorsement_oracle.py
│   │   ├── utility_tools.py           # calculate / table_query / summarize_opaque / request_endorsement
│   │   └── …
│   ├── attacks/                       # important_instructions, dos, agentic …
│   ├── default_suites/                # AgentDojo suites v1, v1.1, v1.1.1, v1.1.2
│   └── task_suite/, benchmark.py, functions_runtime.py, …
├── run/
│   ├── quick_eval.py                  # parallel sampled benchmark runner (AgentDojo)
│   └── data_module.py
├── examples/
│   └── run_single_case.py             # one-case end-to-end demo
└── FlowBench/                         # G1/G2/G3 control-information flow benchmark
    ├── README.md                      # how to run + headline results
    ├── framework/                     # task spec, graders, oracle, runner, attack templates
    ├── scenarios/                     # 7 G1/G2/G3 task families × 15 variants = 105 tasks
    ├── web_mock/browser.py            # mock web env (used by web_g2)
    ├── run_pilot.py                   # 7 hand-curated pilot tasks
    └── run_full.py                    # parallel dataset sweep
```

## Available defenses

| Name in code | Description |
|---|---|
| `None` | No defense — vanilla tool-calling baseline |
| `spotlighting_with_delimiting` | Delimiter-based defense ([Hines et al., 2024](https://arxiv.org/abs/2403.14720)) |
| `grade_dual` | **GraDual** — graph-based dual-agent defense with schema isolation |

## Install

Requires Python ≥ 3.10.

```bash
cd <repo-root>/GraDual
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set your OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://yunwu.ai/v1   # or https://api.openai.com/v1, vLLM, etc.
```

## Run a single case

```bash
python examples/run_single_case.py \
    --suite workspace --uid 3 --iid 1 \
    --defenses None spotlighting_with_delimiting grade_dual \
    --model qwen3-coder
```

Output:
```
Defense                          Utility   Safe  HITL    In tok   Out tok  Wall-s
----------------------------------------------------------------------------------
None                                True  False     0    25,970       463    14.1
spotlighting_with_delimiting        True  False     0    25,722       448    14.1
grade_dual                          True   True     0   132,189     4,429    95.7
```

## Reproduce the parallel benchmark

```bash
# 10% sample of all 4 suites, 4 defenses compared, 8 worker threads
python run/quick_eval.py \
    --defenses None spotlighting_with_delimiting grade_dual \
    --model qwen3-coder \
    --sample_pct 0.1 \
    --workers 8
```

Outputs land in `evaluation_results/quick_eval_parallel_…/`:
- `summary.json` — aggregated stats + per-case results
- `detailed_logs.jsonl` — full message trace per case
- `config.json` — run configuration

## Evaluated overhead

See `docs/overhead_comparison.md` for the headline numbers (qwen3-coder, 10% sample of AgentDojo v1.1.2, 4 suites). Summary:

| Defense | ×baseline tokens | avg/case latency | TCR@∞ | Safety |
|---|---:|---:|---:|---:|
| None | 1.00× | 14.1 s | 55% | 56% |
| spotlight | 0.99× | 14.1 s | 58% | 69% |
| **grade_dual** | **5.17×** | **95.7 s** | **68%** | **100%** |

GraDual trades a 5× cost for 100% safety on IPI attack.

## FlowBench

`FlowBench/` is a separate, self-contained benchmark for control-information
flow attacks (G1/G2/G3 categories). 7 scenarios × 15 variants = 105 tasks
across banking / email / homework / slack / web. Headline result on this
benchmark (GPT-4o backbone): GraDual attacked SAFE = **115/120 (95.8%)**;
qwen3-coder cross-backbone: **118/120 (98.3%)**.

```bash
python -m FlowBench.run_pilot                       # 7-task sequential smoke
python -m FlowBench.run_full --variants 15          # full 105-task parallel sweep
```

See `FlowBench/README.md` for the full results matrix and architecture.

## Architecture (one-line)

`grade_dual` is a two-phase, two-agent design implemented entirely in `src/agentdojo/agent_pipeline/llms/grade_dual_llm.py`:

1. **Phase 1 — Construct.** A *Main Agent* reads the user query and produces a data-flow graph of `SemanticNode`/`EntityNode`/`ControlNode` describing the planned tool-call sequence.
2. **Phase 2 — Execute.** Each `ControlNode` fires its tool, and a stateless **Dual Agent** fills the response into a typed schema with `safety_mode=opaque_ref` for any free-text body. Bodies live behind opaque handles (`<ref:NODE_ID.path>`) and are read only via `summarize_opaque` (which is itself a Dual-Agent call with policy verifiers + relay-audit). Untrusted text never reaches the Main Agent's argument-resolution path verbatim.

Key files to read:
- `src/agentdojo/agent_pipeline/llms/grade_dual_llm.py` — defense
- `src/agentdojo/agent_pipeline/agent_pipeline.py` — wiring
- `src/agentdojo/agent_pipeline/endorsement_oracle.py` — HITL endorsement protocol (v8)

