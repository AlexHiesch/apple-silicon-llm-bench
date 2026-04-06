# Quality & Agentic Benchmark Design
## llm-bench — Phase 2

*Scope document for the dimensions that throughput benchmarks miss.*
*Counterpart to the existing `benchmark.py` speed suite.*

---

## Why this exists

The current `benchmark.py` measures **infrastructure performance**: TTFT, decode t/s, prefill t/s, memory RSS. These predict latency budgets but say nothing about whether the model actually works for the use case.

Document §6 identifies the real production failure modes for agentic coding:

> "The most commonly cited agentic failure mode is not hallucination in output text — it is syntactically valid tool calls with semantically wrong arguments. The tool executes, returns garbage, and the agent doesn't notice."

This document scopes six benchmark dimensions that measure those failure modes directly.

---

## Dimension 1 — Tool Calling Accuracy

### What it measures
Whether the model fills tool call parameters correctly — not just structurally valid JSON, but semantically correct argument values.

### Why it's critical
- Sequential-dependent tool calling: 14.8% parameter-filling accuracy in best-case English pipeline, 4.3% multilingual (VoiceAgentBench, AAAI 2026)
- A structurally valid but semantically wrong call is the hardest failure to catch — the pipeline continues silently
- Quantization degrades structured output reliability before it degrades free-text quality

### Test cases

**Tier 1 — Single tool, single call**
```
Tools: read_file(path: str), write_file(path: str, content: str), run_command(cmd: str)
Task: "Read the file at /src/utils.py"
Expected: read_file(path="/src/utils.py")
Variants: 20 tasks × 5 tool schemas
```

**Tier 2 — Multi-tool sequential**
```
Tools: search_code(query: str) → returns file list
       read_file(path: str) → returns content
       edit_file(path: str, old: str, new: str)
Task: "Find where the login function is defined and rename the parameter from 'user' to 'username'"
Expected: search_code → read_file → edit_file (each step depends on prior output)
Variants: 10 tasks, 3-step chains
```

**Tier 3 — Error recovery**
```
Tool returns: {"error": "file not found", "suggestion": "/src/util.py"}
Expected: model retries with corrected path
Variants: 5 error types × 3 retry strategies
```

### Metrics
| Metric | Definition |
|--------|-----------|
| `tool_call_valid` | JSON parses + all required fields present (already in benchmark.py, unused) |
| `tool_call_accurate` | All parameter values semantically correct vs expected |
| `tool_call_recovery` | Correct retry after error response |
| `chain_success_rate` | Full multi-step chain completes with correct final result |

### Implementation
- `quality_bench/tool_calling.py`: 35 test cases in JSON, each with `task`, `tools_schema`, `expected_calls`, `scoring_fn`
- Scoring: deterministic where possible (exact match on paths, regex for flexible matches), LLM-as-judge for semantic equivalence on complex args
- Integrate into `benchmark.py` via `--test-tools` flag (stub already exists)

### Priority models
All backends where response speed makes agentic use realistic: mlx-lm, Ollama 0.19 NVFP4/INT4, oMLX, TurboQuant. Also test across quant levels (4bit vs 8bit) to measure quantization impact on tool accuracy.

---

## Dimension 2 — Long Context Retrieval (Needle + RULER)

### What it measures
Whether the model can actually use information at long context depths — the gap between advertised and effective context window.

### Why it's critical
- 10 out of 12 tested models achieved ~50% of short-context performance at 32K tokens
- U-shaped recall: >30% drop for information in the middle of context
- GPT-4 lost 15.4 points going from 4K to 128K tokens
- Our hardware can run 128K context on MoE models — but can the model reason over it?

### Test cases

**Needle-in-a-haystack (NIAH)**
```
Haystack: Paul Graham essays (or code files — more realistic for coding)
Needle: single unique fact planted at depth D% through the context
Task: "What is the special value mentioned in the document?"
Depths: 10%, 25%, 50%, 75%, 90% of context
Context lengths: 4k, 8k, 16k, 32k, 64k, 128k (model permitting)
```

**Multi-needle (more agentic)**
```
Haystack: codebase files concatenated
Needles: 3–5 function signatures planted across files
Task: "List all functions that take a 'config' parameter"
Tests: recall, precision, ordering independence
```

**RULER-style tasks**
```
Variable tracking: "The value of X is 42. [10k tokens of distraction]. What is X?"
Multi-hop: "File A imports File B. File B imports File C. Does A depend on C?"
CWE (common word extraction): count occurrences of target word across long document
```

### Metrics
| Metric | Definition |
|--------|-----------|
| `retrieval_accuracy@depth` | Correct at each needle depth position |
| `retrieval_accuracy@length` | Correct at each context length |
| `position_bias_score` | Difference between best (start/end) and worst (middle) depth |
| `effective_context_window` | Largest context length where accuracy stays above 80% |

### Implementation
- `quality_bench/needles.py`: generates haystacks at each length, plants needles, scores responses
- `quality_bench/ruler_tasks.py`: RULER-inspired tasks, self-contained (no external API)
- Run via `benchmark.py --test-quality` (stub already exists)
- Context lengths: reuse existing `CONTEXT_TIERS` dict from benchmark.py

### Coding-specific variant
Replace Paul Graham essays with real code: use the mlx-lm codebase itself as haystack (~200K tokens). Ask questions about function definitions, import chains, class hierarchies. More predictive of actual Claude Code use than abstract recall.

---

## Dimension 3 — Quantization × Context Quality Interaction

### What it measures
How much quantization degrades *quality* (not speed) as context grows — the compound error effect.

### Why it's critical
- Weight quant + KV quant each add small errors; under long context these compound
- The same model loses 0.5 ppl points at short context under Q4_K_M but may show larger capability gaps at 16K+ tokens on multi-step reasoning
- This interaction is almost never benchmarked

### Test design
```
Fixed task: multi-step reasoning problem requiring 3-5 logical steps
Fixed model: Qwen3.5-35B-A3B (as reference)
Variable 1: quantization level (4bit, 8bit)
Variable 2: context length (1k, 4k, 16k, 32k, 128k)
Variable 3: KV cache quant (none, q8, TurboQuant-4bit, TurboQuant-3bit)

For each (quant × context × kv_quant) cell:
  - Run 10 reasoning tasks
  - Score: correct/incorrect (binary), not perplexity
  - Compute accuracy@(quant, context, kv)
```

### Output: degradation heatmap
```
                 | 1K ctx | 4K ctx | 16K ctx | 32K ctx | 128K ctx
4bit weight / FP16 KV | 94% | 92% | 88% | 81% | 72%
4bit weight / TQ-4 KV | 93% | 91% | 87% | 80% | 71%
8bit weight / FP16 KV | 96% | 95% | 93% | 89% | 84%
```

Shows the "cliff" where quantization error becomes unacceptable for agentic tasks.

### Implementation
- 10 GSM8K-style coding problems with unambiguous correct/incorrect scoring
- Pad context with relevant-looking but non-interfering code to reach target length
- Matrix runner: `quality_bench/quant_context_matrix.py`

---

## Dimension 4 — Thinking Token Overhead

### What it measures
The real latency cost of Qwen3's `/think` mode at different budget caps — and whether the quality improvement justifies the overhead for agentic coding.

### Why it's critical
- Thinking tokens are invisible in decode t/s benchmarks (they're just more tokens)
- A 32K thinking chain at 90 t/s = 5.5 minutes of latency before any visible output
- KV cache fills with thinking tokens → memory pressure compounds over multi-turn sessions
- The optimal budget cap for coding tasks is unknown

### Test design
```
Models: Qwen3.5-35B-A3B, Qwen3-Coder-Next (both support /think)
Backends: mlx-lm (baseline), Ollama 0.19 NVFP4

Thinking budgets tested:
  --no-think (0 thinking tokens)
  --max-tokens-think 1024  (tight cap)
  --max-tokens-think 4096  (moderate)
  --max-tokens-think 16384 (generous)
  uncapped (default)

Tasks (10 per tier):
  Tier A — Simple: "write a function to reverse a string"
  Tier B — Medium: "refactor this class to use dependency injection" [300 token class]
  Tier C — Hard: "find the bug in this concurrent code" [500 token snippet with race condition]
  Tier D — Agentic: multi-step, tool-dependent plan requiring 3 tool calls
```

### Metrics
| Metric | Definition |
|--------|-----------|
| `thinking_tokens` | Actual thinking tokens generated (already in benchmark.py) |
| `task_accuracy` | Correct solution per tier |
| `latency_per_correct_solution` | Total time / accuracy — the real tradeoff metric |
| `memory_pressure_at_turn_N` | Peak RSS after N multi-turn exchanges with thinking |

### Key question
Does 4K thinking tokens deliver 80% of uncapped accuracy at 20% of the latency? If so, `--max-tokens-think 4096` is the production setting for agentic coding.

---

## Dimension 5 — Cache Hit Rate in Agentic Workflows

### What it measures
Whether prefix caching actually works for the Claude Code pattern — and by how much it reduces effective TTFT in a multi-turn session.

### Why it's critical
- Agentic coding: large fixed prefix (system prompt + tool definitions + file context) + small varying suffix (new user message)
- oMLX claims 10× faster prefill at 8K context via persistent KV cache
- Ollama 0.19 cross-conversation prefix reuse is the key new feature
- Our current A_OLL_CTX test measures cold vs warm TTFT for a single pair of runs — not a realistic multi-turn simulation

### Test design
```
Shared prefix: Claude Code system prompt (~4K tokens) + 3 tool definitions (~2K) + 2 files (~6K) = ~12K tokens total
Varying suffix: 10 different user messages (200–500 tokens each)

Session simulation:
  Turn 1: full context (12K prefix + suffix_1) — COLD
  Turn 2: same prefix + suffix_2 — should be WARM (prefix cached)
  Turn 3–10: same prefix + suffix_N

Measure per turn:
  - TTFT
  - Whether cache was hit (inferred from TTFT speedup or prefill t/s)
  - Decode t/s (should be unaffected)

Run across backends:
  Ollama 0.19 NVFP4 (intelligent prefix checkpoints)
  oMLX (SSD persistent cache)
  mlx-lm (no cross-request cache — baseline)
  llama-server --cache-reuse 256 (KV shifting)
```

### Metrics
| Metric | Definition |
|--------|-----------|
| `ttft_turn_1` | Cold TTFT — no cache |
| `ttft_turn_2+_median` | Warm TTFT — cached prefix |
| `cache_speedup_ratio` | ttft_turn_1 / ttft_turn_2 |
| `effective_throughput` | Tokens output per wall-clock second across a 10-turn session |

### Implementation
- `quality_bench/agentic_session.py`: builds the shared prefix, sends 10 sequential requests to the running server, records per-turn TTFT
- Reuses `benchmark.py` server management (start_server, wait_for_server, stop_server)
- Can be run as a new benchmark.py subcommand: `--test-session`

---

## Dimension 6 — Context Rot Index

### What it measures
Performance delta between turn 1 and turn 10 of a realistic agentic session — does accumulated context degrade the model's ability to follow instructions?

### Why it's critical
- Agentic sessions grow context over time with tool outputs, file contents, error messages
- The model at turn 10 sees 40K+ tokens of history before the current task
- "Stuffing 100K tokens of history degrades the model's ability to reason about what actually matters" (quoted in the document)
- This is distinct from long-context retrieval: the degradation is cognitive, not just memory

### Test design
```
Session structure:
  Turn 1: simple coding task (baseline quality)
  Turns 2–9: unrelated tool calls that expand context by ~3K tokens each
  Turn 10: exact same task as Turn 1 (repeated)

Score: quality(Turn 10) / quality(Turn 1) = context rot index
  1.0 = no degradation
  0.8 = 20% degradation — marginal
  0.5 = 50% degradation — severe

Tasks:
  10 coding tasks × 3 session types (light/medium/heavy context accumulation)
```

### Metrics
| Metric | Definition |
|--------|-----------|
| `context_rot_index` | quality(turn_10) / quality(turn_1) per task |
| `rot_threshold_tokens` | Context size where index drops below 0.8 |
| `quant_rot_delta` | Difference in rot index between 4bit and 8bit |

---

## Implementation Roadmap

### Phase 2a — Foundation (2–3 days)
1. `quality_bench/` package with shared utilities: prompt builder, response scorer, session runner
2. Tool calling tests (Dimension 1, Tier 1 only) — 20 single-call test cases
3. Basic NIAH — single needle, 5 depths × 4 context lengths
4. Wire into `benchmark.py --test-tools` and `--test-quality` flags (stubs already exist)

### Phase 2b — Coverage (1 week)
5. Multi-tool sequential (Dimension 1 Tier 2)
6. Thinking token matrix (Dimension 4) — already have `thinking_tokens` in CSV schema
7. Cache hit simulation (Dimension 5) — builds on existing server management

### Phase 2c — Full matrix (1–2 weeks)
8. Quantization × context heatmap (Dimension 3)
9. Context rot index (Dimension 6)
10. RULER-style tasks (Dimension 2 extension)
11. Combined HTML report with quality + speed in same view

### Data schema additions to benchmark.py

```python
# Already in TestResult (unused):
tool_call_valid: Optional[bool] = None
quality_pass: Optional[bool] = None

# New fields needed:
tool_call_accurate: Optional[bool] = None   # semantic correctness
retrieval_accuracy: Optional[float] = None  # 0.0–1.0 per test
thinking_budget: Optional[int] = None       # cap used
session_turn: Optional[int] = None          # for multi-turn tests
```

---

## Test corpus sources (no external API needed)

| Source | Use |
|--------|-----|
| GSM8K (public) | Reasoning tasks with verifiable answers |
| HumanEval (public) | Code generation with unit test scoring |
| mlx-lm source code | Realistic coding haystack for NIAH |
| Claude Code system prompt (approximated) | Realistic agentic prefix for cache tests |
| Synthetic tool schemas | Tool calling accuracy — hand-crafted for coverage |

All scoring is local: unit tests, regex, exact match, or a small local judge model (Qwen3.5-9B already in LM Studio — 10.45GB, fast for scoring).

---

## Estimated effort vs insight value

| Dimension | Effort | Insight value |
|-----------|--------|---------------|
| Tool calling (Tier 1) | 2 days | Very high — direct failure mode |
| NIAH single-needle | 1 day | High — reveals effective context window |
| Thinking token matrix | 1 day | High — production config decision |
| Cache hit simulation | 1 day | High — validates oMLX/Ollama 0.19 claims |
| Quant × context heatmap | 3 days | Medium — confirms compound degradation |
| Context rot index | 3 days | Medium — complex to score reliably |
| Multi-tool sequential | 3 days | High — but hard to score fairly |
| RULER full | 5 days | Medium — overlaps with NIAH |

**Recommended start:** Tool calling Tier 1 + NIAH + Thinking token matrix. Three days of work, answers the three highest-stakes questions for the Claude Code replacement decision.
