# Next Benchmark Run — TODO

## Status: IN PROGRESS (2026-04-02 20:06)

Running `run_complete_matrix.py` — supersedes this file.
Check live: `tail -f /tmp/complete_matrix.log`

---

## What it runs (in order)

| Step | Test IDs | Why |
|------|----------|-----|
| 1 | F_TQ4_1, F_TQ35_1 | TurboQuant baselines re-run — tainted CSV deleted (bad USB-C cable) |
| 2 | A_OLL_CTX_INT4 | Ollama 0.19 INT4 ctx-32k/64k — tainted CSV deleted (44.9 t/s outlier) |
| 3 | B_CTX_1 | Coder-Next mlx-lm ctx-128k — 32k/64k done, 128k never collected (might OOM) |
| 4 | G_CTX_1, G_CTX_2 | oMLX ctx-64k/128k — 32k done, long-ctx tiers never collected |
| 5 | Group I | vllm-mlx — never completed, killed mid ctx-128k |
| 6 | Group J | LM Studio — never reached |
| 7 | HTML rebuild | `complete_results.html` from all CSVs |

ETA: ~2.5–3 hours total from 20:06 → expected ~23:00

---

## What is already clean (do NOT re-run)

All CSVs currently in results/ are from clean runs on AC power:

| CSV | Tests | Notes |
|-----|-------|-------|
| bench_20260401_201324.csv | A/B groups baseline | Overnight run |
| bench_20260402_002211.csv | A_Q4_9 (Docker MR) | |
| bench_20260402_005026.csv | A_Q8_1, B_Q4_4/5 | |
| bench_20260402_020645.csv | A+B groups + context | AC, full run |
| bench_20260402_021826.csv | B group context | |
| bench_20260402_042831.csv | C_Q4_1 (Gemma) | |
| bench_20260402_045440.csv | D group | |
| bench_20260402_054209.csv | E_Q4_1 (Llama 70B) | |
| bench_20260402_121221.csv | H_CR_1, H_CTX_1 | From contaminated orchestration run — but H itself ran sequentially and cleanly |
| bench_20260402_121554.csv | A_OLL_CTX (NVFP4) | |
| bench_20260402_125002.csv | F group (TurboQuant) | Clean — battery ~22% at start, decode 89-93 t/s matches baseline |
| bench_20260402_130142.csv | G group (oMLX) | |
| bench_20260402_131505.csv | H group (clean run) | |
| bench_20260402_131931.csv | A_OLL_CTX (NVFP4) | Clean re-run |

## Deleted (tainted by bad USB-C cable)

- ~~bench_20260402_132151.csv~~ — F_TQ4_1/F_TQ35_1 re-run, bad cable
- ~~bench_20260402_132519.csv~~ — A_OLL_CTX_INT4, 44.9 t/s outlier at ctx-32k run 2

---

## After the run: check for anomalies

vllm-mlx context TTFT was showing high variance (82s → 111s → 113s at ctx-64k) during the
tainted run. On clean power, variance should be <10% between runs. If still noisy, that's
vllm-mlx's characteristic behavior, not throttling.

B_CTX_1 ctx-128k: if OOM (connection refused), no row is saved — that's expected.
G_CTX_1/2 ctx-64k/128k: SSD paging — TTFT will be higher; variance expected from NVMe I/O.

---

## Remaining gaps after this run (needs separate work)

| Gap | What's needed |
|-----|--------------|
| MLC-LLM | Not installable via pip/brew on this system; needs mlc.ai wheels or conda |
| mlx-optiq 35B | Pre-built OptiQ models only exist for ≤9B on HF; need to run optiq CLI on 35B |
| Quality benchmarks | See QUALITY_BENCH_DESIGN.md — Phase 2 scope doc |
| Tool calling accuracy | New infrastructure, ~2 days |
| NIAH / long-context quality | New infrastructure, ~1 day |
| Thinking token matrix | New infrastructure, ~1 day |
