#!/usr/bin/env python3
"""Consolidate all benchmark CSVs into a single interactive HTML report."""

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUT_HTML = RESULTS_DIR / "complete_results.html"

NUMERIC = {"ttft_ms", "decode_tps", "prefill_tps", "completion_tokens", "prompt_tokens",
           "total_time_s", "model_load_s", "thinking_tokens", "visible_tokens",
           "cold_ttft_ms", "peak_mem_mb", "peak_cpu_pct"}

MODEL_MAP = {
    "Qwen3.5":          "Qwen3.5-35B-A3B",
    "Coder":            "Qwen3-Coder-Next",
    "Qwen3-Coder-Next": "Qwen3-Coder-Next",
    "Qwen3-Coder":      "Qwen3-Coder-Next",
    "Qwen3-32B":        "Qwen3-32B",
    "Gemma3-27B":       "Gemma3-27B",
    "Gemma4-26B":       "Gemma4-26B-A4B",
    "Gemma4-31B":       "Gemma4-31B",
    "Gemma4-12B":       "Gemma4-12B",
    "Gemma4-E4B":       "Gemma4-E4B",
    "Gemma4-E2B":       "Gemma4-E2B",
    "Llama3.3-70B":     "Llama3.3-70B",
    "Qwen3.6-27B":      "Qwen3.6-27B",
    "Qwen3.6-35B":      "Qwen3.6-35B",
    "Qwen3.6-35B-A3B":  "Qwen3.6-35B-A3B",
    "Qwen3.6":          "Qwen3.6-35B",
    "LFM2.5-8B":        "LFM2.5-8B",
    "LFM2-24B":         "LFM2-24B",
    "Nemotron-Nano":    "Nemotron-Nano-30B",
    "Nemotron-3-Nano":  "Nemotron-Nano-30B",
    "NEM3":             "Nemotron-Nano-30B",
    "Nemotron-Cascade": "Nemotron-Cascade2-30B",
    "NEMC":             "Nemotron-Cascade2-30B",
    "GLM-4.7":          "GLM-4.7-Flash",
    "GLM":              "GLM-4.7-Flash",
    "Granite-4.1":      "Granite-4.1-8B",
    "GR41":             "Granite-4.1-8B",
    "Phi-mini-MoE":     "Phi-mini-MoE",
    "PHIM":             "Phi-mini-MoE",
    "Reka-Flash":       "Reka-Flash-3.1",
    "REKA":             "Reka-Flash-3.1",
    "Falcon3":          "Falcon3-10B",
    "FAL3":             "Falcon3-10B",
    "Laguna":           "Laguna-XS.2",
    "LAG":              "Laguna-XS.2",
    "North-Mini-Code":  "North-Mini-Code",
    "NMC":              "North-Mini-Code",
}

def extract_model(test_name):
    for prefix, model in sorted(MODEL_MAP.items(), key=lambda x: -len(x[0])):
        if test_name.startswith(prefix):
            return model
    return test_name.split()[0] if test_name else "unknown"

def load_all_csvs():
    rows = []
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r.get("test_id") or not r.get("ttft_ms"):
                    continue
                for k in NUMERIC:
                    if k in r and r[k]:
                        try:
                            r[k] = float(r[k])
                        except (ValueError, TypeError):
                            r[k] = 0.0
                    else:
                        r[k] = 0.0
                r["_source"] = csv_path.name
                rows.append(r)
    return rows

def deduplicate(rows):
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        ver = r.get("backend_version", "")
        key = (r["test_id"], r["prompt_type"], ver, r["_source"])
        groups[key].append(r)

    id_prompt_ver = defaultdict(list)
    for (tid, pt, ver, src), rs in groups.items():
        id_prompt_ver[(tid, pt, ver)].append((src, rs))

    results = []
    for (tid, pt, ver), source_groups in id_prompt_ver.items():
        source_groups.sort(key=lambda x: x[0], reverse=True)
        latest_src, latest_rows = source_groups[0]

        median_row = dict(latest_rows[0])
        for k in NUMERIC:
            vals = [r[k] for r in latest_rows if r[k] > 0]
            if vals:
                median_row[k] = statistics.median(vals)
            else:
                median_row[k] = 0.0

        mem_vals = [r["peak_mem_mb"] for r in latest_rows if r["peak_mem_mb"] > 0]
        if mem_vals:
            median_row["peak_mem_mb"] = max(mem_vals)

        results.append(median_row)

    return results

BACKEND_LABELS = {
    "llama-server": "llama.cpp",
    "llama-b8670": "llama.cpp",
    "llama-b8920": "llama.cpp",
    "llama-b9020": "llama.cpp",
    "mlx-lm-0.31.2": "mlx-lm",
    "mlx-lm-0.31.3": "mlx-lm",
    "mlx-vlm-0.4.3": "mlx-vlm",
    "mlx-vlm-0.4.4": "mlx-vlm",
    "omlx-0.3.4": "omlx",
    "dflash": "DFlash",
    "dflash-0.1.0": "DFlash",
    "unsloth-studio": "Unsloth Studio",
}

MODEL_META = {
    "Qwen3-32B":        {"elo": 1342, "aa": 14.5, "released": "2025-04-28", "provider": "Alibaba"},
    "Qwen3.5":          {"elo": 1321, "aa": 30.7, "released": "2026-02-24", "provider": "Alibaba"},
    "Qwen3.5-35B-A3B":  {"elo": 1321, "aa": 30.7, "released": "2026-02-24", "provider": "Alibaba"},
    "Qwen3-Coder-Next": {"elo": 1354, "aa": 28.3, "released": "2026-02-03", "provider": "Alibaba"},
    "Coder":            {"elo": 1354, "aa": 28.3, "released": "2026-02-03", "provider": "Alibaba"},
    "Gemma3-27B":       {"elo": 1358, "aa": 10.3, "released": "2025-03-12", "provider": "Google"},
    "Gemma4-12B":       {"elo": 1335, "aa": 19.5, "released": "2026-06-03", "provider": "Google"},
    "Gemma4-26B-A4B":   {"elo": None, "aa": 27.1, "released": "2026-04-02", "provider": "Google"},
    "Gemma4-31B":       {"elo": None, "aa": 32.3, "released": "2026-04-02", "provider": "Google"},
    "Gemma4-E4B":       {"elo": 1307, "aa": 14.8, "released": "2026-04-03", "provider": "Google"},
    "Gemma4-E2B":       {"elo": None, "aa": 12.1, "released": "2026-04-02", "provider": "Google"},
    "Llama3.3-70B":     {"elo": 1278, "aa": 14.5, "released": "2024-12-06", "provider": "Meta"},
    "Qwen3.6-27B":      {"elo": None, "aa": 37.1, "released": "2026-04-22", "provider": "Alibaba"},
    "Qwen3.6-35B":      {"elo": None, "aa": 31.5, "released": "2026-04-16", "provider": "Alibaba"},
    "Qwen3.6-35B-A3B":  {"elo": None, "aa": 31.5, "released": "2026-04-16", "provider": "Alibaba"},
    "DiffusionGemma":   {"elo": None, "aa": None,  "released": None, "provider": "Google"},
    "NorthCode":        {"elo": None, "aa": 27.6, "released": "2026-06-09", "provider": "Arctic"},
    "Holo3.1":          {"elo": None, "aa": None,  "released": None, "provider": "Holocene"},
    "Qwen3.5-27B":      {"elo": 1321, "aa": 37.1, "released": "2026-02-24", "provider": "Alibaba"},
    "Qwen3.5-122B":     {"elo": None, "aa": 35.9, "released": "2026-02-24", "provider": "Alibaba"},
    "MistralSmall4":    {"elo": 1341, "aa": 27.8, "released": "2026-03-16", "provider": "Mistral"},
    "Devstral2":        {"elo": None, "aa": 22.0, "released": "2025-12-09", "provider": "Mistral"},
    "LFM2.5-8B":        {"elo": None, "aa": None,  "released": "2026-05-24", "provider": "Liquid AI"},
    "LFM2-24B":         {"elo": None, "aa": None,  "released": "2026-02-17", "provider": "Liquid AI"},
    "Nemotron-Nano-30B":     {"elo": None, "aa": 28.5, "released": "2026-05-19", "provider": "NVIDIA"},
    "Nemotron-Cascade2-30B": {"elo": None, "aa": None,  "released": "2026-05-19", "provider": "NVIDIA"},
    "GLM-4.7-Flash":         {"elo": None, "aa": None,  "released": "2026-04-30", "provider": "Zhipu AI"},
    "Granite-4.1-8B":        {"elo": None, "aa": None,  "released": "2026-05-28", "provider": "IBM"},
    "Phi-mini-MoE":          {"elo": None, "aa": None,  "released": "2025-08-12", "provider": "Microsoft"},
    "Reka-Flash-3.1":        {"elo": None, "aa": None,  "released": "2026-04-15", "provider": "Reka"},
    "Falcon3-10B":           {"elo": None, "aa": None,  "released": "2024-12-12", "provider": "TII"},
    "Laguna-XS.2":           {"elo": None, "aa": None,  "released": "2026-05-07", "provider": "Poolside"},
    "North-Mini-Code":       {"elo": None, "aa": 27.6, "released": "2026-06-09", "provider": "Arctic"},
}

def _parse_bench_date(source_filename):
    import re
    m = re.search(r'_(\d{4})(\d{2})(\d{2})_', source_filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None

def build_json_rows(rows):
    json_rows = []
    for r in rows:
        raw_backend = r.get("backend", "")
        model = r.get("model", "") or extract_model(r.get("test_name", ""))
        meta = MODEL_META.get(model, {})
        json_rows.append({
            "id": r.get("test_id", ""),
            "name": r.get("test_name", ""),
            "model": model,
            "provider": meta.get("provider", ""),
            "backend": BACKEND_LABELS.get(raw_backend, raw_backend),
            "ver": r.get("backend_version", ""),
            "fmt": r.get("fmt", ""),
            "quant": r.get("quant", ""),
            "kv": r.get("kv_cache", ""),
            "prompt": r.get("prompt_type", ""),
            "ttft": round(r.get("ttft_ms", 0), 1),
            "cold": round(r.get("cold_ttft_ms", 0), 1),
            "decode": round(r.get("decode_tps", 0), 1),
            "prefill": round(r.get("prefill_tps", 0), 1),
            "tokens": int(r.get("completion_tokens", 0)),
            "total": round(r.get("total_time_s", 0), 2),
            "mem_mb": round(r.get("peak_mem_mb", 0), 0),
            "elo": meta.get("elo"),
            "aa": meta.get("aa"),
            "released": meta.get("released"),
            "bench_date": _parse_bench_date(r.get("_source", "")),
        })

    json_rows.sort(key=lambda r: (r["model"], r["backend"], r["ver"] or "zzz", r["id"], r["prompt"]))
    return json_rows

def generate_html(json_rows, total_raw):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_json = json.dumps(json_rows, indent=None)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llm-bench — Apple M3 Max · 64GB</title>
<script src="https://cdn.jsdelivr.net/npm/ag-grid-community@35.3.1/dist/ag-grid-community.min.js"></script>
<style>
:root {{
  --bg: #09090b; --bg2: #111114; --card: #18181b;
  --border: #27272a; --border2: #3f3f46;
  --text: #fafafa; --text2: #a1a1aa; --dim: #71717a;
  --accent: #818cf8; --accent2: #6366f1;
  --green: #34d399; --red: #f87171; --yellow: #fbbf24; --blue: #60a5fa;
  --nav-bg: rgba(9,9,11,0.88); --nav-text: #a1a1aa;
  --nav-hover: rgba(255,255,255,0.06); --nav-active: #fafafa;
  --shadow: 0 4px 24px rgba(0,0,0,.4);
  --radius: 10px;
}}
html.light {{
  --bg: #fafafa; --bg2: #f4f4f5; --card: #ffffff;
  --border: #e4e4e7; --border2: #d4d4d8;
  --text: #18181b; --text2: #52525b; --dim: #a1a1aa;
  --accent: #6366f1; --accent2: #4f46e5;
  --green: #059669; --red: #dc2626; --yellow: #d97706; --blue: #2563eb;
  --nav-bg: rgba(250,250,250,0.88); --nav-text: #71717a;
  --nav-hover: rgba(0,0,0,0.04); --nav-active: #18181b;
  --shadow: 0 4px 24px rgba(0,0,0,.08);
}}
@media (prefers-color-scheme: light) {{
  html:not(.dark) {{
    --bg: #fafafa; --bg2: #f4f4f5; --card: #ffffff;
    --border: #e4e4e7; --border2: #d4d4d8;
    --text: #18181b; --text2: #52525b; --dim: #a1a1aa;
    --accent: #6366f1; --accent2: #4f46e5;
    --green: #059669; --red: #dc2626; --yellow: #d97706; --blue: #2563eb;
    --nav-bg: rgba(250,250,250,0.88); --nav-text: #71717a;
    --nav-hover: rgba(0,0,0,0.04); --nav-active: #18181b;
    --shadow: 0 4px 24px rgba(0,0,0,.08);
  }}
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', system-ui, sans-serif;
  font-size: 13px; padding: 0;
  -webkit-font-smoothing: antialiased;
}}

/* ── Nav ── */
.site-nav {{
  position: sticky; top: 0; z-index: 100;
  background: var(--nav-bg);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 0 2rem;
  display: flex; align-items: center; height: 48px; gap: 1rem;
}}
.site-nav a {{
  color: var(--nav-text); text-decoration: none; font-size: 13px;
  padding: 5px 12px; border-radius: 6px;
  transition: all .15s ease;
}}
.site-nav a:hover {{ color: var(--nav-active); background: var(--nav-hover); }}
.site-nav .active {{ color: var(--nav-active); background: var(--nav-hover); }}
.site-nav .back {{
  display: flex; align-items: center; gap: 5px;
  margin-right: auto; font-size: 12px; font-weight: 500;
}}
.site-nav .back svg {{ width: 14px; height: 14px; stroke: currentColor; fill: none; }}
.theme-btn {{
  background: var(--nav-hover); border: 1px solid var(--border);
  color: var(--nav-text); border-radius: 6px; padding: 5px 9px;
  cursor: pointer; font-size: 13px; line-height: 1;
  transition: all .15s ease; display: flex; align-items: center;
}}
.theme-btn:hover {{ color: var(--nav-active); border-color: var(--accent); background: var(--nav-hover); }}
body.in-iframe .site-nav {{ display: none; }}

/* ── Page ── */
.page-wrap {{ padding: 1.8rem 2rem 1.5rem; max-width: 100%; }}
.header {{ margin-bottom: 1.5rem; }}
.header h1 {{
  font-size: 1.5rem; font-weight: 700;
  background: linear-gradient(135deg, var(--accent), var(--blue));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin-bottom: .3rem;
}}
.header .meta {{
  color: var(--text2); font-size: .82rem;
  display: flex; flex-wrap: wrap; gap: .3rem 1.2rem;
}}
.header .meta span {{ display: inline-flex; align-items: center; gap: .3rem; }}
.header .meta .dot {{ width: 4px; height: 4px; border-radius: 50%; background: var(--accent); }}

/* ── Controls ── */
.controls {{
  display: flex; flex-wrap: wrap; gap: .5rem;
  align-items: center; margin-bottom: .8rem;
}}
.search-wrap {{
  position: relative; display: flex; align-items: center;
}}
.search-wrap svg {{
  position: absolute; left: 10px; width: 14px; height: 14px;
  stroke: var(--dim); fill: none; pointer-events: none;
}}
.search-wrap input {{
  background: var(--bg2); border: 1px solid var(--border);
  color: var(--text); border-radius: 8px; padding: .45rem .7rem .45rem 32px;
  font-size: 12px; width: 280px; outline: none;
  transition: border-color .15s, box-shadow .15s;
}}
.search-wrap input:focus {{
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(99,102,241,.15);
}}
.btn {{
  background: var(--bg2); border: 1px solid var(--border);
  color: var(--text2); border-radius: 8px; padding: .42rem 1rem;
  font-size: 12px; cursor: pointer; transition: all .15s ease;
  font-weight: 500;
}}
.btn:hover {{ border-color: var(--accent); color: var(--accent); }}
.btn-accent {{
  background: var(--accent); border-color: var(--accent);
  color: #fff;
}}
.btn-accent:hover {{ background: var(--accent2); border-color: var(--accent2); color: #fff; }}
.stats {{
  margin-left: auto; display: flex; gap: 1rem; align-items: center;
}}
.stat {{
  display: flex; flex-direction: column; align-items: center;
  padding: .2rem .8rem; border-radius: 6px;
  background: var(--bg2); border: 1px solid var(--border);
}}
.stat-val {{ font-size: 14px; font-weight: 700; color: var(--accent); line-height: 1.2; }}
.stat-label {{ font-size: 10px; color: var(--dim); text-transform: uppercase; letter-spacing: .04em; }}

/* ── Grid ── */
#grid-container {{
  width: 100%;
  height: calc(100vh - 240px);
  min-height: 450px;
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}}

/* ── AG Grid overrides ── */
.ag-theme-quartz, .ag-theme-quartz-dark {{
  --ag-font-family: -apple-system, BlinkMacSystemFont, 'Inter', system-ui, sans-serif;
  --ag-font-size: 12px;
  --ag-row-height: 34px;
  --ag-header-height: 38px;
  --ag-grid-size: 4px;
}}
.ag-theme-quartz-dark {{
  --ag-background-color: #111114;
  --ag-header-background-color: #18181b;
  --ag-odd-row-background-color: #141417;
  --ag-row-hover-color: #1e1e24;
  --ag-border-color: #27272a;
  --ag-secondary-border-color: #27272a;
  --ag-header-foreground-color: #a1a1aa;
  --ag-foreground-color: #e4e4e7;
}}
.ag-theme-quartz {{
  --ag-background-color: #ffffff;
  --ag-header-background-color: #fafafa;
  --ag-odd-row-background-color: #fafafa;
  --ag-row-hover-color: #f4f4f5;
  --ag-border-color: #e4e4e7;
  --ag-secondary-border-color: #e4e4e7;
  --ag-header-foreground-color: #52525b;
  --ag-foreground-color: #18181b;
}}

/* Custom floating filter select styling */
.custom-select-filter {{
  width: 100%; height: 24px;
  background: var(--bg2); border: 1px solid var(--border);
  color: var(--text); border-radius: 4px;
  font-size: 11px; padding: 0 4px;
  outline: none; cursor: pointer;
}}
.custom-select-filter:focus {{ border-color: var(--accent); }}

/* ── Legend ── */
.legend {{
  margin-top: 1.2rem; color: var(--dim);
  font-size: .75rem; line-height: 1.9;
  padding: 1rem 1.2rem; border-radius: var(--radius);
  background: var(--bg2); border: 1px solid var(--border);
}}
.legend strong {{ color: var(--text2); }}
</style>
</head>
<body>
<script>
if (window !== window.top) document.body.classList.add('in-iframe');
(function() {{
  const html = document.documentElement;
  const stored = localStorage.getItem('bench-theme');
  if (stored === 'light') html.classList.add('light');
  else if (stored === 'dark') html.classList.add('dark');
}})();
</script>

<nav class="site-nav">
  <a href="/projects/llm-benchmark-harness/" class="back">
    <svg viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round"><path d="M19 12H5M12 5l-7 7 7 7"/></svg>
    Project
  </a>
  <a href="/blog">Blog</a>
  <a href="/projects" class="active">Projects</a>
  <a href="/about">About</a>
  <button class="theme-btn" id="theme-toggle" title="Toggle theme">
    <svg id="theme-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
  </button>
</nav>

<div class="page-wrap">
<div class="header">
  <h1>llm-bench</h1>
  <div class="meta">
    <span><span class="dot"></span> Apple M3 Max · 64 GB</span>
    <span>{now}</span>
    <span>{len(json_rows)} configs</span>
    <span>{total_raw} measurements</span>
    <span>median of 3 runs</span>
  </div>
</div>

<div class="controls">
  <div class="search-wrap">
    <svg viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
    <input type="search" id="q" placeholder="Search all columns…">
  </div>
  <button class="btn" id="btn-reset">Reset</button>
  <button class="btn" id="btn-export">Export CSV</button>
  <div class="stats">
    <div class="stat"><span class="stat-val" id="stat-shown">{len(json_rows)}</span><span class="stat-label">shown</span></div>
    <div class="stat"><span class="stat-val">{len(json_rows)}</span><span class="stat-label">configs</span></div>
    <div class="stat"><span class="stat-val">{total_raw}</span><span class="stat-label">runs</span></div>
  </div>
</div>

<div id="grid-container"></div>

<div class="legend">
  <strong>Arena ELO</strong> = LMSYS Chatbot Arena (model-level; higher = smarter) &nbsp;·&nbsp;
  <strong>AA Index</strong> = Artificial Analysis Intelligence Index v4 (0–60, non-reasoning) &nbsp;·&nbsp;
  <strong>TTFT</strong> = time to first token (warm, ms) &nbsp;·&nbsp;
  <strong>Cold</strong> = first-request TTFT &nbsp;·&nbsp;
  <strong>Decode</strong> = generation t/s &nbsp;·&nbsp;
  <strong>Prefill</strong> = prompt eval t/s &nbsp;·&nbsp;
  <strong>Total</strong> = wall-clock seconds &nbsp;·&nbsp;
  <strong>Peak RSS</strong> = max process RAM.
  Median of 3 runs (Peak RSS = max). Q4 ≈ FP16 quality within ~1–3%.
</div>
</div>

<script>
const RAW = {data_json};
const TOTAL = RAW.length;

// ── Theme ──
function getThemeClass() {{
  const html = document.documentElement;
  if (html.classList.contains('light')) return 'light';
  if (html.classList.contains('dark')) return 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}}

function applyGridTheme() {{
  const c = document.getElementById('grid-container');
  c.className = getThemeClass() === 'dark' ? 'ag-theme-quartz-dark' : 'ag-theme-quartz';
}}

document.getElementById('theme-toggle').addEventListener('click', function() {{
  const html = document.documentElement;
  const icon = document.getElementById('theme-icon');
  if (html.classList.contains('light')) {{
    html.classList.remove('light'); html.classList.add('dark');
    localStorage.setItem('bench-theme', 'dark');
    icon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
  }} else if (html.classList.contains('dark')) {{
    html.classList.remove('dark');
    localStorage.removeItem('bench-theme');
    icon.innerHTML = '<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>';
  }} else {{
    html.classList.add('light');
    localStorage.setItem('bench-theme', 'light');
    icon.innerHTML = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
  }}
  applyGridTheme();
}});

(function() {{
  const stored = localStorage.getItem('bench-theme');
  const icon = document.getElementById('theme-icon');
  if (stored === 'dark') icon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
  else if (!stored) icon.innerHTML = '<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>';
}})();

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {{
  if (!localStorage.getItem('bench-theme')) applyGridTheme();
}});

// ── Custom Select Floating Filter (text columns) ──
class SelectFloatingFilter {{
  init(params) {{
    this.params = params;
    this.eGui = document.createElement('div');
    this.eGui.style.width = '100%';
    this.eGui.style.padding = '2px 4px';

    const select = document.createElement('select');
    select.className = 'custom-select-filter';

    const allOpt = document.createElement('option');
    allOpt.value = ''; allOpt.text = 'All';
    select.appendChild(allOpt);

    const field = params.column.getColDef().field;
    const values = [...new Set(RAW.map(r => r[field]).filter(v => v != null && v !== ''))].sort();
    values.forEach(v => {{
      const opt = document.createElement('option');
      opt.value = v; opt.text = v;
      select.appendChild(opt);
    }});

    select.addEventListener('change', () => {{
      if (select.value === '') {{
        params.parentFilterInstance(instance => instance.setModel(null));
      }} else {{
        params.parentFilterInstance(instance => instance.setModel({{
          type: 'equals', filter: select.value
        }}));
      }}
    }});

    this.select = select;
    this.eGui.appendChild(select);
  }}
  getGui() {{ return this.eGui; }}
  onParentModelChanged(model) {{
    this.select.value = model ? (model.filter || '') : '';
  }}
}}

// ── Custom Select Floating Filter (number columns) ──
class SelectNumberFloatingFilter {{
  init(params) {{
    this.params = params;
    this.eGui = document.createElement('div');
    this.eGui.style.width = '100%';
    this.eGui.style.padding = '2px 4px';

    const select = document.createElement('select');
    select.className = 'custom-select-filter';

    const allOpt = document.createElement('option');
    allOpt.value = ''; allOpt.text = 'All';
    select.appendChild(allOpt);

    const field = params.column.getColDef().field;
    const values = [...new Set(RAW.map(r => r[field]).filter(v => v != null && v > 0))].sort((a,b) => a - b);
    values.forEach(v => {{
      const opt = document.createElement('option');
      opt.value = v;
      opt.text = field === 'mem_mb' ? (v >= 1024 ? (v/1024).toFixed(1)+'G' : Math.round(v)+'M')
               : field === 'ttft' || field === 'cold' ? (v >= 1000 ? (v/1000).toFixed(1)+'s' : Math.round(v)+'ms')
               : field === 'total' ? v.toFixed(1)+'s'
               : field === 'aa' ? v.toFixed(1)
               : v;
      select.appendChild(opt);
    }});

    select.addEventListener('change', () => {{
      if (select.value === '') {{
        params.parentFilterInstance(instance => instance.setModel(null));
      }} else {{
        params.parentFilterInstance(instance => instance.setModel({{
          type: 'equals', filter: Number(select.value)
        }}));
      }}
    }});

    this.select = select;
    this.eGui.appendChild(select);
  }}
  getGui() {{ return this.eGui; }}
  onParentModelChanged(model) {{
    this.select.value = model ? String(model.filter || '') : '';
  }}
}}

// ── Decode bar renderer ──
function decodeRenderer(params) {{
  if (!params.value || params.value <= 0) return '<span style="color:var(--dim)">—</span>';
  const maxDecode = Math.max(...RAW.filter(r => r.prompt === params.data.prompt).map(r => r.decode));
  const pct = Math.max(5, Math.round(100 * params.value / maxDecode));
  const hue = Math.round(pct * 1.2);
  return `<div style="display:flex;align-items:center;gap:8px;height:100%">
    <div style="width:70px;height:7px;background:var(--border);border-radius:4px;overflow:hidden;flex-shrink:0">
      <div style="width:${{pct}}%;height:100%;border-radius:4px;background:hsl(${{hue}},60%,48%)"></div>
    </div>
    <span>${{params.value.toFixed(1)}}</span>
  </div>`;
}}

// ── Column definitions ──
const columnDefs = [
  {{ field: 'id', headerName: 'ID', minWidth: 100, pinned: 'left',
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter,
    cellStyle: {{ fontFamily: 'ui-monospace, monospace', fontSize: '11px', color: 'var(--accent)' }} }},
  {{ field: 'model', headerName: 'Model', minWidth: 120,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter,
    cellStyle: {{ fontWeight: 600 }} }},
  {{ field: 'provider', headerName: 'Provider', minWidth: 80,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter }},
  {{ field: 'elo', headerName: 'ELO', minWidth: 95,
    filter: 'agNumberColumnFilter', floatingFilterComponent: SelectNumberFloatingFilter,
    valueFormatter: p => p.value || '—',
    cellStyle: p => p.value ? {{color: 'var(--yellow)', fontWeight: 600}} : {{color: 'var(--dim)'}} }},
  {{ field: 'aa', headerName: 'AA', minWidth: 90,
    filter: 'agNumberColumnFilter', floatingFilterComponent: SelectNumberFloatingFilter,
    valueFormatter: p => p.value ? p.value.toFixed(1) : '—',
    cellStyle: p => p.value ? {{color: 'var(--green)', fontWeight: 600}} : {{color: 'var(--dim)'}} }},
  {{ field: 'prompt', headerName: 'Prompt', minWidth: 75,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter }},
  {{ field: 'backend', headerName: 'Backend', minWidth: 90,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter }},
  {{ field: 'ver', headerName: 'Ver', minWidth: 70,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter,
    valueFormatter: p => p.value || '—', cellStyle: p => !p.value ? {{color: 'var(--dim)'}} : null }},
  {{ field: 'fmt', headerName: 'Fmt', minWidth: 60,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter }},
  {{ field: 'quant', headerName: 'Quant', minWidth: 75,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter }},
  {{ field: 'kv', headerName: 'KV', minWidth: 70,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter }},
  {{ field: 'ttft', headerName: 'TTFT', minWidth: 80,
    filter: 'agNumberColumnFilter', floatingFilterComponent: SelectNumberFloatingFilter,
    valueFormatter: p => p.value >= 1000 ? (p.value/1000).toFixed(1)+'s' : Math.round(p.value)+'ms' }},
  {{ field: 'cold', headerName: 'Cold', minWidth: 75,
    filter: 'agNumberColumnFilter', floatingFilterComponent: SelectNumberFloatingFilter,
    valueFormatter: p => p.value > 0 ? (p.value >= 1000 ? (p.value/1000).toFixed(1)+'s' : Math.round(p.value)+'ms') : '—',
    cellStyle: p => !p.value ? {{color: 'var(--dim)'}} : null }},
  {{ field: 'decode', headerName: 'Decode t/s', minWidth: 145, sort: 'desc',
    filter: 'agNumberColumnFilter', floatingFilterComponent: SelectNumberFloatingFilter,
    cellRenderer: decodeRenderer }},
  {{ field: 'prefill', headerName: 'Prefill', minWidth: 80,
    filter: 'agNumberColumnFilter', floatingFilterComponent: SelectNumberFloatingFilter,
    valueFormatter: p => p.value > 0 ? Math.round(p.value)+' t/s' : '—',
    cellStyle: p => !p.value ? {{color: 'var(--dim)'}} : null }},
  {{ field: 'tokens', headerName: 'Tok', minWidth: 80,
    filter: 'agNumberColumnFilter', floatingFilterComponent: SelectNumberFloatingFilter }},
  {{ field: 'total', headerName: 'Total', minWidth: 70,
    filter: 'agNumberColumnFilter', floatingFilterComponent: SelectNumberFloatingFilter,
    valueFormatter: p => p.value > 0 ? p.value.toFixed(1)+'s' : '—' }},
  {{ field: 'mem_mb', headerName: 'RSS', minWidth: 80,
    filter: 'agNumberColumnFilter', floatingFilterComponent: SelectNumberFloatingFilter,
    valueFormatter: p => p.value > 0 ? (p.value >= 1024 ? (p.value/1024).toFixed(1)+'G' : Math.round(p.value)+'M') : '—',
    cellStyle: p => !p.value ? {{color: 'var(--dim)'}} : null }},
  {{ field: 'released', headerName: 'Released', minWidth: 90,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter,
    valueFormatter: p => p.value || '—', cellStyle: p => !p.value ? {{color: 'var(--dim)'}} : {{color: 'var(--text2)'}} }},
  {{ field: 'bench_date', headerName: 'Benched', minWidth: 90,
    filter: 'agTextColumnFilter', floatingFilterComponent: SelectFloatingFilter,
    valueFormatter: p => p.value || '—', cellStyle: p => !p.value ? {{color: 'var(--dim)'}} : {{color: 'var(--text2)'}} }},
];

// ── Grid init ──
applyGridTheme();
const gridOptions = {{
  columnDefs,
  rowData: RAW,
  defaultColDef: {{
    sortable: true,
    resizable: true,
    floatingFilter: true,
    filterParams: {{ buttons: ['reset'], debounceMs: 150 }},
    suppressHeaderMenuButton: false,
  }},
  autoSizeStrategy: {{ type: 'fitCellContents' }},
  animateRows: true,
  rowSelection: 'single',
  suppressCellFocus: true,
  enableCellTextSelection: true,
  onFilterChanged: updateCount,
  onSortChanged: updateCount,
}};

const gridApi = agGrid.createGrid(document.getElementById('grid-container'), gridOptions);

function updateCount() {{
  const n = gridApi.getDisplayedRowCount();
  document.getElementById('stat-shown').textContent = n;
}}

// Quick filter
document.getElementById('q').addEventListener('input', function(e) {{
  gridApi.setGridOption('quickFilterText', e.target.value);
  updateCount();
}});

// Reset
document.getElementById('btn-reset').addEventListener('click', function() {{
  document.getElementById('q').value = '';
  gridApi.setGridOption('quickFilterText', '');
  gridApi.setFilterModel(null);
  updateCount();
}});

// CSV export
document.getElementById('btn-export').addEventListener('click', function() {{
  gridApi.exportDataAsCsv({{ fileName: 'llm-bench-results.csv' }});
}});
</script>
</body>
</html>'''

def main():
    rows = load_all_csvs()
    total_raw = len(rows)
    print(f"Loaded {total_raw} raw measurements from {len(list(RESULTS_DIR.glob('*.csv')))} CSVs")

    deduped = deduplicate(rows)
    print(f"Deduplicated to {len(deduped)} unique configs (latest source per test_id + prompt)")

    json_rows = build_json_rows(deduped)
    html = generate_html(json_rows, total_raw)

    OUT_HTML.write_text(html)
    print(f"Written: {OUT_HTML} ({len(html):,} bytes, {len(json_rows)} configs)")

if __name__ == "__main__":
    main()
