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

# Map from test_name prefix → clean model name
MODEL_MAP = {
    "Qwen3.5":          "Qwen3.5-35B-A3B",
    "Coder":            "Qwen3-Coder-Next",
    "Qwen3-Coder-Next": "Qwen3-Coder-Next",
    "Qwen3-Coder":      "Qwen3-Coder-Next",
    "Qwen3-32B":        "Qwen3-32B",
    "Gemma3-27B":       "Gemma3-27B",
    "Gemma4-26B":       "Gemma4-26B-A4B",
    "Gemma4-31B":       "Gemma4-31B",
    "Gemma4-E4B":       "Gemma4-E4B",
    "Gemma4-E2B":       "Gemma4-E2B",
    "Llama3.3-70B":     "Llama3.3-70B",
}

def extract_model(test_name):
    """Extract clean model name from test_name."""
    for prefix, model in sorted(MODEL_MAP.items(), key=lambda x: -len(x[0])):
        if test_name.startswith(prefix):
            return model
    return test_name.split()[0] if test_name else "unknown"

def load_all_csvs():
    """Load all CSVs, keeping only rows with valid data."""
    rows = []
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r.get("test_id") or not r.get("ttft_ms"):
                    continue
                # Convert numeric fields
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
    """Group by (test_id, prompt_type), take latest source file per group,
    then compute median of numeric fields across runs."""
    from collections import defaultdict

    # Group by test_id + prompt_type + source
    groups = defaultdict(list)
    for r in rows:
        key = (r["test_id"], r["prompt_type"], r["_source"])
        groups[key].append(r)

    # For each (test_id, prompt_type), pick the latest source file
    id_prompt = defaultdict(list)
    for (tid, pt, src), rs in groups.items():
        id_prompt[(tid, pt)].append((src, rs))

    results = []
    for (tid, pt), source_groups in id_prompt.items():
        # Sort by source filename (timestamp-based) and take the latest
        source_groups.sort(key=lambda x: x[0], reverse=True)
        latest_src, latest_rows = source_groups[0]

        # Compute median across runs
        median_row = dict(latest_rows[0])  # copy first row as template
        for k in NUMERIC:
            vals = [r[k] for r in latest_rows if r[k] > 0]
            if vals:
                median_row[k] = statistics.median(vals)
            else:
                median_row[k] = 0.0

        # Peak memory: take max, not median
        mem_vals = [r["peak_mem_mb"] for r in latest_rows if r["peak_mem_mb"] > 0]
        if mem_vals:
            median_row["peak_mem_mb"] = max(mem_vals)

        results.append(median_row)

    return results

def build_json_rows(rows):
    """Convert to the JSON format expected by the HTML template."""
    json_rows = []
    for r in rows:
        json_rows.append({
            "id": r.get("test_id", ""),
            "name": r.get("test_name", ""),
            "model": extract_model(r.get("test_name", "")),
            "backend": r.get("backend", ""),
            "fmt": r.get("fmt", ""),
            "quant": r.get("quant", ""),
            "kv": r.get("kv_cache", ""),
            "prompt": r.get("prompt_type", ""),
            "ttft": r.get("ttft_ms", 0),
            "cold": r.get("cold_ttft_ms", 0),
            "decode": r.get("decode_tps", 0),
            "prefill": r.get("prefill_tps", 0),
            "tokens": int(r.get("completion_tokens", 0)),
            "total": r.get("total_time_s", 0),
            "mem_mb": r.get("peak_mem_mb", 0),
            "tool": r.get("tool_call_valid", ""),
            "quality": r.get("quality_pass", ""),
        })

    # Sort: by model, then backend, then id, then prompt
    json_rows.sort(key=lambda r: (r["model"], r["backend"], r["id"], r["prompt"]))
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
<style>
:root {{
  --bg: #0f1117; --card: #1a1d27; --card2: #1e2133;
  --border: #2d3148; --text: #e2e4f0; --dim: #7b7f9e;
  --green: #4ade80; --red: #f87171; --accent: #6366f1;
  --yellow: #fbbf24; --sort-arrow: #6366f1;
  --th-bg: #22253a; --row-hover: #232640; --bar-track: #2a2e48;
  --nav-bg: rgba(28,25,23,0.85); --nav-text: #a8a29e;
  --nav-hover: rgba(255,255,255,0.08); --nav-active: #fff;
  --shadow: #0008;
}}
@media (prefers-color-scheme: light) {{
  :root {{
    --bg: #f5f5f4; --card: #ffffff; --card2: #fafaf9;
    --border: #d6d3d1; --text: #1c1917; --dim: #78716c;
    --green: #16a34a; --red: #dc2626; --accent: #4f46e5;
    --yellow: #d97706; --sort-arrow: #4f46e5;
    --th-bg: #f5f5f4; --row-hover: #fafaf9; --bar-track: #e7e5e4;
    --nav-bg: rgba(245,245,244,0.85); --nav-text: #57534e;
    --nav-hover: rgba(0,0,0,0.05); --nav-active: #1c1917;
    --shadow: #0002;
  }}
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg); color: var(--text);
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 13px; padding: 1.5rem 2rem; min-width: 900px;
}}
h1 {{ font-size: 1.3rem; color: var(--accent); margin-bottom: .2rem; }}
.meta {{ color: var(--dim); font-size: .82rem; margin-bottom: 1.2rem; }}

/* ── Controls ── */
.controls {{
  display: flex; flex-wrap: wrap; gap: .5rem;
  align-items: center; margin-bottom: .8rem;
}}
.controls input[type=search] {{
  background: var(--card2); border: 1px solid var(--border);
  color: var(--text); border-radius: 5px; padding: .35rem .6rem;
  font-size: 12px; width: 180px; outline: none;
}}
.controls input[type=search]:focus {{ border-color: var(--accent); }}
.controls select {{
  background: var(--card2); border: 1px solid var(--border);
  color: var(--text); border-radius: 5px; padding: .33rem .5rem;
  font-size: 12px; outline: none; cursor: pointer;
}}
.controls select:focus {{ border-color: var(--accent); }}
.btn-reset {{
  background: transparent; border: 1px solid var(--border);
  color: var(--dim); border-radius: 5px; padding: .33rem .8rem;
  font-size: 12px; cursor: pointer; transition: .15s;
}}
.btn-reset:hover {{ border-color: var(--accent); color: var(--accent); }}
.count {{ color: var(--dim); font-size: 11px; margin-left: auto; }}

/* ── Table ── */
.table-wrap {{ overflow-x: auto; border-radius: 8px; box-shadow: 0 2px 16px var(--shadow); }}
table {{
  width: 100%; border-collapse: collapse;
  background: var(--card); white-space: nowrap;
}}
thead {{ position: sticky; top: 0; z-index: 2; }}
th {{
  background: var(--th-bg); color: var(--dim);
  font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
  padding: .55rem .75rem; text-align: left; cursor: pointer;
  user-select: none; border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
th:hover {{ color: var(--text); }}
th.sort-asc  .arrow::after {{ content: " ▲"; color: var(--sort-arrow); font-size: .75em; }}
th.sort-desc .arrow::after {{ content: " ▼"; color: var(--sort-arrow); font-size: .75em; }}
th .arrow {{ display: inline; }}
td {{
  padding: .42rem .75rem; border-bottom: 1px solid var(--border);
  vertical-align: middle;
}}
tr:last-child td {{ border-bottom: none; }}
tbody tr:hover td {{ background: var(--row-hover); }}
tbody tr.hidden {{ display: none; }}
code {{ font-family: ui-monospace, monospace; font-size: .82em; color: var(--accent); }}

/* ── Bar ── */
.bar-cell {{
  display: flex; align-items: center; gap: .5rem;
  min-width: 160px;
}}
.bar-track {{
  width: 80px; flex-shrink: 0;
  height: 8px; background: var(--bar-track); border-radius: 4px; overflow: hidden;
}}
.bar-fill {{ height: 100%; border-radius: 4px; }}
.bar-val {{ color: var(--text); }}

/* ── Badges ── */
.pass {{ color: var(--green); font-weight: 600; }}
.fail {{ color: var(--red); }}
.dim  {{ color: var(--dim); }}

/* ── Legend ── */
.legend {{
  margin-top: 1rem; color: var(--dim);
  font-size: .76rem; line-height: 1.8; max-width: 900px;
}}

/* ── Site nav (hidden when embedded in iframe) ── */
.site-nav {{
  position: sticky; top: 0; z-index: 100;
  background: var(--nav-bg);
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--border);
  padding: 0 2rem; margin: -1.5rem -2rem 1.5rem -2rem;
  display: flex; align-items: center; height: 48px; gap: 1.5rem;
}}
.site-nav a {{
  color: var(--nav-text); text-decoration: none; font-size: 13px;
  padding: 4px 10px; border-radius: 6px;
  transition: color .15s, background .15s;
}}
.site-nav a:hover {{ color: var(--nav-active); background: var(--nav-hover); }}
.site-nav .active {{ color: var(--nav-active); background: var(--nav-hover); }}
.site-nav .back {{
  display: flex; align-items: center; gap: 4px;
  margin-right: auto; font-size: 12px;
}}
.site-nav .back svg {{ width: 14px; height: 14px; stroke: currentColor; fill: none; }}
body.in-iframe .site-nav {{ display: none; }}
body.in-iframe {{ padding-top: .5rem; }}
</style>
</head>
<body>
<script>if (window !== window.top) document.body.classList.add('in-iframe');</script>

<nav class="site-nav">
  <a href="/projects/llm-benchmark-harness/" class="back">
    <svg viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round"><path d="M19 12H5M12 5l-7 7 7 7"/></svg>
    Back to project
  </a>
  <a href="/blog">Blog</a>
  <a href="/projects" class="active">Projects</a>
  <a href="/about">About</a>
</nav>

<h1>llm-bench results</h1>
<p class="meta">Apple M3 Max · 64 GB &nbsp;·&nbsp; {now} &nbsp;·&nbsp; {len(json_rows)} configs (median of 3 runs) &nbsp;·&nbsp; {total_raw} total measurements</p>

<div class="controls">
  <input type="search" id="q" placeholder="Search ID or name…">
  <select id="f-prompt"><option value="">All prompts</option></select>
  <select id="f-backend"><option value="">All backends</option></select>
  <select id="f-model"><option value="">All models</option></select>
  <select id="f-fmt"><option value="">All formats</option></select>
  <select id="f-quant"><option value="">All quants</option></select>
  <select id="f-kv"><option value="">All KV</option></select>
  <button class="btn-reset" onclick="reset()">Reset</button>
  <span class="count" id="count"></span>
</div>

<div class="table-wrap">
<table id="tbl">
<thead>
<tr>
  <th data-col="id"      data-type="str"><span class="arrow">ID</span></th>
  <th data-col="name"    data-type="str"><span class="arrow">Name</span></th>
  <th data-col="model"   data-type="str"><span class="arrow">Model</span></th>
  <th data-col="prompt"  data-type="str"><span class="arrow">Prompt</span></th>
  <th data-col="backend" data-type="str"><span class="arrow">Backend</span></th>
  <th data-col="fmt"     data-type="str"><span class="arrow">Fmt</span></th>
  <th data-col="quant"   data-type="str"><span class="arrow">Quant</span></th>
  <th data-col="kv"      data-type="str"><span class="arrow">KV Cache</span></th>
  <th data-col="ttft"    data-type="num"><span class="arrow">TTFT</span></th>
  <th data-col="decode"  data-type="num"><span class="arrow">Decode</span></th>
  <th data-col="prefill" data-type="num"><span class="arrow">Prefill</span></th>
  <th data-col="tokens"  data-type="num"><span class="arrow">Tokens</span></th>
  <th data-col="total"   data-type="num"><span class="arrow">Total</span></th>
  <th data-col="mem_mb"  data-type="num"><span class="arrow">Peak RSS</span></th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>

<p class="legend">
  <strong>TTFT</strong> = time to first token (warm, with prefix cache where available) &nbsp;·&nbsp;
  <strong>Cold</strong> = first-request TTFT (shown in parentheses when &gt;1.5× warm) &nbsp;·&nbsp;
  <strong>Decode</strong> = generation tokens/s &nbsp;·&nbsp;
  <strong>Prefill</strong> = prompt eval tokens/s &nbsp;·&nbsp;
  <strong>Total</strong> = wall-clock time for full response &nbsp;·&nbsp;
  <strong>Peak RSS</strong> = process tree RAM during inference.
  All values are median of 3 runs except Peak RSS (max).
  Backends: mlx-lm 0.31.2, mlx-vlm 0.4.3/0.4.4, Ollama 0.19/0.20, oMLX 0.3.4, llama.cpp b5220/b8670, vllm-mlx 0.1, LM Studio, Docker Model Runner.
</p>

<script>
const RAW = {data_json};

const promptRange = {{}};
for (const r of RAW) {{
  if (!promptRange[r.prompt]) promptRange[r.prompt] = {{min: Infinity, max: -Infinity}};
  promptRange[r.prompt].max = Math.max(promptRange[r.prompt].max, r.decode);
  promptRange[r.prompt].min = Math.min(promptRange[r.prompt].min, r.decode);
}}

function barHtml(decode, prompt) {{
  const range = promptRange[prompt] || {{min: 0, max: 1}};
  const span = range.max - range.min || 1;
  const pct = Math.max(4, Math.round(100 * (decode - range.min) / span));
  const hue = Math.round(pct);
  const color = `hsl(${{hue}},65%,45%)`;
  return `<div class="bar-cell">
    <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
    <span class="bar-val">${{decode.toFixed(1)}} t/s</span>
  </div>`;
}}

function fmtTTFT(r) {{
  const cold = r.cold > r.ttft * 1.5 && r.cold > 500
    ? ` <span class="dim">(c:${{(r.cold/1000).toFixed(1)}}s)</span>` : '';
  if (r.ttft >= 1000) return `${{(r.ttft/1000).toFixed(1)}}s${{cold}}`;
  return `${{r.ttft.toFixed(0)}} ms${{cold}}`;
}}

function fmtMem(mb) {{
  if (!mb) return '<span class="dim">&mdash;</span>';
  return mb >= 1024 ? `${{(mb/1024).toFixed(1)}} GB` : `${{mb.toFixed(0)}} MB`;
}}

function renderRows(data) {{
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = data.map(r => `
<tr data-id="${{r.id}}" data-name="${{r.name}}" data-prompt="${{r.prompt}}"
    data-model="${{r.model}}" data-backend="${{r.backend}}" data-fmt="${{r.fmt}}" data-quant="${{r.quant}}" data-kv="${{r.kv}}">
  <td><code>${{r.id}}</code></td>
  <td>${{r.name}}</td>
  <td>${{r.model}}</td>
  <td>${{r.prompt}}</td>
  <td>${{r.backend}}</td>
  <td>${{r.fmt}}</td>
  <td>${{r.quant}}</td>
  <td>${{r.kv}}</td>
  <td>${{fmtTTFT(r)}}</td>
  <td>${{barHtml(r.decode, r.prompt)}}</td>
  <td>${{r.prefill > 0 ? r.prefill.toFixed(0) + ' t/s' : '<span class="dim">&mdash;</span>'}}</td>
  <td>${{r.tokens}}</td>
  <td>${{r.total.toFixed(1)}} s</td>
  <td>${{fmtMem(r.mem_mb)}}</td>
</tr>`).join('');
  document.getElementById('count').textContent = `${{data.length}} / ${{RAW.length}} configs`;
}}

function populateSelect(id, key) {{
  const sel = document.getElementById(id);
  const vals = [...new Set(RAW.map(r => r[key]))].sort();
  vals.forEach(v => {{ const o = document.createElement('option'); o.value = o.text = v; sel.appendChild(o); }});
}}
populateSelect('f-model',   'model');
populateSelect('f-prompt',  'prompt');
populateSelect('f-backend', 'backend');
populateSelect('f-fmt',     'fmt');
populateSelect('f-quant',   'quant');
populateSelect('f-kv',      'kv');

let sortCol = null, sortDir = 1;

function getFiltered() {{
  const q       = document.getElementById('q').value.toLowerCase();
  const model   = document.getElementById('f-model').value;
  const prompt  = document.getElementById('f-prompt').value;
  const backend = document.getElementById('f-backend').value;
  const fmt     = document.getElementById('f-fmt').value;
  const quant   = document.getElementById('f-quant').value;
  const kv      = document.getElementById('f-kv').value;

  return RAW.filter(r =>
    (!q       || r.id.toLowerCase().includes(q) || r.name.toLowerCase().includes(q)) &&
    (!model   || r.model   === model)  &&
    (!prompt  || r.prompt  === prompt)  &&
    (!backend || r.backend === backend) &&
    (!fmt     || r.fmt     === fmt)     &&
    (!quant   || r.quant   === quant)   &&
    (!kv      || r.kv      === kv)
  );
}}

function applySort(data) {{
  if (!sortCol) return data;
  return [...data].sort((a, b) => {{
    const av = a[sortCol], bv = b[sortCol];
    if (typeof av === 'number') return sortDir * (av - bv);
    return sortDir * String(av).localeCompare(String(bv));
  }});
}}

function update() {{
  renderRows(applySort(getFiltered()));
}}

document.querySelectorAll('th[data-col]').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if (sortCol === col) {{ sortDir *= -1; }}
    else {{ sortCol = col; sortDir = 1; }}
    document.querySelectorAll('th').forEach(t => t.classList.remove('sort-asc', 'sort-desc'));
    th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
    update();
  }});
}});

['q','f-model','f-prompt','f-backend','f-fmt','f-quant','f-kv'].forEach(id =>
  document.getElementById(id).addEventListener('input', update));

function reset() {{
  document.getElementById('q').value = '';
  ['f-model','f-prompt','f-backend','f-fmt','f-quant','f-kv'].forEach(id =>
    document.getElementById(id).value = '');
  sortCol = null; sortDir = 1;
  document.querySelectorAll('th').forEach(t => t.classList.remove('sort-asc', 'sort-desc'));
  update();
}}

update();
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
