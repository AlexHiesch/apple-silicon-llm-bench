#!/usr/bin/env python3
"""
llm-bench — Local LLM Hosting Benchmark
========================================
Apples-to-apples comparison across backends, formats and quantizations.
All test definitions live in config.yaml — no code changes required.

Backends: mlx-lm, mlx-vlm, llama-server (llama.cpp), Ollama,
          LM Studio, Docker Model Runner, vllm-mlx, omlx

Usage:
  python3 benchmark.py                        # Run all enabled tests
  python3 benchmark.py --list                 # Show test matrix + readiness
  python3 benchmark.py --test A_Q4_1 A_Q4_5  # Run specific test IDs
  python3 benchmark.py --group "A"            # Run all tests in a group
  python3 benchmark.py --runs 5               # Override iteration count
  python3 benchmark.py --prompt short         # Only short prompt
  python3 benchmark.py --prompt code          # Realistic 2 K-token coding prompt
  python3 benchmark.py --prompt context-8k    # Context scaling test
  python3 benchmark.py --max-tokens 256       # Override generation length
  python3 benchmark.py --no-think             # Disable thinking for Qwen3
  python3 benchmark.py --test-tools           # Tool calling compatibility check
  python3 benchmark.py --test-quality         # Output code quality check
  python3 benchmark.py --report-html          # Also save an HTML report
  python3 benchmark.py --config my.yaml       # Use a custom config file
  python3 benchmark.py --skip-unavailable     # Skip tests with unmet prereqs
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from urllib.error import URLError
from urllib.request import Request, urlopen

import psutil
import yaml

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"
RESULTS_DIR = Path(__file__).parent / "results"

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
GGUF_CACHE = Path.home() / ".cache" / "llmfit" / "models"
VLLM_MLX_VENV = Path.home() / ".local" / "share" / "vllm-mlx-bench" / ".venv"
HF_PY = "/opt/homebrew/Cellar/mlx-lm/0.31.1/libexec/bin/python"

# Port constants (overridable in config)
OLLAMA_PORT = 11434
LM_STUDIO_PORT = 1234
DMR_PORT = 12434
DMR_SOCKET = Path.home() / "Library/Containers/com.docker.docker/Data/inference.sock"

# ANSI colours
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_CYAN = "\033[36m"

# ── Terminal helpers ───────────────────────────────────────────────────────────


def info(msg):  print(f"  {C_CYAN}▸{C_RESET} {msg}")
def ok(msg):    print(f"  {C_GREEN}✔{C_RESET} {msg}")
def warn(msg):  print(f"  {C_YELLOW}⚠{C_RESET} {msg}")
def err(msg):   print(f"  {C_RED}✘{C_RESET} {msg}", file=sys.stderr)


# ── Resource Monitor ──────────────────────────────────────────────────────────

class ResourceMonitor:
    """Poll a process tree for peak RSS and CPU usage."""

    def __init__(self, pid: int, interval: float = 0.2):
        self._pid = pid
        self._interval = interval
        self._peak_mem_bytes = 0
        self._peak_cpu_pct = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _sample(self):
        try:
            proc = psutil.Process(self._pid)
            children = proc.children(recursive=True)
            mem = sum(p.memory_info().rss for p in [proc] + children)
            cpu = sum(p.cpu_percent(interval=0) for p in [proc] + children)
            self._peak_mem_bytes = max(self._peak_mem_bytes, mem)
            self._peak_cpu_pct = max(self._peak_cpu_pct, cpu)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _poll_loop(self):
        self._sample()
        while self._running:
            time.sleep(self._interval)
            self._sample()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def reset(self):
        self._peak_mem_bytes = 0
        self._peak_cpu_pct = 0.0
        self._sample()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        return self.peak_mem_mb, self.peak_cpu_pct

    @property
    def peak_mem_mb(self) -> float:
        return round(self._peak_mem_bytes / (1024 * 1024), 1)

    @property
    def peak_cpu_pct(self) -> float:
        return round(self._peak_cpu_pct, 1)


def _find_server_pid(backend: str, proc) -> Optional[int]:
    if isinstance(proc, subprocess.Popen):
        return proc.pid
    if backend == "ollama":
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = p.info["name"] or ""
                cmd = " ".join(p.info.get("cmdline") or [])
                if "ollama_llama_server" in name or "ollama_llama_server" in cmd:
                    return p.info["pid"]
                if name == "ollama" and "serve" in cmd:
                    return p.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return None


# ── Built-in Test Prompts ─────────────────────────────────────────────────────

BUILTIN_PROMPTS: dict[str, Union[str, list[dict]]] = {
    "short": (
        "Write a Python function that checks if a number is prime. "
        "Include a docstring and type hints."
    ),
    "long": (
        "Explain the complete lifecycle of an HTTP request from the moment "
        "a user types a URL in their browser until the page is fully rendered. "
        "Cover DNS resolution, TCP handshake, TLS negotiation, HTTP request "
        "and response, server-side processing, and browser rendering pipeline "
        "including DOM construction, CSSOM, layout, painting, and compositing. "
        "Include details about caching at each layer, connection pooling, and "
        "HTTP/2 multiplexing. Provide concrete examples of headers involved "
        "at each stage and explain how each step can be optimized for performance."
    ),
    "code": [
        {
            "role": "system",
            "content": (
                "You are a senior Python developer. You write clean, well-tested, "
                "production-quality code. When asked to refactor, you preserve all "
                "existing behavior while improving structure and readability. "
                "Always include type hints and docstrings."
            ),
        },
        {
            "role": "user",
            "content": '''\
I have this data pipeline module. Please refactor it to use the Strategy pattern
for the transform steps — extract each transform into its own class with a common
interface, and make it easy to add new transforms without modifying the pipeline.
Preserve all existing behavior.

```python
"""Data pipeline for processing sales records."""
import csv, json, logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

@dataclass
class SalesRecord:
    transaction_id: str
    timestamp: datetime
    customer_id: str
    product_id: str
    product_name: str
    category: str
    quantity: int
    unit_price: Decimal
    discount_pct: Decimal
    tax_rate: Decimal
    region: str
    store_id: str
    payment_method: str
    is_return: bool = False
    notes: Optional[str] = None

    @property
    def subtotal(self) -> Decimal:
        return (self.unit_price * self.quantity).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @property
    def discount_amount(self) -> Decimal:
        return (self.subtotal * self.discount_pct / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @property
    def taxable_amount(self) -> Decimal:
        return self.subtotal - self.discount_amount

    @property
    def tax_amount(self) -> Decimal:
        return (self.taxable_amount * self.tax_rate / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @property
    def total(self) -> Decimal:
        sign = Decimal("-1") if self.is_return else Decimal("1")
        return sign * (self.taxable_amount + self.tax_amount)


class SalesPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.min_amount = Decimal(str(config.get("min_amount", "0")))
        self.max_amount = Decimal(str(config.get("max_amount", "999999")))

    def transform_currency(self, record):
        rate = Decimal(str(self.config.get("currency_rate", "1.0")))
        if rate != 1:
            record.unit_price = (record.unit_price * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return record

    def transform_anonymize(self, record):
        import hashlib
        record.customer_id = hashlib.sha256(record.customer_id.encode()).hexdigest()[:16]
        return record

    def transform_categorize(self, record):
        total = abs(record.total)
        if total >= 1000:
            record.notes = (record.notes or "") + " [TIER:premium]"
        elif total >= 100:
            record.notes = (record.notes or "") + " [TIER:standard]"
        else:
            record.notes = (record.notes or "") + " [TIER:basic]"
        return record
```''',
        },
    ],
}

# ── Context Scaling Prompts ───────────────────────────────────────────────────

_CODE_BLOCK = '''\
def process_{name}(data: list[dict], config: dict) -> list[dict]:
    """Process {name} records with configured validation and enrichment."""
    validated, seen_keys = [], set()
    lookup = config.get("{name}_lookup", {{}})
    for i, record in enumerate(data):
        key = record.get("id", f"unknown_{{i}}")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if not isinstance(record.get("value"), (int, float)):
            continue
        if record["value"] < config.get("min_{name}", 0):
            continue
        normalized = {{
            "id": key, "value": round(float(record["value"]), 4),
            "category": record.get("category", "unknown").lower().strip(),
            "timestamp": record.get("timestamp", ""),
            "source": "{name}",
            "metadata": {{"enriched": key in lookup, "lookup_data": lookup.get(key, {{}})}},
        }}
        validated.append(normalized)
    validated.sort(key=lambda r: r["timestamp"])
    return validated

'''

_BLOCK_NAMES = [
    "inventory", "orders", "shipments", "payments", "returns",
    "customers", "products", "categories", "warehouses", "suppliers",
    "invoices", "receipts", "transfers", "adjustments", "allocations",
]

CONTEXT_SYSTEM = (
    "You are a senior Python developer. You write clean, well-tested, "
    "production-quality code. When asked to refactor, you preserve all "
    "existing behavior while improving structure and readability."
)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def generate_context_prompt(target_tokens: int) -> list[dict]:
    system = {"role": "system", "content": CONTEXT_SYSTEM}
    instruction = (
        "\n\nRefactor the above module to eliminate the repetitive function pattern. "
        "Extract a common base using a class or higher-order function, keeping all "
        "per-domain configuration parameterized. Preserve all existing behavior."
    )
    blocks = []
    current = estimate_tokens(CONTEXT_SYSTEM + instruction)
    suffix = 1
    while current < target_tokens:
        for name in _BLOCK_NAMES:
            label = name if suffix == 1 else f"{name}_v{suffix}"
            block = _CODE_BLOCK.format(name=label)
            tok = estimate_tokens(block)
            if current + tok > target_tokens:
                break
            blocks.append(block)
            current += tok
        else:
            suffix += 1
            continue
        break

    user = "```python\n" + "\n".join(blocks) + "```" + instruction
    return [system, {"role": "user", "content": user}]


CONTEXT_TIERS = {
    "context-4k":   4096,
    "context-8k":   8192,
    "context-16k":  16384,
    "context-32k":  32768,
    "context-64k":  65536,
    "context-128k": 131072,
    "context-256k": 262144,
}

# ── Tool Calling Test ─────────────────────────────────────────────────────────

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute file path to read"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

TOOL_TEST_PROMPT = [
    {"role": "system", "content": "You are a coding assistant with access to tools. Use the appropriate tool to fulfill the user's request."},
    {"role": "user", "content": "Read the contents of the file /etc/hostname and tell me what it says."},
]

# ── Quality Spot-Check ────────────────────────────────────────────────────────

QUALITY_PROMPT = (
    "Write a Python function called `fib` that returns the nth Fibonacci number. "
    "fib(0) should return 0, fib(1) should return 1, fib(10) should return 55. "
    "Return ONLY the function definition, no explanation, no test code."
)
QUALITY_TESTS = "assert fib(10)==55\nassert fib(0)==0\nassert fib(1)==1"


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class TestConfig:
    id: str
    name: str
    group: str
    backend: str
    model_id: str
    fmt: str
    quant: str
    kv_cache: str
    extra_args: list
    port: int
    prereq: str
    no_think_override: Optional[bool] = None  # None = follow global flag
    prompts: Optional[list] = None            # per-test prompt override (list of prompt names)


@dataclass
class BenchResult:
    test_id: str
    test_name: str
    backend: str
    fmt: str
    quant: str
    kv_cache: str
    prompt_type: str
    run_num: int
    ttft_ms: float
    decode_tps: float
    prefill_tps: float
    completion_tokens: int
    prompt_tokens: int
    total_time_s: float
    model_load_s: float
    thinking_tokens: int = 0
    visible_tokens: int = 0
    cold_ttft_ms: float = 0
    peak_mem_mb: float = 0
    peak_cpu_pct: float = 0
    tool_call_valid: Optional[bool] = None
    quality_pass: Optional[bool] = None


# ── Config loading + prereq auto-detection ────────────────────────────────────

def _ollama_gguf_blob(model_tag: str) -> str:
    try:
        out = subprocess.check_output(
            ["ollama", "show", model_tag, "--modelfile"],
            stderr=subprocess.STDOUT, text=True, timeout=10,
        )
        for line in out.splitlines():
            if line.startswith("FROM ") and ".ollama" in line:
                return line.split("FROM ", 1)[1].strip()
    except Exception:
        pass
    return ""


def _resolve_model_id(model_id: str, backend: str) -> tuple[str, str]:
    """Resolve special model_id prefixes. Returns (resolved_id, prereq)."""
    if backend == "llama-server" and model_id.startswith("ollama:"):
        tag = model_id[7:]
        blob = _ollama_gguf_blob(tag)
        if blob:
            return blob, ""
        return "", f"ollama pull {tag}"
    if model_id.startswith("~"):
        resolved = str(Path(model_id).expanduser())
        if not Path(resolved).exists():
            return resolved, f"GGUF not found: {resolved}"
        return resolved, ""
    return model_id, ""


def hf_model_cached(repo_id: str) -> bool:
    short = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    p = HF_CACHE / f"models--{repo_id.replace('/', '--')}"
    if not p.exists():
        # Try mlx-community prefix
        p2 = HF_CACHE / f"models--mlx-community--{short}"
        if not p2.exists():
            return False
        p = p2
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total > 1_000_000_000


def ollama_has_model(tag: str) -> bool:
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        return tag in out
    except Exception:
        return False


def ollama_version() -> tuple[int, ...]:
    try:
        out = subprocess.check_output(["ollama", "--version"], stderr=subprocess.STDOUT, text=True)
        for part in out.strip().split():
            if part[0].isdigit():
                return tuple(int(x) for x in part.split(".")[:3])
    except Exception:
        pass
    return (0, 0, 0)


def lm_studio_running() -> bool:
    try:
        with urlopen(f"http://localhost:{LM_STUDIO_PORT}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def dmr_has_model(model_tag: str) -> bool:
    try:
        out = subprocess.check_output(["docker", "model", "list"], text=True, timeout=10)
        return model_tag.lower().replace("ai/", "").split(":")[0] in out.lower()
    except Exception:
        return False


def vllm_mlx_installed() -> bool:
    return (VLLM_MLX_VENV / "bin" / "vllm-mlx").exists()


def omlx_installed() -> bool:
    try:
        subprocess.check_output(["omlx", "-h"], stderr=subprocess.STDOUT, timeout=5)
        return True
    except Exception:
        return False


def _auto_prereq(cfg_entry: dict, ollama_ver: tuple[int, ...]) -> str:
    """Return empty string if test is ready, else a human-readable description."""
    backend = cfg_entry["backend"]
    model_id = cfg_entry["model_id"]
    is_019 = ollama_ver >= (0, 19, 0)

    if backend in ("mlx-lm", "mlx-vlm", "mlx-lm-turboquant"):
        if backend == "mlx-lm-turboquant":
            try:
                result = subprocess.run(
                    [HF_PY, "-c", "import turboquant"],
                    capture_output=True, timeout=5
                )
                if result.returncode != 0:
                    return "turboquant-mlx not installed (run: install_turboquant.sh)"
            except Exception:
                return "turboquant-mlx not installed"
        return "" if hf_model_cached(model_id) else f"Model not in HF cache: {model_id}"

    if backend == "ollama":
        tag = model_id
        need_019 = "nvfp4" in tag or "int4" in tag
        has_model = ollama_has_model(tag)
        needs = []
        if need_019 and not is_019:
            needs.append("brew upgrade ollama (need ≥0.19)")
        if not has_model:
            needs.append(f"ollama pull {tag}")
        return " && ".join(needs)

    if backend == "llama-server":
        if model_id.startswith("ollama:"):
            tag = model_id[7:]
            return "" if _ollama_gguf_blob(tag) else f"ollama pull {tag}"
        path = Path(model_id).expanduser()
        return "" if path.exists() else f"GGUF not found: {path}"

    if backend == "vllm-mlx":
        if not vllm_mlx_installed():
            return "vllm-mlx not installed"
        return "" if hf_model_cached(model_id) else f"Model not in HF cache: {model_id}"

    if backend == "omlx":
        if not omlx_installed():
            return "oMLX not installed (brew install omlx)"
        model_dir = Path.home() / ".omlx" / "models" / model_id
        if not (model_dir / "config.json").exists():
            return f"oMLX model not found at {model_dir}"
        return ""

    if backend == "lm-studio":
        return "" if lm_studio_running() else "Start LM Studio app (or run: lms server start)"

    if backend == "docker-model-runner":
        short = model_id.lower().replace("docker.io/ai/", "").split(":")[0]
        return "" if dmr_has_model(short) else f"docker model pull {model_id}"

    return ""


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_tests(cfg: dict, bench_port: int) -> list[TestConfig]:
    """Build test list from config, auto-detecting prereqs."""
    ollama_ver = ollama_version()
    tests = []
    custom_prompts = cfg.get("custom_prompts", {})

    for entry in cfg.get("tests", []):
        if not entry.get("enabled", True):
            continue

        backend = entry["backend"]
        raw_model_id = entry["model_id"]
        resolved_model_id, resolve_prereq = _resolve_model_id(raw_model_id, backend)

        # Manual prereq override takes priority, then auto-detect
        manual_prereq = entry.get("prereq", None)
        if manual_prereq is not None:
            prereq = manual_prereq
        elif resolve_prereq:
            prereq = resolve_prereq
        else:
            prereq = _auto_prereq({**entry, "model_id": resolved_model_id}, ollama_ver)

        port = {
            "ollama": OLLAMA_PORT,
            "lm-studio": LM_STUDIO_PORT,
            "docker-model-runner": DMR_PORT,
        }.get(backend, bench_port)

        no_think_override = entry.get("no_think", None)

        tests.append(TestConfig(
            id=entry["id"],
            name=entry["name"],
            group=entry.get("group", "default"),
            backend=backend,
            model_id=resolved_model_id or raw_model_id,
            fmt=entry.get("fmt", ""),
            quant=entry.get("quant", ""),
            kv_cache=entry.get("kv_cache", "default"),
            extra_args=entry.get("extra_args", []),
            port=port,
            prereq=prereq,
            no_think_override=no_think_override,
            prompts=entry.get("prompts") or None,
        ))

    return tests


def resolve_prompts(prompt_arg: str, custom_prompts: dict) -> list[tuple[str, list[dict]]]:
    """Resolve prompt name(s) to (name, messages) pairs."""
    def to_messages(v):
        if isinstance(v, str):
            return [{"role": "user", "content": v}]
        return v

    if prompt_arg == "all":
        return [(n, to_messages(BUILTIN_PROMPTS[n])) for n in ("short", "long", "code")]
    if prompt_arg == "context":
        return [(n, generate_context_prompt(t)) for n, t in CONTEXT_TIERS.items()]
    if prompt_arg in CONTEXT_TIERS:
        return [(prompt_arg, generate_context_prompt(CONTEXT_TIERS[prompt_arg]))]
    if prompt_arg in BUILTIN_PROMPTS:
        return [(prompt_arg, to_messages(BUILTIN_PROMPTS[prompt_arg]))]
    if prompt_arg in custom_prompts:
        entry = custom_prompts[prompt_arg]
        msgs = entry.get("messages") or [{"role": entry.get("type", "user"), "content": entry.get("content", "")}]
        return [(prompt_arg, msgs)]
    raise ValueError(f"Unknown prompt: '{prompt_arg}'. "
                     f"Built-in: {list(BUILTIN_PROMPTS)} + {list(CONTEXT_TIERS)}. "
                     f"Custom: {list(custom_prompts)}")


# ── Server Management ──────────────────────────────────────────────────────────

def kill_port(port: int):
    try:
        pids = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        for pid in pids.split("\n"):
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
        time.sleep(2)
    except subprocess.CalledProcessError:
        pass


def wait_for_server(port: int, timeout: int, path: str = "/v1/models") -> bool:
    deadline = time.time() + timeout
    url = f"http://localhost:{port}{path}"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def wait_for_ollama(timeout: int = 30) -> bool:
    return wait_for_server(OLLAMA_PORT, timeout, "/api/tags")


def _popen_server(cmd: list, label: str) -> subprocess.Popen:
    info(f"Starting {label}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=open("/tmp/llm-bench-server.log", "w"), stderr=subprocess.STDOUT)


def start_server(test: TestConfig, server_timeout: int):
    if test.backend in ("mlx-lm",):
        kill_port(test.port)
        time.sleep(1)
        cmd = ["mlx_lm.server", "--model", test.model_id,
               "--port", str(test.port), "--host", "127.0.0.1"] + test.extra_args
        return _popen_server(cmd, "mlx-lm")

    if test.backend == "mlx-vlm":
        kill_port(test.port)
        time.sleep(1)
        cmd = ["mlx_vlm.server", "--model", test.model_id,
               "--port", str(test.port), "--host", "127.0.0.1"] + test.extra_args
        return _popen_server(cmd, "mlx-vlm")

    if test.backend == "mlx-lm-turboquant":
        kill_port(test.port)
        time.sleep(1)
        script = Path(__file__).parent / "turboquant_server.py"
        cmd = [HF_PY, str(script),
               "--model", test.model_id,
               "--port", str(test.port), "--host", "127.0.0.1"] + test.extra_args
        return _popen_server(cmd, "TurboQuant")

    if test.backend == "llama-server":
        kill_port(test.port)
        time.sleep(1)
        cmd = ["llama-server", "-m", test.model_id,
               "--port", str(test.port), "--host", "127.0.0.1",
               "-ngl", "99"] + test.extra_args
        return _popen_server(cmd, "llama-server")

    if test.backend == "vllm-mlx":
        kill_port(test.port)
        time.sleep(1)
        bin_ = VLLM_MLX_VENV / "bin" / "vllm-mlx"
        cmd = [str(bin_), "serve", test.model_id,
               "--host", "127.0.0.1", "--port", str(test.port)] + test.extra_args
        return _popen_server(cmd, "vllm-mlx")

    if test.backend == "omlx":
        kill_port(test.port)
        time.sleep(1)
        model_dir = Path.home() / ".omlx" / "models"
        cache_dir = Path.home() / ".omlx" / "cache"
        cmd = ["omlx", "serve",
               "--model-dir", str(model_dir),
               "--port", str(test.port), "--host", "127.0.0.1",
               "--paged-ssd-cache-dir", str(cache_dir),
               "--hot-cache-max-size", "8GB"] + test.extra_args
        return _popen_server(cmd, "oMLX")

    if test.backend == "ollama":
        info(f"Ollama — preloading {test.model_id}")
        if not wait_for_ollama(10):
            subprocess.Popen(["ollama", "serve"],
                             stdout=open("/tmp/llm-bench-ollama.log", "w"),
                             stderr=subprocess.STDOUT)
            if not wait_for_ollama(30):
                err("Could not start Ollama")
                return None
        payload = json.dumps({"model": test.model_id,
                               "messages": [{"role": "user", "content": "hi"}],
                               "stream": False, "options": {"num_predict": 1}}).encode()
        req = Request(f"http://localhost:{OLLAMA_PORT}/api/chat", data=payload,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=server_timeout) as r:
                r.read()
        except Exception as e:
            err(f"Failed to preload model: {e}")
            return None
        return "ollama"

    if test.backend == "lm-studio":
        info(f"LM Studio — ensuring server is up")
        lms = Path.home() / ".lmstudio" / "bin" / "lms"
        if not lm_studio_running():
            if lms.exists():
                subprocess.run([str(lms), "server", "start"], timeout=30, capture_output=True)
                time.sleep(2)
            if not lm_studio_running():
                err("LM Studio server not running")
                return None
        if lms.exists():
            try:
                subprocess.run([str(lms), "load", test.model_id, "--context-length", "32768"],
                               timeout=120, capture_output=True)
            except Exception:
                pass
        return "lm-studio"

    if test.backend == "docker-model-runner":
        info("Docker Model Runner — checking proxy")
        if not _check_dmr():
            return None
        # Preload
        payload = json.dumps({"model": test.model_id,
                               "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 1, "stream": False}).encode()
        req = Request(f"http://localhost:{DMR_PORT}/v1/chat/completions", data=payload,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=server_timeout) as r:
                r.read()
        except Exception as e:
            err(f"Failed to preload DMR model: {e}")
            return None
        return "docker-model-runner"

    err(f"Unknown backend: {test.backend}")
    return None


def _check_dmr() -> bool:
    def _dmr_up():
        try:
            with urlopen(f"http://localhost:{DMR_PORT}/v1/models", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    if _dmr_up():
        return True
    if not DMR_SOCKET.exists():
        err("Docker Model Runner socket not found — is Docker Desktop running?")
        return False
    try:
        proc = subprocess.Popen(
            ["socat", f"TCP-LISTEN:{DMR_PORT},reuseaddr,fork", f"UNIX-CONNECT:{DMR_SOCKET}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        return _dmr_up()
    except FileNotFoundError:
        err("socat not installed — brew install socat")
        return False


def stop_server(proc, test: TestConfig):
    if test.backend == "ollama":
        try:
            payload = json.dumps({"model": test.model_id, "keep_alive": 0}).encode()
            req = Request(f"http://localhost:{OLLAMA_PORT}/api/generate", data=payload,
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=10) as r:
                r.read()
        except Exception:
            pass
        return
    if test.backend == "lm-studio":
        lms = Path.home() / ".lmstudio" / "bin" / "lms"
        if lms.exists():
            try:
                subprocess.run([str(lms), "unload", test.model_id], timeout=10, capture_output=True)
            except Exception:
                pass
        return
    if test.backend == "docker-model-runner":
        return
    if proc and isinstance(proc, subprocess.Popen):
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    kill_port(test.port)


# ── Benchmark Requests ────────────────────────────────────────────────────────

def count_thinking_tokens(text: str) -> tuple[int, int]:
    blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    think = "".join(blocks)
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return len(think.split()) if think else 0, len(visible.split()) if visible else 0


def bench_openai_streaming(port: int, messages: list[dict], max_tokens: int,
                            model_id: str = "default", no_think: bool = False) -> dict:
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if no_think:
        body["extra_body"] = {"enable_thinking": False}
        body["chat_template_kwargs"] = {"enable_thinking": False}

    payload = json.dumps(body).encode()
    req = Request(f"http://localhost:{port}/v1/chat/completions", data=payload,
                  headers={"Content-Type": "application/json"})

    t_start = time.perf_counter()
    t_first = None
    token_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    full_text = ""

    try:
        with urlopen(req, timeout=600) as resp:
            buf = ""
            for raw in resp:
                buf += raw.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line or line == "data: [DONE]" or not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    usage = data.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if content:
                        full_text += content
                        if t_first is None:
                            t_first = time.perf_counter()
                        token_count += 1
    except Exception as e:
        err(f"Request failed: {e}")
        return {}

    t_end = time.perf_counter()
    if t_first is None:
        err("No tokens received")
        return {}

    total_time = t_end - t_start
    ttft = (t_first - t_start) * 1000
    decode_time = t_end - t_first
    if completion_tokens == 0:
        completion_tokens = token_count
    if prompt_tokens == 0:
        prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

    decode_tps = (completion_tokens - 1) / decode_time if decode_time > 0 and completion_tokens > 1 else 0
    prefill_tps = prompt_tokens / (ttft / 1000) if prompt_tokens > 0 and ttft > 0 else 0
    think_tok, vis_tok = count_thinking_tokens(full_text)

    return {
        "ttft_ms": round(ttft, 1),
        "decode_tps": round(decode_tps, 1),
        "prefill_tps": round(prefill_tps, 1),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_time_s": round(total_time, 2),
        "thinking_tokens": think_tok,
        "visible_tokens": vis_tok,
    }


def bench_ollama_streaming(model: str, messages: list[dict], max_tokens: int,
                            no_think: bool = False) -> dict:
    body = {"model": model, "messages": messages, "stream": True,
            "options": {"temperature": 0.0, "num_predict": max_tokens}}
    if no_think:
        body["think"] = False
    payload = json.dumps(body).encode()
    req = Request(f"http://localhost:{OLLAMA_PORT}/api/chat", data=payload,
                  headers={"Content-Type": "application/json"})

    t_start = time.perf_counter()
    t_first = None
    token_count = 0
    eval_count = eval_dur = prompt_eval_count = prompt_eval_dur = 0
    full_text = thinking_text = ""

    try:
        with urlopen(req, timeout=600) as resp:
            for raw in resp:
                try:
                    data = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                msg = data.get("message", {})
                content = msg.get("content", "")
                thinking = msg.get("thinking", "")
                if content or thinking:
                    full_text += content
                    thinking_text += thinking
                    if t_first is None:
                        t_first = time.perf_counter()
                    token_count += 1
                if data.get("done"):
                    eval_count = data.get("eval_count", 0)
                    eval_dur = data.get("eval_duration", 0)
                    prompt_eval_count = data.get("prompt_eval_count", 0)
                    prompt_eval_dur = data.get("prompt_eval_duration", 0)
    except Exception as e:
        err(f"Request failed: {e}")
        return {}

    t_end = time.perf_counter()
    if t_first is None:
        err("No tokens received")
        return {}

    ttft = (t_first - t_start) * 1000
    decode_tps = eval_count / (eval_dur / 1e9) if eval_count and eval_dur else \
                 (token_count - 1) / (t_end - t_first) if token_count > 1 else 0
    prefill_tps = prompt_eval_count / (prompt_eval_dur / 1e9) if prompt_eval_count and prompt_eval_dur else 0

    if thinking_text:
        think_tok = estimate_tokens(thinking_text)
        vis_tok = estimate_tokens(full_text)
    else:
        think_tok, vis_tok = count_thinking_tokens(full_text)

    return {
        "ttft_ms": round(ttft, 1),
        "decode_tps": round(decode_tps, 1),
        "prefill_tps": round(prefill_tps, 1),
        "completion_tokens": eval_count or token_count,
        "prompt_tokens": prompt_eval_count,
        "total_time_s": round(t_end - t_start, 2),
        "thinking_tokens": think_tok,
        "visible_tokens": vis_tok,
    }


def run_single_bench(test: TestConfig, messages: list[dict], max_tokens: int,
                     no_think: bool) -> dict:
    if test.backend == "ollama":
        return bench_ollama_streaming(test.model_id, messages, max_tokens, no_think=no_think)
    return bench_openai_streaming(test.port, messages, max_tokens,
                                  model_id=test.model_id, no_think=no_think)


# ── Tool + Quality Tests ──────────────────────────────────────────────────────

def run_tool_test(test: TestConfig, no_think: bool) -> dict:
    if test.backend == "ollama":
        body = {"model": test.model_id, "messages": TOOL_TEST_PROMPT, "tools": TOOL_SCHEMA,
                "stream": False, "options": {"temperature": 0.0, "num_predict": 256}}
        if no_think:
            body["think"] = False
        url = f"http://localhost:{OLLAMA_PORT}/api/chat"
    else:
        body = {"model": test.model_id, "messages": TOOL_TEST_PROMPT, "tools": TOOL_SCHEMA,
                "tool_choice": "auto", "max_tokens": 256, "temperature": 0.0, "stream": False}
        if no_think:
            body["extra_body"] = {"enable_thinking": False}
            body["chat_template_kwargs"] = {"enable_thinking": False}
        url = f"http://localhost:{test.port}/v1/chat/completions"

    payload = json.dumps(body).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    t_start = time.perf_counter()
    try:
        with urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        return {"tool_call_valid": False, "error": str(e), "tool_call_latency_ms": 0}

    latency = (time.perf_counter() - t_start) * 1000

    try:
        if test.backend == "ollama":
            tool_calls = data.get("message", {}).get("tool_calls", [])
        else:
            tool_calls = data["choices"][0]["message"].get("tool_calls", [])

        if not tool_calls:
            content = (data.get("message") or data["choices"][0]["message"]).get("content", "")
            return {"tool_call_valid": False,
                    "error": f"No tool_calls (got text: {content[:80]})",
                    "tool_call_latency_ms": round(latency, 1)}

        tc = tool_calls[0]
        fn = tc.get("function", {})
        fn_name = fn.get("name", "")
        fn_args_raw = fn.get("arguments", "{}")
        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
        return {
            "tool_call_valid": True,
            "tool_name": fn_name,
            "tool_name_correct": fn_name == "read_file",
            "tool_args": fn_args,
            "tool_call_latency_ms": round(latency, 1),
        }
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {"tool_call_valid": False, "error": str(e),
                "tool_call_latency_ms": round(latency, 1)}


def run_quality_test(test: TestConfig, no_think: bool) -> dict:
    messages = [{"role": "user", "content": QUALITY_PROMPT}]
    if test.backend == "ollama":
        body = {"model": test.model_id, "messages": messages, "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512}}
        if no_think:
            body["think"] = False
        url = f"http://localhost:{OLLAMA_PORT}/api/chat"
    else:
        body = {"model": test.model_id, "messages": messages,
                "max_tokens": 512, "temperature": 0.0, "stream": False}
        if no_think:
            body["extra_body"] = {"enable_thinking": False}
            body["chat_template_kwargs"] = {"enable_thinking": False}
        url = f"http://localhost:{test.port}/v1/chat/completions"

    payload = json.dumps(body).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
        if test.backend == "ollama":
            msg = data.get("message", {})
            text = msg.get("content", "")
            if not text and msg.get("thinking", ""):
                return {"quality_pass": False, "error": "All tokens spent on thinking"}
        else:
            text = data["choices"][0]["message"]["content"]
    except Exception as e:
        return {"quality_pass": False, "error": str(e)}

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = m.group(1).strip() if m else text.strip()

    namespace = {}
    try:
        old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(10)
        try:
            exec(code, namespace)
            exec(QUALITY_TESTS, namespace)
            quality_pass, error = True, None
        except Exception as e:
            quality_pass, error = False, str(e)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    except Exception as e:
        quality_pass, error = False, str(e)

    return {"quality_pass": quality_pass, "output_length": len(text),
            "output_preview": text[:200], "error": error}


# ── Results Formatting ────────────────────────────────────────────────────────

BACKEND_LABELS = {
    "mlx-lm": "mlx-lm",
    "mlx-vlm": "mlx-vlm",
    "mlx-lm-turboquant": "TurboQuant",
    "ollama": "Ollama",
    "llama-server": "llama-server",
    "vllm-mlx": "vllm-mlx",
    "omlx": "oMLX",
    "lm-studio": "LM Studio",
    "docker-model-runner": "Docker MR",
}


def _sort_id(tid: str):
    m = re.match(r"([A-Za-z]+)_([A-Za-z]+)(\d+)_(\d+)", tid)
    if m:
        return (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
    return (tid, "", 0, 0)


def print_results_table(results: list, hardware: dict,
                        tool_results: dict = None, quality_results: dict = None):
    if not results:
        print("\nNo results.")
        return

    hw = hardware.get("name", "")
    mem = hardware.get("memory_gb", "")
    hw_str = f"{hw} {mem}GB" if mem else hw

    print(f"\n{'═' * 108}")
    print(f"  {C_BOLD}BENCHMARK RESULTS{C_RESET}  ·  {C_DIM}{hw_str}  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}{C_RESET}")
    print(f"{'═' * 108}")

    any_think = any(r.thinking_tokens > 0 for r in results)
    any_mem = any(r.peak_mem_mb > 0 for r in results)

    from collections import defaultdict
    by_prompt = defaultdict(list)
    for r in results:
        by_prompt[r.prompt_type].append(r)

    for prompt_type in sorted(by_prompt):
        subset = by_prompt[prompt_type]
        print(f"\n  {C_BOLD}Prompt: {prompt_type}{C_RESET}")
        print(f"  {'─' * 104}")

        col_w = 52 + (10 if any_think else 0) + (10 if any_mem else 0)
        hdr = (f"  {'ID':<12} {'Name':<28} {'Fmt':<6} {'Quant':<8} {'KV':<12} "
               f"{'TTFT':>8} {'Decode':>10} {'Prefill':>10} "
               f"{'Tokens':>{'14' if any_think else '7'}} {'Total':>7}")
        if any_mem:
            hdr += f"  {'PeakRSS':>9}"
        print(hdr)
        print(f"  {'─' * 104}")

        by_test: dict[str, list] = defaultdict(list)
        for r in subset:
            by_test[r.test_id].append(r)

        current_group = ""
        sorted_ids = sorted(by_test.keys(), key=_sort_id)

        # Print group headers by looking up the group from the first result
        # (group info not stored in BenchResult — use test_id prefix heuristic)
        # We store group inside test_name via the data we have
        for tid in sorted_ids:
            runs = by_test[tid]
            r0 = runs[0]

            # Aggregate stats (median)
            mid = len(runs) // 2
            med_ttft = sorted(r.ttft_ms for r in runs)[mid]
            med_dec = sorted(r.decode_tps for r in runs)[mid]
            med_pre = sorted(r.prefill_tps for r in runs)[mid]
            med_tot = sorted(r.total_time_s for r in runs)[mid]
            avg_tok = sum(r.completion_tokens for r in runs) // len(runs)
            avg_think = sum(r.thinking_tokens for r in runs) // len(runs)
            avg_vis = sum(r.visible_tokens for r in runs) // len(runs)

            cold = runs[0].cold_ttft_ms
            cold_s = f" (c:{cold:.0f})" if cold > 0 and cold > med_ttft * 2 else ""

            pre_s = f"{med_pre:>8.0f} t/s" if med_pre > 0 else f"{'—':>10}"
            tok_s = f"{avg_tok:>4}t ({avg_vis}v)" if any_think and avg_think > 0 else f"{avg_tok:>5}t"
            mem_s = f"  {max(r.peak_mem_mb for r in runs):>7.0f}MB" if any_mem else ""

            blabel = BACKEND_LABELS.get(r0.backend, r0.backend)
            print(f"  {tid:<12} {r0.test_name:<28.27} {r0.fmt:<6} {r0.quant:<8} {r0.kv_cache:<12} "
                  f"{med_ttft:>6.0f}ms{cold_s:<8} {med_dec:>8.1f}t/s {pre_s} "
                  f"{tok_s:>{'14' if any_think else '7'}} {med_tot:>5.1f}s{mem_s}")

    print(f"\n{'═' * 108}")
    print(f"  {C_DIM}Metrics: median of runs · TTFT=time to first token · "
          f"Decode=gen tok/s · Prefill=prompt eval tok/s · c:=cold TTFT{C_RESET}")
    if any_mem:
        print(f"  {C_DIM}PeakRSS: process tree RSS. "
              f"mmap backends (Ollama, llama-server) may underreport; "
              f"MLX backends reflect Metal buffer usage.{C_RESET}")
    if any_think:
        print(f"  {C_DIM}Tokens: total (visible). Thinking = total − visible.{C_RESET}")

    if tool_results or quality_results:
        print(f"\n  {C_BOLD}TOOL CALLING & QUALITY{C_RESET}")
        print(f"  {'─' * 60}")
        for tid in sorted(set(list(tool_results or {}) + list(quality_results or {})), key=_sort_id):
            parts = [f"  {tid:<12}"]
            tr = (tool_results or {}).get(tid)
            if tr:
                if tr.get("tool_call_valid"):
                    parts.append(f"Tool: {C_GREEN}PASS{C_RESET} ({tr.get('tool_name','?')}, "
                                 f"{tr.get('tool_call_latency_ms',0):.0f}ms)")
                else:
                    parts.append(f"Tool: {C_RED}FAIL{C_RESET} ({tr.get('error','?')[:50]})")
            qr = (quality_results or {}).get(tid)
            if qr:
                if qr.get("quality_pass"):
                    parts.append(f"Quality: {C_GREEN}PASS{C_RESET}")
                else:
                    parts.append(f"Quality: {C_RED}FAIL{C_RESET} ({qr.get('error','?')[:40]})")
            print("  ".join(parts))
    print()


def save_results_csv(results: list, path: Path):
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchResult.__dataclass_fields__)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    ok(f"CSV saved → {path}")


def save_results_html(results: list, hardware: dict, path: Path,
                      tool_results: dict = None, quality_results: dict = None):
    """Generate a self-contained HTML report with sortable/filterable columns."""
    if not results:
        return

    from collections import defaultdict

    hw = hardware.get("name", "Unknown")
    mem = hardware.get("memory_gb", "")
    hw_str = f"{hw} · {mem}GB" if mem else hw
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Compute per-(test_id, prompt_type) medians
    by_tp: dict[tuple, list] = defaultdict(list)
    for r in results:
        by_tp[(r.test_id, r.prompt_type)].append(r)

    rows_data = []  # list of dicts — one per (test_id, prompt_type)
    for (tid, pt), runs in sorted(by_tp.items(), key=lambda x: (_sort_id(x[0][0]), x[0][1])):
        mid = len(runs) // 2
        r0 = runs[0]
        ttft    = sorted(r.ttft_ms        for r in runs)[mid]
        decode  = sorted(r.decode_tps     for r in runs)[mid]
        prefill = sorted(r.prefill_tps    for r in runs)[mid]
        total   = sorted(r.total_time_s   for r in runs)[mid]
        tokens  = sorted(r.completion_tokens for r in runs)[mid]
        mem_mb  = max(r.peak_mem_mb for r in runs)
        cold    = runs[0].cold_ttft_ms or 0

        tr = (tool_results or {}).get(tid)
        qr = (quality_results or {}).get(tid)

        rows_data.append({
            "id": tid,
            "name": r0.test_name,
            "backend": BACKEND_LABELS.get(r0.backend, r0.backend),
            "fmt": r0.fmt,
            "quant": r0.quant,
            "kv": r0.kv_cache,
            "prompt": pt,
            "ttft": round(ttft, 1),
            "cold": round(cold, 1),
            "decode": round(decode, 1),
            "prefill": round(prefill, 1),
            "tokens": tokens,
            "total": round(total, 2),
            "mem_mb": round(mem_mb, 0),
            "tool": "pass" if (tr or {}).get("tool_call_valid") else ("fail" if tr else ""),
            "quality": "pass" if (qr or {}).get("quality_pass") else ("fail" if qr else ""),
        })

    rows_json = json.dumps(rows_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llm-bench — {hw_str}</title>
<style>
:root {{
  --bg: #0f1117; --card: #1a1d27; --card2: #1e2133;
  --border: #2d3148; --text: #e2e4f0; --dim: #7b7f9e;
  --green: #4ade80; --red: #f87171; --accent: #6366f1;
  --yellow: #fbbf24; --sort-arrow: #6366f1;
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
.table-wrap {{ overflow-x: auto; border-radius: 8px; box-shadow: 0 2px 16px #0008; }}
table {{
  width: 100%; border-collapse: collapse;
  background: var(--card); white-space: nowrap;
}}
thead {{ position: sticky; top: 0; z-index: 2; }}
th {{
  background: #22253a; color: var(--dim);
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
tbody tr:hover td {{ background: #232640; }}
tbody tr.hidden {{ display: none; }}
code {{ font-family: ui-monospace, monospace; font-size: .82em; color: var(--accent); }}

/* ── Bar ── */
.bar-cell {{
  display: flex; align-items: center; gap: .5rem;
  min-width: 160px;
}}
.bar-track {{
  width: 80px; flex-shrink: 0;
  height: 8px; background: #2a2e48; border-radius: 4px; overflow: hidden;
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
</style>
</head>
<body>
<h1>llm-bench results</h1>
<p class="meta">{hw_str} &nbsp;·&nbsp; {ts} &nbsp;·&nbsp; median of 3 runs per test</p>

<div class="controls">
  <input type="search" id="q" placeholder="Search ID or name…">
  <select id="f-prompt"><option value="">All prompts</option></select>
  <select id="f-backend"><option value="">All backends</option></select>
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
  <th data-col="prompt"  data-type="str"><span class="arrow">Prompt</span></th>
  <th data-col="backend" data-type="str"><span class="arrow">Backend</span></th>
  <th data-col="fmt"     data-type="str"><span class="arrow">Fmt</span></th>
  <th data-col="quant"   data-type="str"><span class="arrow">Quant</span></th>
  <th data-col="kv"      data-type="str"><span class="arrow">KV Cache</span></th>
  <th data-col="ttft"    data-type="num"><span class="arrow">TTFT ↓</span></th>
  <th data-col="decode"  data-type="num"><span class="arrow">Decode ↑</span></th>
  <th data-col="prefill" data-type="num"><span class="arrow">Prefill ↑</span></th>
  <th data-col="tokens"  data-type="num"><span class="arrow">Tokens</span></th>
  <th data-col="total"   data-type="num"><span class="arrow">Total ↓</span></th>
  <th data-col="mem_mb"  data-type="num"><span class="arrow">Peak RSS</span></th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>

<p class="legend">
  <strong>TTFT</strong> = time to first token &nbsp;·&nbsp;
  <strong>Decode</strong> = generation tokens/s (↑ higher = faster) &nbsp;·&nbsp;
  <strong>Prefill</strong> = prompt eval tokens/s &nbsp;·&nbsp;
  <strong>Total</strong> = wall-clock time for full response &nbsp;·&nbsp;
  <strong>Peak RSS</strong> = process tree RAM during inference
  (MLX reflects Metal buffer allocations; mmap backends underreport).
</p>

<script>
const RAW = {rows_json};

// ── compute per-prompt decode range for bar scaling ──────────────────────
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
  const hue = Math.round(pct);          // 0 = red, 100 = green
  const color = `hsl(${{hue}},65%,45%)`;
  return `<div class="bar-cell">
    <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
    <span class="bar-val">${{decode.toFixed(1)}} t/s</span>
  </div>`;
}}

function fmtTTFT(r) {{
  const cold = r.cold > r.ttft * 1.5 && r.cold > 500
    ? ` <span class="dim">(c:${{(r.cold/1000).toFixed(1)}}s)</span>` : '';
  return `${{r.ttft.toFixed(0)}} ms${{cold}}`;
}}

function fmtMem(mb) {{
  if (!mb) return '<span class="dim">—</span>';
  return mb >= 1024 ? `${{(mb/1024).toFixed(1)}} GB` : `${{mb.toFixed(0)}} MB`;
}}

// ── render rows ──────────────────────────────────────────────────────────
function renderRows(data) {{
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = data.map(r => `
<tr data-id="${{r.id}}" data-name="${{r.name}}" data-prompt="${{r.prompt}}"
    data-backend="${{r.backend}}" data-fmt="${{r.fmt}}" data-quant="${{r.quant}}" data-kv="${{r.kv}}">
  <td><code>${{r.id}}</code></td>
  <td>${{r.name}}</td>
  <td>${{r.prompt}}</td>
  <td>${{r.backend}}</td>
  <td>${{r.fmt}}</td>
  <td>${{r.quant}}</td>
  <td>${{r.kv}}</td>
  <td>${{fmtTTFT(r)}}</td>
  <td>${{barHtml(r.decode, r.prompt)}}</td>
  <td>${{r.prefill > 0 ? r.prefill.toFixed(0) + ' t/s' : '<span class="dim">—</span>'}}</td>
  <td>${{r.tokens}}</td>
  <td>${{r.total.toFixed(1)}} s</td>
  <td>${{fmtMem(r.mem_mb)}}</td>
</tr>`).join('');
  document.getElementById('count').textContent = `${{data.length}} / ${{RAW.length}} rows`;
}}

// ── populate filter dropdowns ────────────────────────────────────────────
function populateSelect(id, key) {{
  const sel = document.getElementById(id);
  const vals = [...new Set(RAW.map(r => r[key]))].sort();
  vals.forEach(v => {{ const o = document.createElement('option'); o.value = o.text = v; sel.appendChild(o); }});
}}
populateSelect('f-prompt',  'prompt');
populateSelect('f-backend', 'backend');
populateSelect('f-fmt',     'fmt');
populateSelect('f-quant',   'quant');
populateSelect('f-kv',      'kv');

// ── filter + sort state ──────────────────────────────────────────────────
let sortCol = null, sortDir = 1;

function getFiltered() {{
  const q       = document.getElementById('q').value.toLowerCase();
  const prompt  = document.getElementById('f-prompt').value;
  const backend = document.getElementById('f-backend').value;
  const fmt     = document.getElementById('f-fmt').value;
  const quant   = document.getElementById('f-quant').value;
  const kv      = document.getElementById('f-kv').value;

  return RAW.filter(r =>
    (!q       || r.id.toLowerCase().includes(q) || r.name.toLowerCase().includes(q)) &&
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

// ── sorting ──────────────────────────────────────────────────────────────
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

// ── filter events ────────────────────────────────────────────────────────
['q','f-prompt','f-backend','f-fmt','f-quant','f-kv'].forEach(id =>
  document.getElementById(id).addEventListener('input', update));

function reset() {{
  document.getElementById('q').value = '';
  ['f-prompt','f-backend','f-fmt','f-quant','f-kv'].forEach(id =>
    document.getElementById(id).value = '');
  sortCol = null; sortDir = 1;
  document.querySelectorAll('th').forEach(t => t.classList.remove('sort-asc', 'sort-desc'));
  update();
}}

// ── initial render ───────────────────────────────────────────────────────
update();
</script>
</body>
</html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    ok(f"HTML report → {path}")


def print_test_list(tests: list):
    print(f"\n  {C_BOLD}llm-bench — Test Matrix{C_RESET}")
    print(f"  {'─' * 90}")
    print(f"  {'ID':<14} {'Backend':<15} {'Fmt':<7} {'Quant':<8} {'KV':<12} {'Status'}")
    print(f"  {'─' * 90}")

    current_group = ""
    for t in tests:
        if t.group != current_group:
            current_group = t.group
            print(f"\n  {C_BOLD}{t.group}{C_RESET}")
        status = (f"{C_GREEN}✔ Ready{C_RESET}" if not t.prereq
                  else f"{C_YELLOW}⚠ {t.prereq}{C_RESET}")
        blabel = BACKEND_LABELS.get(t.backend, t.backend)
        print(f"  {t.id:<14} {blabel:<15} {t.fmt:<7} {t.quant:<8} {t.kv_cache:<12} {status}")

    ready = sum(1 for t in tests if not t.prereq)
    print(f"\n  {ready}/{len(tests)} tests ready\n")


# ── Main ───────────────────────────────────────────────────────────────────────

PROMPT_CHOICES = ["short", "long", "code", "context-4k", "context-8k",
                  "context-16k", "context-32k", "context-64k", "context-128k",
                  "context", "all"]


def main():
    parser = argparse.ArgumentParser(description="llm-bench — Local LLM Benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--list", action="store_true", help="Show test matrix and exit")
    parser.add_argument("--test", nargs="+", metavar="ID", help="Run specific test IDs")
    parser.add_argument("--group", metavar="GROUP", help="Run all tests in a group")
    parser.add_argument("--runs", type=int, help="Override benchmark iterations")
    parser.add_argument("--warmup", type=int, help="Override warmup iterations")
    parser.add_argument("--max-tokens", type=int, help="Override max tokens to generate")
    parser.add_argument("--prompt", choices=PROMPT_CHOICES + ["custom"],
                        help="Which prompt(s) to test (default: from config). "
                             "'context' runs all context tiers; 'all' runs short+long+code.")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking tokens for Qwen3 models")
    parser.add_argument("--test-tools", action="store_true",
                        help="Run tool-calling compatibility test per config")
    parser.add_argument("--test-quality", action="store_true",
                        help="Run code quality spot-check (fib function) per config")
    parser.add_argument("--report-html", action="store_true",
                        help="Also save an HTML report alongside the CSV")
    parser.add_argument("--skip-unavailable", action="store_true",
                        help="Silently skip tests with unmet prerequisites")
    args = parser.parse_args()

    if not args.config.exists():
        err(f"Config not found: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)
    s = cfg.get("settings", {})
    hardware = cfg.get("hardware", {})
    custom_prompts = cfg.get("custom_prompts", {})

    # Settings: CLI args override config
    runs = args.runs or s.get("runs", 3)
    warmup = args.warmup or s.get("warmup", 1)
    max_tokens = args.max_tokens or s.get("max_tokens", 512)
    server_timeout = s.get("server_timeout", 600)
    cooldown = s.get("cooldown", 8)
    bench_port = s.get("bench_port", 8090)

    all_tests = build_tests(cfg, bench_port)

    if args.list:
        print_test_list(all_tests)
        return

    # Select tests
    if args.test:
        ids = {t.upper() for t in args.test}
        selected = [t for t in all_tests if t.id.upper() in ids]
        missing = ids - {t.id.upper() for t in selected}
        if missing:
            err(f"Unknown test IDs: {', '.join(sorted(missing))}")
            return
    elif args.group:
        selected = [t for t in all_tests if t.group == args.group]
    else:
        selected = all_tests

    if args.skip_unavailable:
        selected = [t for t in selected if not t.prereq]
    else:
        blocked = [t for t in selected if t.prereq]
        if blocked:
            warn(f"{len(blocked)} test(s) have unmet prerequisites:")
            for t in blocked:
                print(f"    {t.id}: {t.prereq}")
            print()
            selected = [t for t in selected if not t.prereq]
            if not selected:
                err("No tests to run.")
                return

    # Resolve prompts
    prompt_arg = args.prompt or cfg.get("prompts", ["short"])[0] if \
        len(cfg.get("prompts", [])) == 1 else None

    if args.prompt:
        prompt_items = resolve_prompts(args.prompt, custom_prompts)
    else:
        cfg_prompts = cfg.get("prompts", ["short"])
        prompt_items = []
        for p in cfg_prompts:
            prompt_items.extend(resolve_prompts(p, custom_prompts))

    # Banner
    hw_str = f"{hardware.get('name', 'Unknown')} {hardware.get('memory_gb', '')}GB".strip()
    print(f"\n{'═' * 80}")
    print(f"  {C_BOLD}llm-bench{C_RESET}")
    print(f"  {C_DIM}{hw_str}  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}{C_RESET}")
    flags = [f for f, v in [("no-think", args.no_think), ("tools", args.test_tools),
                             ("quality", args.test_quality)] if v]
    print(f"  Tests: {len(selected)}  ·  Prompts: {len(prompt_items)}  ·  "
          f"Runs: {runs}+{warmup}w  ·  Max tokens: {max_tokens}"
          + (f"  ·  {', '.join(flags)}" if flags else ""))
    print(f"{'═' * 80}\n")

    all_results = []
    tool_results = {}
    quality_results = {}

    for idx, test in enumerate(selected):
        print(f"\n{'━' * 80}")
        print(f"  {C_BOLD}[{idx + 1}/{len(selected)}] {test.id}: {test.name}{C_RESET}")
        print(f"  Backend: {test.backend}  ·  Fmt: {test.fmt}  ·  "
              f"Quant: {test.quant}  ·  KV: {test.kv_cache}")
        print(f"{'━' * 80}")

        t_load_start = time.perf_counter()
        proc = start_server(test, server_timeout)
        if proc is None:
            err("Failed to start server — skipping")
            continue

        if test.backend not in ("ollama", "lm-studio", "docker-model-runner"):
            info("Waiting for server...")
            if not wait_for_server(test.port, server_timeout):
                err(f"Server not ready after {server_timeout}s — skipping")
                stop_server(proc, test)
                continue

        t_load = time.perf_counter() - t_load_start
        ok(f"Ready  (load: {t_load:.1f}s)")

        monitor = None
        pid = _find_server_pid(test.backend, proc)
        if pid:
            monitor = ResourceMonitor(pid)
            monitor.start()

        # Effective no-think flag
        no_think = test.no_think_override if test.no_think_override is not None else args.no_think

        # Per-test prompts override the global list
        if test.prompts is not None:
            test_prompt_items = []
            for p in test.prompts:
                test_prompt_items.extend(resolve_prompts(p, custom_prompts))
        else:
            test_prompt_items = prompt_items

        for prompt_name, prompt_messages in test_prompt_items:
            est = sum(estimate_tokens(m.get("content", "")) for m in prompt_messages)
            print(f"\n  {C_CYAN}── Prompt: {prompt_name}  (~{est} tokens) ──{C_RESET}")

            if monitor:
                monitor.reset()

            cold_ttft = 0.0
            for w in range(warmup):
                info(f"Warmup {w + 1}/{warmup}...")
                wr = run_single_bench(test, prompt_messages, max_tokens, no_think)
                if wr and w == 0:
                    cold_ttft = wr.get("ttft_ms", 0.0)

            if monitor:
                monitor.reset()

            for run_num in range(1, runs + 1):
                info(f"Run {run_num}/{runs}...")
                r = run_single_bench(test, prompt_messages, max_tokens, no_think)
                if not r:
                    err(f"Run {run_num} failed")
                    continue

                br = BenchResult(
                    test_id=test.id, test_name=test.name,
                    backend=test.backend, fmt=test.fmt, quant=test.quant,
                    kv_cache=test.kv_cache, prompt_type=prompt_name, run_num=run_num,
                    ttft_ms=r["ttft_ms"], decode_tps=r["decode_tps"],
                    prefill_tps=r["prefill_tps"],
                    completion_tokens=r["completion_tokens"],
                    prompt_tokens=r["prompt_tokens"],
                    total_time_s=r["total_time_s"],
                    model_load_s=round(t_load, 1),
                    thinking_tokens=r.get("thinking_tokens", 0),
                    visible_tokens=r.get("visible_tokens", 0),
                    cold_ttft_ms=round(cold_ttft, 1),
                    peak_mem_mb=monitor.peak_mem_mb if monitor else 0,
                    peak_cpu_pct=monitor.peak_cpu_pct if monitor else 0,
                )
                all_results.append(br)

                think_s = (f"  think={br.thinking_tokens} vis={br.visible_tokens}"
                           if br.thinking_tokens > 0 else "")
                mem_s = f"  RSS={br.peak_mem_mb:.0f}MB" if br.peak_mem_mb > 0 else ""
                ok(f"TTFT={br.ttft_ms:.0f}ms  dec={br.decode_tps:.1f}t/s  "
                   f"pre={br.prefill_tps:.0f}t/s  tok={br.completion_tokens}"
                   f"{think_s}  total={br.total_time_s:.1f}s{mem_s}")

        if args.test_tools:
            info("Tool call test...")
            tr = run_tool_test(test, no_think)
            tool_results[test.id] = tr
            if tr.get("tool_call_valid"):
                ok(f"Tool: {tr.get('tool_name','?')} ({tr.get('tool_call_latency_ms',0):.0f}ms)")
            else:
                err(f"Tool FAIL: {tr.get('error','?')[:60]}")

        if args.test_quality:
            info("Quality test...")
            qr = run_quality_test(test, no_think)
            quality_results[test.id] = qr
            (ok if qr.get("quality_pass") else err)(
                "Quality: PASS" if qr.get("quality_pass")
                else f"Quality FAIL: {qr.get('error','?')[:60]}")

        if monitor:
            monitor.stop()
        info("Stopping server...")
        stop_server(proc, test)
        ok("Done")

        if idx < len(selected) - 1:
            info(f"Cooldown {cooldown}s...")
            time.sleep(cooldown)

    # Annotate results with tool/quality
    for r in all_results:
        tr = tool_results.get(r.test_id)
        if tr:
            r.tool_call_valid = tr.get("tool_call_valid")
        qr = quality_results.get(r.test_id)
        if qr:
            r.quality_pass = qr.get("quality_pass")

    print_results_table(all_results, hardware, tool_results, quality_results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"bench_{ts}.csv"
    save_results_csv(all_results, csv_path)

    if args.report_html:
        html_path = RESULTS_DIR / f"bench_{ts}.html"
        save_results_html(all_results, hardware, html_path, tool_results, quality_results)


if __name__ == "__main__":
    main()
