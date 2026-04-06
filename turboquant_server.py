#!/usr/bin/env python3
"""mlx-lm server with TurboQuant KV cache (drop-in patch for mlx-lm.server).

Run as:
  python3 turboquant_server.py --model <hf-repo-id> --port 8090 --tq-bits 4

TurboQuant (Google, ICLR 2026) replaces mlx-lm's KVCache with a PolarQuant-based
quantized cache. Repo: https://github.com/sharpner/turboquant-mlx
Install: run_turboquant_omlx.py handles this automatically.
"""
import sys
import argparse
from pathlib import Path

# Ensure the repo dir is importable even without a proper pip install
_repo_dir = Path("/tmp/turboquant-mlx")
if _repo_dir.exists() and str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

# Strip TurboQuant-specific args before passing to mlx-lm
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--tq-bits", type=float, default=4.0,
                    help="TurboQuant KV quantization bits (2, 3, 3.5, or 4)")
tq_args, remaining = parser.parse_known_args()
_bits = tq_args.tq_bits

# --- Patch 1: SDPA dispatch (routes TurboQuant cache types to fast attention) ---
_sdpa_patched = False
try:
    import turboquant.patch as tq_patch
    tq_patch.apply()
    _sdpa_patched = True
    print(f"[TurboQuant] SDPA patch applied", flush=True)
except ImportError as e:
    print(f"[TurboQuant] WARNING: import failed ({e}) — TurboQuant not available", flush=True)
except Exception as e:
    print(f"[TurboQuant] WARNING: SDPA patch failed ({e})", flush=True)

# --- Patch 2: Cache factory (replaces KVCache() with TurboQuantKVCacheV2) ---
# mlx-lm server calls make_prompt_cache(model) to create per-layer caches.
# We replace it to return TurboQuant caches. head_dim is read from model.layers[0].
if _sdpa_patched:
    try:
        from turboquant.cache_v2 import TurboQuantKVCacheV2
        import mlx_lm.models.cache as _cache_module

        _orig_make_prompt_cache = _cache_module.make_prompt_cache

        def _tq_make_prompt_cache(model, max_kv_size=None):
            # Try common attention attribute names across architectures
            try:
                layer = model.layers[0]
                attn = (getattr(layer, "self_attn", None)
                        or getattr(layer, "attn", None)
                        or getattr(layer, "attention", None)
                        or layer)
                head_dim = attn.head_dim
            except AttributeError:
                print("[TurboQuant] WARNING: can't determine head_dim — falling back to standard cache",
                      flush=True)
                return _orig_make_prompt_cache(model, max_kv_size)

            bits = int(_bits) if _bits == int(_bits) else _bits  # keep float for 3.5
            n_layers = len(model.layers)
            print(f"[TurboQuant] Creating {n_layers} × TurboQuantKVCacheV2 "
                  f"(head_dim={head_dim}, bits={bits})", flush=True)
            return [
                TurboQuantKVCacheV2(head_dim=head_dim, bits=bits, seed=42 + i)
                for i in range(n_layers)
            ]

        _cache_module.make_prompt_cache = _tq_make_prompt_cache
        print(f"[TurboQuant] make_prompt_cache patched for {_bits}-bit KV compression",
              flush=True)
    except Exception as e:
        print(f"[TurboQuant] WARNING: cache factory patch failed ({e}) — "
              "SDPA patch active but standard KV cache in use", flush=True)

# Run mlx-lm server with remaining args
sys.argv = [sys.argv[0]] + remaining
from mlx_lm.server import main
main()
