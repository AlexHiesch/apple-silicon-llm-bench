#!/bin/bash
set -e

# Quality benchmark suite for ThinkingCap-Qwen3.6-27B-MLX-4bit
# Server must be started separately with --chat-template-args '{"enable_thinking":false}'
# for non-thinking evals (lm-eval / coding / context). Thinking-token evals enable thinking.

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV="${ROOT}/.venv"
VENV_EVAL="${ROOT}/.venv-eval"
BASE_URL="http://localhost:8090/v1"
CHAT_URL="${BASE_URL}/chat/completions"

# Single shared model for all harnesses
TC_MODEL="$(${VENV}/bin/python -c 'from harness_model import DEFAULT_MODEL; print(DEFAULT_MODEL)')"

start_server() {
    local model=$1
    local thinking=${2:-false}
    echo "=== Starting server with model: $model (thinking=$thinking) ==="
    pkill -f "mlx_lm.server" 2>/dev/null || true
    sleep 3
    if [ "$thinking" = "true" ]; then
        $VENV/bin/python -m mlx_lm.server --model "$model" --port 8090 &
    else
        $VENV/bin/python -m mlx_lm.server \
            --model "$model" \
            --port 8090 \
            --chat-template-args '{"enable_thinking":false}' &
    fi
    sleep 12
    curl -sf "$BASE_URL/models" > /dev/null || { echo "FAILED to start server"; exit 1; }
    echo "Server ready."
}

run_lm_eval() {
    local model=$1
    local label=$2
    echo ""
    echo "=== lm-eval: GSM8K (200 samples) — $label ==="
    $VENV/bin/lm_eval run \
        --model local-chat-completions \
        --model_args "model=$model,base_url=$CHAT_URL,tokenized_requests=False,max_gen_toks=4096" \
        --tasks gsm8k_cot_zeroshot \
        --limit 200 \
        --apply_chat_template \
        --output_path "results/quality/${label}_gsm8k"

    echo ""
    echo "=== lm-eval: MMLU-Pro Math (200 samples) — $label ==="
    $VENV/bin/lm_eval run \
        --model local-chat-completions \
        --model_args "model=$model,base_url=$CHAT_URL,tokenized_requests=False,max_gen_toks=4096" \
        --tasks mmlu_pro_math \
        --limit 200 \
        --apply_chat_template \
        --output_path "results/quality/${label}_mmlupro_math"
}

run_bigcodebench() {
    local model=$1
    local label=$2
    echo ""
    echo "=== BigCodeBench Hard (148 tasks) — $label ==="
    OPENAI_API_KEY=dummy $VENV_EVAL/bin/bigcodebench.generate \
        "$model" \
        instruct \
        hard \
        --backend openai \
        --base_url "$BASE_URL" \
        --greedy True \
        --n_samples 1 \
        --temperature 0 \
        --root "results/bigcodebench_${label}"
}

run_context_test() {
    local model=$1
    local label=$2
    echo ""
    echo "=== Context Degradation NIAH (2K-32K) — $label ==="
    $VENV/bin/python -c "
import eval_context_degradation as e
from pathlib import Path
e.MODEL = '$model'
e.main()
Path('results/context_degradation.json').rename('results/context_degradation_${label}.json')
"
}

run_coding_test() {
    local model=$1
    local label=$2
    echo ""
    echo "=== Applied Coding (15 tasks) — $label ==="
    $VENV/bin/python -c "
import eval_coding as e
from pathlib import Path
e.MODEL = '$model'
e.main()
Path('results/coding_eval.json').rename('results/coding_eval_${label}.json')
"
}

mkdir -p results/quality

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Quality Benchmark Suite: ThinkingCap-Qwen3.6-27B-MLX-4bit ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Model: $TC_MODEL"
echo ""

start_server "$TC_MODEL" false
run_bigcodebench "$TC_MODEL" "thinkingcap"
run_lm_eval "$TC_MODEL" "thinkingcap"
run_coding_test "$TC_MODEL" "thinkingcap"
run_context_test "$TC_MODEL" "thinkingcap"

pkill -f "mlx_lm.server" 2>/dev/null || true

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  ALL BENCHMARKS COMPLETE              ║"
echo "╚═══════════════════════════════════════╝"
