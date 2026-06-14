#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
RUN_ROOT="${RUN_ROOT:-llama3_kl_1_atk_0.5_aligned}"
OUT_DIR="${OUT_DIR:-adaptive_attack_results/surrogate_ss_llama3}"
GPU="${GPU:-0}"

# Default experiment split: first 5 system prompts as surrogates, next 5 as targets.
SURROGATE_SYS_IDS="${SURROGATE_SYS_IDS:-1,2,3,4,5}"
TARGET_SYS_IDS="${TARGET_SYS_IDS:-}"

EXTRA_ARGS=()
if [[ -z "${ATTACKER_MODEL:-}" ]]; then
  echo "[run_adaptive_surrogate_ss] ATTACKER_MODEL not set; using --fallback-mutator" >&2
  EXTRA_ARGS+=(--fallback-mutator)
fi
if [[ -z "${PLS_MODEL:-}" && -z "${OPENAI_API_KEY:-}" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "[run_adaptive_surrogate_ss] PLS judge not configured; using --fallback-pls-ss-proxy" >&2
  EXTRA_ARGS+=(--fallback-pls-ss-proxy)
fi

CMD=(python -u adaptive_attack_surrogate_ss.py
  --model "$MODEL_PATH"
  --surrogate-run-root "$RUN_ROOT"
  --target-run-root "$RUN_ROOT"
  --surrogate-sys-ids "$SURROGATE_SYS_IDS"
  --out-dir "$OUT_DIR"
  --iterations 100
  --checkpoint-iters "10,20,30,40,50,60,70,80,90,100"
  --candidates-per-iter 5
  --elite-size 5
  --transfer-top-k 1
  --inference-alpha 1
  --disable-thinking
)

if [[ -n "$TARGET_SYS_IDS" ]]; then
  CMD+=(--target-sys-ids "$TARGET_SYS_IDS")
fi
CMD+=("${EXTRA_ARGS[@]}")
CMD+=("$@")

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
