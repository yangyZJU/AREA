#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
RUN_ROOT="${RUN_ROOT:-llama3_kl_1_atk_0.5_aligned}"
OUT_DIR="${OUT_DIR:-adaptive_attack_results/response_judge_llama3}"
GPU="${GPU:-1}"

TARGET_SYS_IDS="${TARGET_SYS_IDS:-6,7,8,9,10}"

EXTRA_ARGS=()
if [[ -z "${ATTACKER_MODEL:-}" ]]; then
  echo "[run_adaptive_response_judge] ATTACKER_MODEL not set; using --fallback-mutator" >&2
  EXTRA_ARGS+=(--fallback-mutator)
fi
if [[ -z "${JUDGE_MODEL:-}" && -z "${ATTACKER_MODEL:-}" ]]; then
  echo "[run_adaptive_response_judge] JUDGE_MODEL not set; using --fallback-judge" >&2
  EXTRA_ARGS+=(--fallback-judge)
fi

CUDA_VISIBLE_DEVICES="$GPU" python -u adaptive_attack_response_judge.py   --model "$MODEL_PATH"   --target-run-root "$RUN_ROOT"   --target-sys-ids "$TARGET_SYS_IDS"   --out-dir "$OUT_DIR"   --iterations 100   --checkpoint-iters "10,20,30,40,50,60,70,80,90,100"   --candidates-per-iter 5   --elite-size 5   --inference-alpha 1   --disable-thinking   "${EXTRA_ARGS[@]}"   "$@"
