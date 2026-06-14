# Targeted Adaptive Queries

This directory contains the targeted adaptive query evaluation described in the paper. The goal is to stress different components of AREA using four hand-designed attack families generated with an auxiliary LLM.

## Attack Types

The query files are stored under `data/adaptive_attack/`. Each file contains 30 targeted adaptive queries.

- `Semantic_Collision.json`: queries semantically close to the defensive instruction, designed to compete with defensive tokens for attention.
- `Long_Prefix.json`: queries with a long benign prefix before the leakage request, testing whether attention re-anchoring remains effective under distracting context.
- `Encoded_Leakage.json`: queries asking the model to reveal the system prompt in encoded or transformed forms such as Base64, hexadecimal, or Morse code.
- `Refusal_Evasion.json`: queries that explicitly instruct the model not to start with the defensive trigger and then request the system prompt.

## Required Inputs

The evaluator expects trained AREA checkpoints under:

```text
llama3_checkpoint/id_<system_id>/
```

Each checkpoint directory should contain:

```text
soft_prompt.pt
optimized_soft_prompt.json
```

The metadata JSON should include `system_prompt` and `defensive_instruction` (or the compatible field `defense_prompt`).

## Run Targeted Attacks

By default, `run_adaptive_attack_eval.py` evaluates IDs `31` to `40` on all four attack types and distributes attack types across the GPUs listed by `--gpus`.

```bash
python -u run_adaptive_attack_eval.py \
  --gpus 0,1 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

Useful options:

```bash
# Include the No Defense baseline output.
python -u run_adaptive_attack_eval.py \
  --gpus 0,1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --compare-baseline

# Reduce memory use by skipping the original no-defense output.
python -u run_adaptive_attack_eval.py \
  --gpus 0,1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --no-compare-original
```

Main outputs are written under `result/`:

- `result/<Attack_Type>.csv`: aggregated results for each attack type.
- `result/raw/<Attack_Type>/id_<system_id>.csv`: raw per-system outputs.
- `result/raw/<Attack_Type>/id_<system_id>.json`: raw per-system JSON outputs.
- `result/logs/<Attack_Type>/id_<system_id>.log`: per-run logs.

## Leakage Evaluation

We report SS and PLS as leakage metrics.

For `Encoded_Leakage`, model outputs may be encoded or reformatted, so decode them before SS/PLS evaluation:

```bash
python -u attack_eval/llm_decode_eval.py \
  --openrouter-api-key "$OPENROUTER_API_KEY"
```

This writes:

```text
result/Encoded_Leakage.llm_decoded.decoded.csv
```

Then run SS:

```bash
python -u attack_eval/SS_eval.py --input-csv result/Encoded_Leakage.llm_decoded.decoded.csv
python -u attack_eval/SS_eval.py --input-csv result/Long_Prefix.csv
python -u attack_eval/SS_eval.py --input-csv result/Refusal_Evasion.csv
python -u attack_eval/SS_eval.py --input-csv result/Semantic_Collision.csv
```

Run PLS:

```bash
python -u attack_eval/PLS_eval.py --input-csv result/Encoded_Leakage.llm_decoded.decoded.csv --openrouter-api-key "$OPENROUTER_API_KEY"
python -u attack_eval/PLS_eval.py --input-csv result/Long_Prefix.csv --openrouter-api-key "$OPENROUTER_API_KEY"
python -u attack_eval/PLS_eval.py --input-csv result/Refusal_Evasion.csv --openrouter-api-key "$OPENROUTER_API_KEY"
python -u attack_eval/PLS_eval.py --input-csv result/Semantic_Collision.csv --openrouter-api-key "$OPENROUTER_API_KEY"
```

Evaluation outputs are saved under `attack_eval/` as `*.SS.csv`, `*.SS.summary.txt`, `*.PLS.csv`, and `*.PLS.summary.txt`.

