
'''
This is the code used to evaluate functional consistency.
'''

from __future__ import annotations

import argparse
import os
import re
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def gpt_rate_text_match_5d(
    client,
    target_text: str,
    generated_text: str,
    *,
    model: str = "openai/gpt-4.1",
    timeout_s: int = 60,
    max_retries: int = 3,
    temperature: float = 0.0,
):
    rubric = (
        "You are an expert evaluator. The Target Text is the ground truth. The Generated Text should be evaluated against it. Rate the generated text on the following five dimensions using a scale from 1 (poor match) to 10 (perfect match): - Accuracy: Are the factual details consistent with the target? - Completeness: Does it cover all key content from the target? - Tone: Is the style and formality consistent with the target? - Sentiment: Is the emotional attitude similar? - Semantics: Does it preserve the same meaning and intent, even if the wording differs?\n"
        'Return only the scores in this exact JSON format (no extra text):\n'
        '{ "Accuracy": X, "Completeness": X, "Tone": X, "Sentiment": X, "Semantics": X }\n\n'
        "[Target Text]\n"
        f"{target_text}\n"
        "[End Target Text]\n\n"
        "[Generated Text]\n"
        f"{generated_text}\n"
        "[End Generated Text]\n"
    )

    dims = ["Accuracy", "Completeness", "Tone", "Sentiment", "Semantics"]

    def _coerce_scores(obj):
        if not isinstance(obj, dict):
            return None
        out: Dict[str, int] = {}
        for k in dims:
            if k not in obj:
                return None
            try:
                v = int(obj[k])
            except Exception:
                return None
            if v < 1 or v > 10:
                return None
            out[k] = v
        return out

    def _extract_json(text):
        if not text:
            return None
        # direct parse
        try:
            obj = json.loads(text)
            coerced = _coerce_scores(obj)
            if coerced is not None:
                return coerced
        except Exception:
            pass

        # regex extract first {...}
        m = JSON_OBJ_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                return _coerce_scores(obj)
            except Exception:
                return None
        return None

    messages = [{"role": "user", "content": rubric}]
    last_text = ""

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout_s,
                temperature=temperature,
            )
            last_text = resp.choices[0].message.content or ""
            parsed = _extract_json(last_text)
            if parsed is not None:
                return parsed

            messages.append({"role": "assistant", "content": last_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        'Output ONLY valid JSON with integer values 1-10 in exactly this schema:\n'
                        '{ "Accuracy": X, "Completeness": X, "Tone": X, "Sentiment": X, "Semantics": X }\n'
                        "No other text."
                    ),
                }
            )

        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(1.2 * (attempt + 1) ** 2)

    raise RuntimeError(f"Unable to parse the rating JSON. Model output:\n{last_text}")


def is_empty(x):
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    if isinstance(x, str) and not x.strip():
        return True
    return False


def evaluate_folder_consistency_baseline_soft_mt(
    root_dir: str,
    csv_name: str,
    start_id: int = 1,
    end_id: int = 50,
    *,
    openrouter_api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "openai/gpt-4.1",
    overwrite: bool = True,
    output_suffix: str = ".5d",
    temperature: float = 0.0,
    sleep_per_call_s: float = 0.1,
    max_workers: int = 10,
):
   

    required_cols = ["user_prompt", "original_output", "baseline_output", "with_soft_output"]
    dims = ["Accuracy", "Completeness", "Tone", "Sentiment", "Semantics"]

    candidates = {
        "baseline": "baseline_output",
        "with_soft": "with_soft_output",
    }

    def avg5(scores: Dict[str, int]) -> float:
        return sum(scores[k] for k in dims) / 5.0


    cache: Dict[Tuple[str, str], Dict[str, int]] = {}
    cache_lock = threading.Lock()


    totals_lock = threading.Lock()
    total_sum_avg = {k: 0.0 for k in candidates.keys()}
    total_cnt = {k: 0 for k in candidates.keys()}

    tls = threading.local()

    def get_client():
        if not hasattr(tls, "client"):
            from openai import OpenAI

            tls.client = OpenAI(base_url=base_url, api_key=openrouter_api_key)
        return tls.client

    def cached_rate(target_text, generated_text):
        key = (target_text, generated_text)
        with cache_lock:
            hit = cache.get(key)
        if hit is not None:
            return hit

        scores = gpt_rate_text_match_5d(
            get_client(),
            target_text,
            generated_text,
            model=model,
            temperature=temperature,
        )

        with cache_lock:
            cache[key] = scores

        if sleep_per_call_s > 0:
            time.sleep(sleep_per_call_s)

        return scores

    def ensure_score_columns(df):
        for cand_key in candidates.keys():
            for dim in dims:
                col = f"{cand_key}_{dim}"
                if col not in df.columns:
                    df[col] = pd.NA
            avg_col = f"{cand_key}_avg"
            if avg_col not in df.columns:
                df[avg_col] = pd.NA

    def process_one_file(file_id):
        folder = Path(root_dir) / f"id_{file_id}"
        csv_path = folder / csv_name

        if not csv_path.exists():
            return {"file_id": file_id, "skipped": True, "reason": "csv_not_found", "saved_csv": None}

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return {"file_id": file_id, "skipped": True, "reason": f"read_failed: {e}", "saved_csv": None}

        df["user_prompt"] = df["user_prompt"].astype("string")

        df["user_prompt_id"] = df["user_prompt"].factorize(sort=False)[0]


        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return {"file_id": file_id, "skipped": True, "reason": f"missing_cols: {missing_cols}", "saved_csv": str(csv_path)}

        ensure_score_columns(df)

        file_sum_avg = {k: 0.0 for k in candidates.keys()}
        file_cnt = {k: 0 for k in candidates.keys()}

        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"id_{file_id} rows",
            position=file_id % max(1, min(max_workers, 20)),
            leave=False,
        ):

            target = row["original_output"]
            if is_empty(target):
                continue
            target = str(target)

            for cand_key, cand_col in candidates.items():
                gen = row[cand_col]
                if is_empty(gen):
                    continue
                gen = str(gen)

                scores = cached_rate(target, gen)

                for dim in dims:
                    df.at[idx, f"{cand_key}_{dim}"] = int(scores[dim])
                df.at[idx, f"{cand_key}_avg"] = avg5(scores)

                file_sum_avg[cand_key] += avg5(scores)
                file_cnt[cand_key] += 1

        out_path = csv_path if overwrite else csv_path.with_name(csv_path.stem + output_suffix + csv_path.suffix)
        try:
            df.to_csv(out_path, index=False)
        except Exception as e:
            return {"file_id": file_id, "skipped": True, "reason": f"write_failed: {e}", "saved_csv": str(out_path)}

        with totals_lock:
            for k in candidates.keys():
                total_sum_avg[k] += file_sum_avg[k]
                total_cnt[k] += file_cnt[k]

        per_file_avg = {k: (file_sum_avg[k] / file_cnt[k] if file_cnt[k] else 0.0) for k in candidates.keys()}

        return {
            "file_id": file_id,
            **{f"{k}_avg": v for k, v in per_file_avg.items()},
            **{f"n_{k}": file_cnt[k] for k in candidates.keys()},
            "saved_csv": str(out_path),
            "skipped": False,
            "reason": None,
        }


    ids = list(range(start_id, end_id + 1))
    per_file_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one_file, fid): fid for fid in ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="files"):
            per_file_results.append(fut.result())

    per_file_df = pd.DataFrame(per_file_results) if per_file_results else pd.DataFrame()

    overall = {}
    for k in candidates.keys():
        overall[f"{k}_avg_overall"] = (total_sum_avg[k] / total_cnt[k]) if total_cnt[k] else 0.0
        overall[f"{k}_n_overall"] = total_cnt[k]


    print("\n==== Overall functional consistency of all files (average score) ====")
    for k, v in overall.items():
        print(f"{k}: {v}")

    return per_file_df, overall

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate functional consistency (FC).")
    parser.add_argument("--root-dir", default="result", help="Directory containing id_<N> result folders.")
    parser.add_argument("--csv-name", default="benign_result.csv", help="CSV file name under each id_<N> folder.")
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=50)
    parser.add_argument(
        "--openrouter-api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input CSV files in place.")
    parser.add_argument("--output-suffix", default=".consistency")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-per-call-s", type=float, default=0.1)
    parser.add_argument("--max-workers", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.openrouter_api_key:
        raise ValueError("Please provide --openrouter-api-key or set OPENROUTER_API_KEY.")

    evaluate_folder_consistency_baseline_soft_mt(
        root_dir=args.root_dir,
        csv_name=args.csv_name,
        start_id=args.start_id,
        end_id=args.end_id,
        openrouter_api_key=args.openrouter_api_key,
        base_url=args.base_url,
        model=args.model,
        overwrite=args.overwrite,
        output_suffix=args.output_suffix,
        sleep_per_call_s=args.sleep_per_call_s,
        temperature=args.temperature,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()