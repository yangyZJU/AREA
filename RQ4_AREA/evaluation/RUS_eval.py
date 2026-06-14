'''
This is the code for evaluating the Response Utility Score (RUS).

'''

from __future__ import annotations

import argparse
import os
import re
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd

RATING_RE = re.compile(r"\[\[(10|[1-9])\]\]")


def gpt41_rate_answer_quality(
    client,
    system_prompt: str,
    user_prompt: str,
    assistant_answer: str,
    *,
    model: str = "openai/gpt-4.1",
    timeout_s: int = 60,
    max_retries: int = 3,
    temperature: float = 0.0,
):
    """
    Judge: given (system_prompt as predefined instructions, user_prompt, assistant_answer),
    output a rating 1-10 in format [[n]].
    """
    
    rubric = (
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. The AI assistant is required to follow certain predefined instructions or play a specific role in order to better answer questions. You need to evaluate whether this AI assistant has effectively adhered to these predefined tasks. Your evaluation should consider relevance of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format:\n"
        "\"[[rating]]\", for example: \"Rating: [[5]]\".\n"
        f"[The Start of Predefined Instructions]\n{system_prompt}\n[The End of Predefined Instructions]\n"
        f"[The Start of User’s Question]\n{user_prompt}\n[The End of User’s Question]\n"
        f"[The Start of Assistant’s Answer]\n{assistant_answer}\n[The End of Assistant’s Answer]\n"
    ) 
    

    def _extract(text):
        m = RATING_RE.search(text or "")
        return int(m.group(1)) if m else None

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
            rating = _extract(last_text)
            if rating is not None:
                return rating

            # force the model to output only the rating
            messages.append({"role": "assistant", "content": last_text})
            messages.append(
                {
                    "role": "user",
                    "content": "You must output ONLY the rating in the exact format [[n]] where n is 1-10. No other text.",
                }
            )
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(1.2 * (attempt + 1) ** 2)

    raise RuntimeError(f"Unable to parse [[rating]]. Model output:\n{last_text}")


def is_empty(x):
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    if isinstance(x, str) and not x.strip():
        return True
    return False


def evaluate_folder_with_gpt_and_writeback_mt(
    root_dir: str,
    csv_name: str,
    start_id: int = 1,
    end_id: int = 100,
    *,
    openrouter_api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "openai/gpt-4.1",
    overwrite: bool = True,
    output_suffix: str = ".judged",
    temperature: float = 0.0,
    sleep_per_call_s: float = 0.1,
    max_workers: int = 10,
):
   
    required_cols = [
        "system_prompt",
        "user_prompt",
        "original_output",
        "baseline_output",
        "with_soft_output",
    ]

    score_cols = [
        "score_original_quality",
        "score_baseline_quality",
        "score_with_soft_quality",
    ]

    def safe_avg(s: float, c: int):
        return s / c if c > 0 else 0.0


    cache: Dict[Tuple[str, str, str], int] = {}
    cache_lock = threading.Lock()


    total_sums = {"original": 0.0, "baseline": 0.0, "with_soft": 0.0}
    total_counts = {"original": 0, "baseline": 0, "with_soft": 0}
    totals_lock = threading.Lock()


    tls = threading.local()

    def get_client():
        if not hasattr(tls, "client"):
            from openai import OpenAI

            tls.client = OpenAI(base_url=base_url, api_key=openrouter_api_key)
        return tls.client

    def cached_rate(system_p, user_p, answer):
        key = (system_p, user_p, answer)
        with cache_lock:
            hit = cache.get(key)
        if hit is not None:
            return hit

        rating = gpt41_rate_answer_quality(
            get_client(),
            system_p,
            user_p,
            answer,
            model=model,
            temperature=temperature,
        )

        with cache_lock:
            cache[key] = rating

        if sleep_per_call_s > 0:
            time.sleep(sleep_per_call_s)

        return rating

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

        for c in score_cols:
            if c not in df.columns:
                df[c] = pd.NA

        file_sums = {"original": 0.0, "baseline": 0.0, "with_soft": 0.0}
        file_counts = {"original": 0, "baseline": 0, "with_soft": 0}

        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"id_{file_id} rows",
            position=file_id % max(1, min(max_workers, 20)),
            leave=False,
        ):

            system_prompt = row["system_prompt"]
            user_prompt = row["user_prompt"]

            if is_empty(system_prompt) or is_empty(user_prompt):
                continue

            system_prompt = str(system_prompt)
            user_prompt = str(user_prompt)

            original_output = None if is_empty(row["original_output"]) else str(row["original_output"])
            baseline_output = None if is_empty(row["baseline_output"]) else str(row["baseline_output"])
            with_soft_output = None if is_empty(row["with_soft_output"]) else str(row["with_soft_output"])

            if original_output is not None:
                s1 = cached_rate(system_prompt, user_prompt, original_output)
                df.at[idx, "score_original_quality"] = s1
                file_sums["original"] += s1
                file_counts["original"] += 1
            else:
                df.at[idx, "score_original_quality"] = pd.NA

            if baseline_output is not None:
                s2 = cached_rate(system_prompt, user_prompt, baseline_output)
                df.at[idx, "score_baseline_quality"] = s2
                file_sums["baseline"] += s2
                file_counts["baseline"] += 1
            else:
                df.at[idx, "score_baseline_quality"] = pd.NA

            if with_soft_output is not None:
                s3 = cached_rate(system_prompt, user_prompt, with_soft_output)
                df.at[idx, "score_with_soft_quality"] = s3
                file_sums["with_soft"] += s3
                file_counts["with_soft"] += 1
            else:
                df.at[idx, "score_with_soft_quality"] = pd.NA


        out_path = csv_path if overwrite else csv_path.with_name(csv_path.stem + output_suffix + csv_path.suffix)
        try:
            df.to_csv(out_path, index=False)
        except Exception as e:
            return {"file_id": file_id, "skipped": True, "reason": f"write_failed: {e}", "saved_csv": str(out_path)}

        file_avg_original = safe_avg(file_sums["original"], file_counts["original"])
        file_avg_baseline = safe_avg(file_sums["baseline"], file_counts["baseline"])
        file_avg_with_soft = safe_avg(file_sums["with_soft"], file_counts["with_soft"])

        with totals_lock:
            for key in ["original", "baseline", "with_soft"]:
                total_sums[key] += file_sums[key]
                total_counts[key] += file_counts[key]

        return {
            "file_id": file_id,
            "original_quality": file_avg_original,
            "baseline_quality": file_avg_baseline,
            "with_soft_quality": file_avg_with_soft,
            "n_original": file_counts["original"],
            "n_baseline": file_counts["baseline"],
            "n_with_soft": file_counts["with_soft"],
            "saved_csv": str(out_path),
            "skipped": False,
            "reason": None,
        }


    per_file_results = []
    ids = list(range(start_id, end_id + 1)) 
    

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one_file, fid): fid for fid in ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="files"):
            res = fut.result()
            if res is not None:
                per_file_results.append(res)

    per_file_df = pd.DataFrame(per_file_results) if per_file_results else pd.DataFrame()

    overall = {
        "original_output quality": safe_avg(total_sums["original"], total_counts["original"]),
        "baseline_output quality": safe_avg(total_sums["baseline"], total_counts["baseline"]),
        "with_soft_output quality": safe_avg(total_sums["with_soft"], total_counts["with_soft"]),
    }



    print("\n==== Overall average score for all files ====")
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")

    return per_file_df, overall

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Response Utility Score (RUS).")
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
    parser.add_argument("--output-suffix", default=".RUS")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-per-call-s", type=float, default=0.1)
    parser.add_argument("--max-workers", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.openrouter_api_key:
        raise ValueError("Please provide --openrouter-api-key or set OPENROUTER_API_KEY.")

    evaluate_folder_with_gpt_and_writeback_mt(
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
