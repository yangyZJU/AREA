import re
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd
from openai import OpenAI

RATING_RE = re.compile(r"\[\[(10|[1-9])\]\]")


def gpt41_rate_answer_quality(
    client: OpenAI,
    original_system_prompt: str,
    test_user_prompt: str,
    assistant_answer: str,
    *,
    model: str = "openai/gpt-4.1",
    timeout_s: int = 60,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> int:
    """
    Judge: given (original_system_prompt as predefined instructions, test_user_prompt, assistant_answer),
    output a rating 1-10 in format [[n]].
    """
    rubric = (
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.The AI assistant is required to follow certain predefined instructions or play a specific role in order to better answer questions. You need to evaluate whether this AI assistant has effectively adhered to these predefined tasks. Your evaluation should consider relevance of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format:\n"
        "\"[[rating]]\", for example: \"Rating: [[5]]\".\n"
        f"[The Start of Predefined Instructions]\n{original_system_prompt}\n[The End of Predefined Instructions]\n"
        f"[The Start of User’s Question]\n{test_user_prompt}\n[The End of User’s Question]\n"
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
        "original_system_prompt",
        "test_user_prompt",
        "origin_output",
        "output_wo_sys",
        "obf_sys_output",
    ]

    score_cols = [
        "score_original_quality",
        "score_wo_sys_quality",
        "score_with_obf_quality",
    ]

    def safe_avg(s: float, c: int) -> float:
        return s / c if c > 0 else 0.0

    cache: Dict[Tuple[str, str, str], int] = {}
    cache_lock = threading.Lock()

    total_sums = {"original": 0.0, "wo_sys": 0.0, "with_obf": 0.0}
    total_counts = {"original": 0, "wo_sys": 0, "with_obf": 0}
    totals_lock = threading.Lock()

    tls = threading.local()

    def get_client():
        if not hasattr(tls, "client"):
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
        folder = Path(root_dir) / f"obfuscate_truthQA_{file_id}"
        csv_path = folder / csv_name

        if not csv_path.exists():
            return {"file_id": file_id, "skipped": True, "reason": "csv_not_found", "saved_csv": None}

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return {"file_id": file_id, "skipped": True, "reason": f"read_failed: {e}", "saved_csv": None}


        df["original_system_prompt"] = df["original_system_prompt"].astype("string")
        df["test_user_prompt"] = df["test_user_prompt"].astype("string")

        df["system_id"] = df["original_system_prompt"].factorize(sort=False)[0]
        df["user_prompt_id"] = df["test_user_prompt"].factorize(sort=False)[0]

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return {"file_id": file_id, "skipped": True, "reason": f"missing_cols: {missing_cols}", "saved_csv": str(csv_path)}

        for c in score_cols:
            if c not in df.columns:
                df[c] = pd.NA

        file_sums = {"original": 0.0, "wo_sys": 0.0, "with_obf": 0.0}
        file_counts = {"original": 0, "wo_sys": 0, "with_obf": 0}

        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"id_{file_id} rows",
            position=file_id % max(1, min(max_workers, 20)),
            leave=False,
        ):
            original_system_prompt = row["original_system_prompt"]
            test_user_prompt = row["test_user_prompt"]

            if is_empty(original_system_prompt) or is_empty(test_user_prompt):
                continue

            original_system_prompt = str(original_system_prompt)
            test_user_prompt = str(test_user_prompt)

            origin_output = None if is_empty(row["origin_output"]) else str(row["origin_output"])
            output_wo_sys = None if is_empty(row["output_wo_sys"]) else str(row["output_wo_sys"])
            obf_sys_output = None if is_empty(row["obf_sys_output"]) else str(row["obf_sys_output"])

            if origin_output is not None:
                s1 = cached_rate(original_system_prompt, test_user_prompt, origin_output)
                df.at[idx, "score_original_quality"] = s1
                file_sums["original"] += s1
                file_counts["original"] += 1
            else:
                df.at[idx, "score_original_quality"] = pd.NA

            if output_wo_sys is not None:
                s2 = cached_rate(original_system_prompt, test_user_prompt, output_wo_sys)
                df.at[idx, "score_wo_sys_quality"] = s2
                file_sums["wo_sys"] += s2
                file_counts["wo_sys"] += 1
            else:
                df.at[idx, "score_wo_sys_quality"] = pd.NA

            if obf_sys_output is not None:
                s3 = cached_rate(original_system_prompt, test_user_prompt, obf_sys_output)
                df.at[idx, "score_with_obf_quality"] = s3
                file_sums["with_obf"] += s3
                file_counts["with_obf"] += 1
            else:
                df.at[idx, "score_with_obf_quality"] = pd.NA

        out_path = csv_path if overwrite else csv_path.with_name(csv_path.stem + output_suffix + csv_path.suffix)
        try:
            df.to_csv(out_path, index=False)
        except Exception as e:
            return {"file_id": file_id, "skipped": True, "reason": f"write_failed: {e}", "saved_csv": str(out_path)}

        file_avg_original = safe_avg(file_sums["original"], file_counts["original"])
        file_avg_wo_sys = safe_avg(file_sums["wo_sys"], file_counts["wo_sys"])
        file_avg_with_soft = safe_avg(file_sums["with_obf"], file_counts["with_obf"])

        with totals_lock:
            for key in ["original", "wo_sys", "with_obf"]:
                total_sums[key] += file_sums[key]
                total_counts[key] += file_counts[key]

        return {
            "file_id": file_id,
            "original_quality": file_avg_original,
            "wo_sys_quality": file_avg_wo_sys,
            "with_obf_quality": file_avg_with_soft,
            "n_original": file_counts["original"],
            "n_wo_sys": file_counts["wo_sys"],
            "n_with_obf": file_counts["with_obf"],
            "saved_csv": str(out_path),
            "skipped": False,
            "reason": None,
        }


    per_file_results: List[Dict] = []
    ids = list(range(start_id, end_id + 1))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one_file, fid): fid for fid in ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="files"):
            res = fut.result()
            if res is not None:
                per_file_results.append(res)

    per_file_df = pd.DataFrame(per_file_results) if per_file_results else pd.DataFrame()

    overall = {
        "origin_output quality": safe_avg(total_sums["original"], total_counts["original"]),
        "output_wo_sys quality": safe_avg(total_sums["wo_sys"], total_counts["wo_sys"]),
        "obf_sys_output quality": safe_avg(total_sums["with_obf"], total_counts["with_obf"]),
    }


    print("\n==== Overall average score (weighted, 1-10) for all files ====")
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")

    return per_file_df, overall



per_file_df, overall = evaluate_folder_with_gpt_and_writeback_mt(
    root_dir="llama3_benign_obfuscation",
    csv_name="obf_sys_output.csv",
    start_id=1,
    end_id=50,
    openrouter_api_key="sk-or-v1-xxx",
    overwrite=False,
    output_suffix=".RUS",
    sleep_per_call_s=0.1,
    temperature=0.0,
    max_workers=20,
    )

