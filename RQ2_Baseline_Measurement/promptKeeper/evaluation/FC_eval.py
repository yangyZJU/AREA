'''
This is the code used to evaluate functional consistency.
'''

import re
import json
import time
import threading
from typing import Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def gpt_rate_text_match_5d(
    client: OpenAI,
    target_text: str,
    generated_text: str,
    *,
    model: str = "openai/gpt-4.1",
    timeout_s: int = 60,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> Dict[str, int]:
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


def evaluate_one_csv_consistency_5d_writeback(
    *,
    csv_path: str,
    openrouter_api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "openai/gpt-4.1",
    overwrite: bool = False,
    output_path: Optional[str] = None,
    output_suffix: str = ".consistency5d",
    temperature: float = 0.0,
    timeout_s: int = 60,
    max_retries: int = 3,
    sleep_per_call_s: float = 0.1,
    max_workers: int = 1,
    cache_enabled: bool = True,
    skip_scored_rows: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    target_col = "first_response"
    gen_col = "final_response"
    dims = ["Accuracy", "Completeness", "Tone", "Sentiment", "Semantics"]

    required_cols = [target_col, gen_col]
    df = pd.read_csv(csv_path)

    df["user_query_id"] = (
        df["user_query_name"]
        .astype(str)
        .str.extract(r"(\d+)$")[0]
        .astype("Int64") 
    )

    df["system_prompt_name"] = pd.to_numeric(df["system_prompt_name"], errors="coerce")
    df["user_query_id"] = pd.to_numeric(df["user_query_id"], errors="coerce")

    missing = [c for c in required_cols if c not in df.columns]
    


    out_cols = {dim: f"consistency_{dim}" for dim in dims}
    avg_col = "consistency_avg"
    for col in out_cols.values():
        if col not in df.columns:
            df[col] = pd.NA
    if avg_col not in df.columns:
        df[avg_col] = pd.NA

    def avg5(scores: Dict[str, int]) -> float:
        return sum(scores[d] for d in dims) / 5.0

    tls = threading.local()

    def get_client() -> OpenAI:
        if not hasattr(tls, "client"):
            tls.client = OpenAI(base_url=base_url, api_key=openrouter_api_key)
        return tls.client

    cache: Dict[Tuple[str, str], Dict[str, int]] = {}
    cache_lock = threading.Lock()

    def cached_rate(target_text, generated_text):
        if not cache_enabled:
            scores = gpt_rate_text_match_5d(
                get_client(),
                target_text,
                generated_text,
                model=model,
                timeout_s=timeout_s,
                max_retries=max_retries,
                temperature=temperature,
            )
            if sleep_per_call_s > 0:
                time.sleep(sleep_per_call_s)
            return scores

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
            timeout_s=timeout_s,
            max_retries=max_retries,
            temperature=temperature,
        )

        with cache_lock:
            cache[key] = scores

        if sleep_per_call_s > 0:
            time.sleep(sleep_per_call_s)

        return scores

    def row_already_scored(idx):
        if not skip_scored_rows:
            return False
        return not is_empty(df.at[idx, avg_col])

    def score_one_row(idx):
        '''
        if row_already_scored(idx):
            return idx, None
        '''
        target = df.at[idx, target_col]
        gen = df.at[idx, gen_col]

        if is_empty(target) or is_empty(gen):
            return idx, None

        scores = cached_rate(str(target), str(gen))
        return idx, scores

    n_scored = 0
    sum_avg = 0.0

    if max_workers <= 1:
        for idx in tqdm(df.index, total=len(df), desc="rows"):
            i, scores = score_one_row(idx)
            if scores is None:
                continue
            for dim in dims:
                df.at[i, out_cols[dim]] = int(scores[dim])
            a = avg5(scores)
            df.at[i, avg_col] = a
            n_scored += 1
            sum_avg += a
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(score_one_row, idx) for idx in df.index]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="rows"):
                i, scores = fut.result()
                if scores is None:
                    continue
                for dim in dims:
                    df.at[i, out_cols[dim]] = int(scores[dim])
                a = avg5(scores)
                df.at[i, avg_col] = a
                n_scored += 1
                sum_avg += a

    avg_overall = (sum_avg / n_scored) if n_scored > 0 else 0.0
    stats = {
        "avg_consistency_avg": avg_overall,
        "n_scored": n_scored,
        "n_total": len(df),
    }

    if overwrite:
        save_path = csv_path
    else:
        if output_path:
            save_path = output_path
        else:
            p = Path(csv_path)
            save_path = str(p.with_name(p.stem + output_suffix + p.suffix))

    df.to_csv(save_path, index=False)

    print("\n==== Summary ====")
    print(f"scored_rows: {n_scored}/{len(df)}")
    print(f"avg(consistency_avg) (1~10): {avg_overall:.4f}")
    print(f"saved_to: {save_path}")

    return df, stats


if __name__ == "__main__":
    df, stats = evaluate_one_csv_consistency_5d_writeback(
        csv_path="llama3_benign.csv",
        openrouter_api_key="sk-or-v1-xxx",
        overwrite=False,
        output_suffix=".consistency",
        model="openai/gpt-4.1",
        temperature=0.0,
        sleep_per_call_s=0.1,
        max_workers=20,
        cache_enabled=True,
        skip_scored_rows=True,
    )

