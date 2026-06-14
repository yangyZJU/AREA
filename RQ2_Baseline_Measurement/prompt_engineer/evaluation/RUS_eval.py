'''
This is the code for evaluating the Response Utility Score (RUS).

'''


import re
import time
import threading
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

RATING_RE = re.compile(r"\[\[(10|[1-9])\]\]")


def gpt41_rate_answer_quality(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    assistant_answer: str,
    *,
    model: str = "openai/gpt-4.1",
    timeout_s: int = 60,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> int:
    """
    Judge: given (system_prompt as predefined instructions, user_prompt, assistant_answer),
    output a rating 1-10 in format [[n]].
    """
    rubric = (
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.The AI assistant is required to follow certain predefined instructions or play a specific role in order to better answer questions. You need to evaluate whether this AI assistant has effectively adhered to these predefined tasks. Your evaluation should consider relevance of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format:\n"
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


def evaluate_one_csv_answer_quality_writeback(
    *,
    csv_path: str,
    openrouter_api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "openai/gpt-4.1",
    temperature: float = 0.0,
    timeout_s: int = 60,
    max_retries: int = 3,
    sleep_per_call_s: float = 0.0,
    overwrite: bool = False,
    output_path: Optional[str] = None,
    max_workers: int = 1,
    cache_enabled: bool = True,
    skip_scored_rows: bool = True,
):
    
    required_cols = ["raw_system_prompt", "user_prompt", "output_text"]
    score_col = "RUS"

    df = pd.read_csv(csv_path)



    missing = [c for c in required_cols if c not in df.columns]

    if score_col not in df.columns:
        df[score_col] = pd.NA


    tls = threading.local()

    def get_client():
        if not hasattr(tls, "client"):
            tls.client = OpenAI(base_url=base_url, api_key=openrouter_api_key)
        return tls.client


    cache: Dict[Tuple[str, str, str], int] = {}
    cache_lock = threading.Lock()

    def cached_rate(system_p, user_p, answer):
        if not cache_enabled:
            r = gpt41_rate_answer_quality(
                get_client(),
                system_p,
                user_p,
                answer,
                model=model,
                timeout_s=timeout_s,
                max_retries=max_retries,
                temperature=temperature,
            )
            if sleep_per_call_s > 0:
                time.sleep(sleep_per_call_s)
            return r

        key = (system_p, user_p, answer)
        with cache_lock:
            hit = cache.get(key)
        if hit is not None:
            return hit

        r = gpt41_rate_answer_quality(
            get_client(),
            system_p,
            user_p,
            answer,
            model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            temperature=temperature,
        )

        with cache_lock:
            cache[key] = r

        if sleep_per_call_s > 0:
            time.sleep(sleep_per_call_s)
        return r


    def score_one_row(idx):

        system_prompt = df.at[idx, "raw_system_prompt"]
        user_prompt = df.at[idx, "user_prompt"]
        assistant_answer = df.at[idx, "output_text"]

        if is_empty(system_prompt) or is_empty(user_prompt) or is_empty(assistant_answer):
            return idx, None

        r = cached_rate(str(system_prompt), str(user_prompt), str(assistant_answer))
        return idx, r

    n_scored = 0
    sum_scores = 0.0

    if max_workers <= 1:
        for idx in tqdm(df.index, total=len(df), desc="rows"):
            i, r = score_one_row(idx)
            if r is not None:
                df.at[i, score_col] = r
                n_scored += 1
                sum_scores += r
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(score_one_row, idx) for idx in df.index]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="rows"):
                i, r = fut.result()
                if r is not None:
                    df.at[i, score_col] = r
                    n_scored += 1
                    sum_scores += r

    avg_score = (sum_scores / n_scored) if n_scored > 0 else 0.0
    stats = {
        "avg_score_output_quality": avg_score,
        "n_scored": n_scored,
        "n_total": len(df),
    }

    if overwrite:
        save_path = csv_path
    else:
        save_path = output_path or (csv_path.replace(".csv", ".RUS.csv"))

    df.to_csv(save_path, index=False)

    print("\n==== Summary ====")
    print(f"scored_rows: {n_scored}/{len(df)}")
    print(f"avg_score_output_quality (1~10): {avg_score:.4f}")
    print(f"saved_to: {save_path}")

    return df, stats


if __name__ == "__main__":
    df, stats = evaluate_one_csv_answer_quality_writeback(
        csv_path="llama3_results/llama3_benign.csv",
        openrouter_api_key="sk-or-v1-xxx",
        overwrite=False,
        output_path=None,
        model="openai/gpt-4.1",
        temperature=0.0,
        sleep_per_call_s=0.1,
        max_workers=20,
        cache_enabled=True,
        skip_scored_rows=True,
    )
