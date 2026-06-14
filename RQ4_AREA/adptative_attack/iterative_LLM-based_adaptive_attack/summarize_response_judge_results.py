#!/usr/bin/env python3
'''
python summarize_response_judge_results.py \
  --result-dir adaptive_attack_results/response_judge_llama3_kl_0.9 \
  --ids 6,7,8,9,10

'''
import argparse
import csv
import json
import os
import re

SCORE_TIE_EPS = 1e-9


def parse_ids(raw):
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return ids


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value):
    try:
        return int(float(value))
    except Exception:
        return 0


def checkpoint_round_from_name(name):
    match = re.match(r"checkpoint_round_(\d+)\.json$", name)
    if not match:
        return None
    return int(match.group(1))


def collect_checkpoint_rows(result_dir, ids):
    rows = []
    missing = []

    for sys_id in ids:
        id_dir = os.path.join(result_dir, f"id_{sys_id}")
        if not os.path.isdir(id_dir):
            missing.append(f"id_{sys_id}: missing dir")
            continue

        checkpoint_files = []
        for name in os.listdir(id_dir):
            checkpoint_round = checkpoint_round_from_name(name)
            if checkpoint_round is not None:
                checkpoint_files.append((checkpoint_round, os.path.join(id_dir, name)))
        checkpoint_files.sort()

        if not checkpoint_files:
            missing.append(f"id_{sys_id}: no checkpoint_round_*.json")
            continue

        for checkpoint_round, path in checkpoint_files:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            best_rows = payload.get("best", [])
            for rank, row in enumerate(best_rows, 1):
                flat = {
                    "sys_id": payload.get("sys_id", sys_id),
                    "checkpoint_round": payload.get("checkpoint_round", checkpoint_round),
                    "rank": rank,
                }
                flat.update(row)
                rows.append(flat)

    rows.sort(
        key=lambda row: (
            safe_int(row.get("sys_id")),
            safe_int(row.get("checkpoint_round")),
            safe_int(row.get("rank")),
        )
    )
    return rows, missing


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = []
    preferred = [
        "sys_id",
        "checkpoint_round",
        "rank",
        "search_step",
        "eval_count",
        "candidate",
        "response",
        "score",
        "judge_reason",
        "judge_feedback",
        "run_dir",
        "offline_ss_to_system_prompt",
        "offline_pls_to_system_prompt",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_round_summary(rows):
    grouped = {}
    for row in rows:
        if safe_int(row.get("rank")) != 1:
            continue
        checkpoint_round = safe_int(row.get("checkpoint_round"))
        grouped.setdefault(checkpoint_round, []).append(row)

    summary = []
    for checkpoint_round in sorted(grouped):
        bucket = grouped[checkpoint_round]
        judge_vals = [safe_float(row.get("score")) for row in bucket]
        ss_vals = [safe_float(row.get("offline_ss_to_system_prompt")) for row in bucket]
        pls_vals = [safe_float(row.get("offline_pls_to_system_prompt")) for row in bucket]
        summary.append(
            {
                "checkpoint_round": checkpoint_round,
                "n_ids": len(bucket),
                "avg_best_judge": sum(judge_vals) / len(judge_vals) if judge_vals else 0.0,
                "avg_offline_ss": sum(ss_vals) / len(ss_vals) if ss_vals else 0.0,
                "avg_offline_pls": sum(pls_vals) / len(pls_vals) if pls_vals else 0.0,
                "max_best_judge": max(judge_vals) if judge_vals else 0.0,
                "max_offline_ss": max(ss_vals) if ss_vals else 0.0,
                "max_offline_pls": max(pls_vals) if pls_vals else 0.0,
            }
        )
    return summary


def build_id_summary(rows):
    grouped = {}
    for row in rows:
        if safe_int(row.get("rank")) != 1:
            continue
        sys_id = safe_int(row.get("sys_id"))
        grouped.setdefault(sys_id, []).append(row)

    summary = []
    for sys_id in sorted(grouped):
        bucket = grouped[sys_id]
        judge_vals = [safe_float(row.get("score")) for row in bucket]
        ss_vals = [safe_float(row.get("offline_ss_to_system_prompt")) for row in bucket]
        pls_vals = [safe_float(row.get("offline_pls_to_system_prompt")) for row in bucket]
        summary.append(
            {
                "sys_id": sys_id,
                "n_checkpoints": len(bucket),
                "avg_best_judge": sum(judge_vals) / len(judge_vals) if judge_vals else 0.0,
                "avg_offline_ss": sum(ss_vals) / len(ss_vals) if ss_vals else 0.0,
                "avg_offline_pls": sum(pls_vals) / len(pls_vals) if pls_vals else 0.0,
                "max_best_judge": max(judge_vals) if judge_vals else 0.0,
                "max_offline_ss": max(ss_vals) if ss_vals else 0.0,
                "max_offline_pls": max(pls_vals) if pls_vals else 0.0,
            }
        )
    return summary


def get_selection_value(row, best_by):
    if best_by == "ss":
        return safe_float(row.get("offline_ss_to_system_prompt"))
    if best_by == "oracle":
        offline_ss = safe_float(row.get("offline_ss_to_system_prompt"))
        offline_pls = safe_float(row.get("offline_pls_to_system_prompt"))
        return 0.5 * offline_ss + 0.5 * (offline_pls / 10.0)
    return safe_float(row.get("score"))


def build_best_prompt_rows(rows, include_tied_best=False, best_by="judge"):
    grouped = {}
    for row in rows:
        key = (safe_int(row.get("sys_id")), safe_int(row.get("checkpoint_round")))
        grouped.setdefault(key, []).append(row)

    best_rows = []
    for bucket in grouped.values():
        max_score = max(get_selection_value(row, best_by) for row in bucket)
        tied_rows = [
            row
            for row in bucket
            if abs(get_selection_value(row, best_by) - max_score) <= SCORE_TIE_EPS
        ]
        tied_rows.sort(
            key=lambda row: (
                safe_int(row.get("search_step")),
                safe_int(row.get("eval_count")),
            ),
            reverse=True,
        )
        selected_rows = tied_rows if include_tied_best else tied_rows[:1]
        tie_count = len(tied_rows)

        for tie_rank, row in enumerate(selected_rows, 1):
            best_rows.append(
                {
                    "sys_id": safe_int(row.get("sys_id")),
                    "checkpoint_round": safe_int(row.get("checkpoint_round")),
                    "search_step": safe_int(row.get("search_step")),
                    "eval_count": safe_int(row.get("eval_count")),
                    "tie_rank": tie_rank,
                    "tie_count": tie_count,
                    "best_attack_prompt": row.get("candidate", ""),
                    "best_response": row.get("response", ""),
                    "judge_score": safe_float(row.get("score")),
                    "offline_ss_to_system_prompt": safe_float(row.get("offline_ss_to_system_prompt")),
                    "offline_pls_to_system_prompt": safe_float(row.get("offline_pls_to_system_prompt")),
                    "best_by": best_by,
                    "best_by_value": get_selection_value(row, best_by),
                    "judge_reason": row.get("judge_reason", ""),
                    "judge_feedback": row.get("judge_feedback", ""),
                    "run_dir": row.get("run_dir", ""),
                }
            )
    best_rows.sort(
        key=lambda row: (
            row["sys_id"],
            row["checkpoint_round"],
            row.get("tie_rank", 0),
        )
    )
    return best_rows


def build_round_avg_from_best_rows(best_rows):
    grouped = {}
    for row in best_rows:
        checkpoint_round = safe_int(row.get("checkpoint_round"))
        grouped.setdefault(checkpoint_round, []).append(row)

    summary = []
    for checkpoint_round in sorted(grouped):
        bucket = grouped[checkpoint_round]
        judge_vals = [safe_float(row.get("judge_score")) for row in bucket]
        ss_vals = [safe_float(row.get("offline_ss_to_system_prompt")) for row in bucket]
        pls_vals = [safe_float(row.get("offline_pls_to_system_prompt")) for row in bucket]
        unique_ids = {safe_int(row.get("sys_id")) for row in bucket}
        summary.append(
            {
                "checkpoint_round": checkpoint_round,
                "n_ids": len(unique_ids),
                "n_best_rows": len(bucket),
                "avg_judge_score": sum(judge_vals) / len(judge_vals) if judge_vals else 0.0,
                "avg_offline_ss": sum(ss_vals) / len(ss_vals) if ss_vals else 0.0,
                "avg_offline_pls": sum(pls_vals) / len(pls_vals) if pls_vals else 0.0,
            }
        )
    return summary


def build_round_avg_default_path(result_dir, best_by, include_tied_best):
    suffix = f"{best_by}"
    if include_tied_best:
        suffix += "_tied"
    filename = f"best_prompt_round_avg_rebuilt_{suffix}.csv"
    return os.path.join(result_dir, filename)


def build_best_prompt_default_path(result_dir, best_by, include_tied_best):
    suffix = f"{best_by}"
    if include_tied_best:
        suffix += "_tied"
    filename = f"best_prompt_by_id_round_rebuilt_{suffix}.csv"
    return os.path.join(result_dir, filename)


def add_mode_suffix(path, best_by, include_tied_best):
    suffix = f"_{best_by}"
    if include_tied_best:
        suffix += "_tied"
    base, ext = os.path.splitext(path)
    ext = ext or ".csv"
    return f"{base}{suffix}{ext}"


def parse_args():
    ap = argparse.ArgumentParser(description="Rebuild response-judge summaries from id_x checkpoint JSON files.")
    ap.add_argument("--result-dir", default="adaptive_attack_results/response_judge_llama3_kl_0.9")
    ap.add_argument("--ids", default="6,7,8,9,10")
    ap.add_argument(
        "--best-prompt-out",
        default=None,
        help="Optional detailed output CSV. If omitted, script only writes round-average CSV.",
    )
    ap.add_argument("--round-avg-out", default=None)
    ap.add_argument(
        "--best-by",
        choices=["judge", "ss", "all"],
        default="all",
        help="Selection metric. 'all' (default) saves judge/ss and tied/non-tied round-average files.",
    )
    ap.add_argument(
        "--include-tied-best",
        action="store_true",
        help="Keep all prompts tied at highest selected metric for each (id, checkpoint_round).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    ids = parse_ids(args.ids)

    rows, missing = collect_checkpoint_rows(args.result_dir, ids)
    if args.best_by == "all":
        modes = [
            ("judge", False),
            ("judge", True),
            ("ss", False),
            ("ss", True),
            ("oracle", False),
            ("oracle", True),
        ]
    else:
        modes = [(args.best_by, args.include_tied_best)]

    for best_by, include_tied_best in modes:
        best_rows = build_best_prompt_rows(
            rows,
            include_tied_best=include_tied_best,
            best_by=best_by,
        )
        if args.round_avg_out:
            round_avg_out = add_mode_suffix(args.round_avg_out, best_by, include_tied_best)
        else:
            round_avg_out = build_round_avg_default_path(args.result_dir, best_by, include_tied_best)
        write_csv(round_avg_out, build_round_avg_from_best_rows(best_rows))

        if args.best_prompt_out:
            if args.best_by == "all":
                best_prompt_out = add_mode_suffix(args.best_prompt_out, best_by, include_tied_best)
            else:
                best_prompt_out = args.best_prompt_out
            write_csv(best_prompt_out, best_rows)
            print(f"Saved best prompt table: {best_prompt_out}")
        else:
            print(f"Skipped best prompt table for {best_by} tied={include_tied_best}.")
        print(f"Saved round average table: {round_avg_out}")
        print(f"Best prompt rows: {len(best_rows)}")

    print(f"Best selection metric mode: {args.best_by}")
    if args.best_by != "all":
        print(f"Include tied best: {args.include_tied_best}")
    if missing:
        print("Missing:")
        for item in missing:
            print(f"- {item}")


if __name__ == "__main__":
    main()
