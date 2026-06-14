#!/usr/bin/env python3
import argparse
import csv
import os


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def build_surrogate_means(rows):
    grouped = {}
    for row in rows:
        if str(row.get("rank", "")) != "1":
            continue
        checkpoint_round = int(row["checkpoint_round"])
        grouped.setdefault(checkpoint_round, {"ss": [], "pls": []})
        grouped[checkpoint_round]["ss"].append(safe_float(row.get("surrogate_ss_to_system_prompt", 0.0)))
        grouped[checkpoint_round]["pls"].append(safe_float(row.get("surrogate_pls_to_system_prompt", 0.0)))

    out = {}
    for checkpoint_round, values in grouped.items():
        ss_vals = values["ss"]
        pls_vals = values["pls"]
        out[checkpoint_round] = {
            "avg_surrogate_ss": sum(ss_vals) / len(ss_vals) if ss_vals else 0.0,
            "avg_surrogate_pls": sum(pls_vals) / len(pls_vals) if pls_vals else 0.0,
            "n_surrogates": len(ss_vals),
        }
    return out


def build_target_means(rows):
    out = {}
    for row in rows:
        checkpoint_round = int(row["checkpoint_round"])
        out[checkpoint_round] = {
            "avg_target_ss": safe_float(row.get("avg_target_ss_across_surrogates", 0.0)),
            "avg_target_pls": safe_float(row.get("avg_target_pls_across_surrogates", 0.0)),
            "n_transfer_rows": int(float(row.get("n_transfer_rows", 0) or 0)),
        }
    return out


def merge_rows(surrogate_means, target_means):
    checkpoint_rounds = sorted(set(surrogate_means) | set(target_means))
    merged = []
    for checkpoint_round in checkpoint_rounds:
        s = surrogate_means.get(checkpoint_round, {})
        t = target_means.get(checkpoint_round, {})
        merged.append(
            {
                "checkpoint_round": checkpoint_round,
                "avg_surrogate_ss": s.get("avg_surrogate_ss", 0.0),
                "avg_surrogate_pls": s.get("avg_surrogate_pls", 0.0),
                "avg_target_ss": t.get("avg_target_ss", 0.0),
                "avg_target_pls": t.get("avg_target_pls", 0.0),
                "n_surrogates": s.get("n_surrogates", 0),
                "n_transfer_rows": t.get("n_transfer_rows", 0),
            }
        )
    return merged


def write_csv(path, rows):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge surrogate and target average SS/PLS by checkpoint round."
    )
    ap.add_argument(
        "--surrogate-checkpoint-summary",
        default="adaptive_attack_results/surrogate_ss_llama3/surrogate_checkpoint_summary.csv",
    )
    ap.add_argument(
        "--overall-transfer-summary",
        default="adaptive_attack_results/surrogate_ss_llama3/overall_transfer_summary.csv",
    )
    ap.add_argument(
        "--out-csv",
        default="adaptive_attack_results/surrogate_ss_llama3/surrogate_target_round_summary.csv",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    surrogate_rows = read_csv_rows(args.surrogate_checkpoint_summary)
    target_rows = read_csv_rows(args.overall_transfer_summary)

    surrogate_means = build_surrogate_means(surrogate_rows)
    target_means = build_target_means(target_rows)
    merged = merge_rows(surrogate_means, target_means)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_csv(args.out_csv, merged)

    print(f"Saved merged summary to {args.out_csv}")
    for row in merged:
        print(
            f"round={row['checkpoint_round']}: "
            f"surrogate_ss={float(row['avg_surrogate_ss']):.4f}, "
            f"surrogate_pls={float(row['avg_surrogate_pls']):.4f}, "
            f"target_ss={float(row['avg_target_ss']):.4f}, "
            f"target_pls={float(row['avg_target_pls']):.4f}"
        )


if __name__ == "__main__":
    main()
