
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ATTACK_FILE_ORDER = [
    "Semantic_Collision.json",
    "Long_Prefix.json",
    "Encoded_Leakage.json",
    "Refusal_Evasion.json",
]

def parse_args():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Run adaptive attack evaluation for selected soft prompts."
    )
    ap.add_argument("--project-root", type=Path, default=root)
    ap.add_argument("--test-script", type=Path, default=root / "test_soft_prompt.py")
    ap.add_argument("--checkpoint-root", type=Path, default=root / "llama3_checkpoint")
    ap.add_argument("--attack-dir", type=Path, default=root / "data" / "adaptive_attack")
    ap.add_argument("--result-dir", type=Path, default=root / "result")
    ap.add_argument("--start-id", type=int, default=31)
    ap.add_argument("--end-id", type=int, default=40)
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--gen-max-new", type=int, default=256)
    ap.add_argument("--inference-alpha", type=float, default=1.0)
    ap.add_argument("--compare-baseline", action="store_true")
    ap.add_argument(
        "--no-compare-original",
        action="store_true",
        help="Skip original output generation to reduce runtime.",
    )
    return ap.parse_args()


def build_type_jobs(args):
    ids = list(range(args.start_id, args.end_id + 1))
    for system_id in ids:
        run_dir = args.checkpoint_root / f"id_{system_id}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {run_dir}")

    jobs = []
    for filename in ATTACK_FILE_ORDER:
        attack_path = args.attack_dir / filename
        if not attack_path.exists():
            raise FileNotFoundError(f"Attack file not found: {attack_path}")
        jobs.append(
            {
                "attack_type": attack_path.stem,
                "attack_path": attack_path,
                "system_ids": ids,
            }
        )
    return jobs


def run_job(attack_type, attack_path, system_id, gpu_id, args):
    run_dir = args.checkpoint_root / f"id_{system_id}"

    raw_dir = args.result_dir / "raw" / attack_type
    log_dir = args.result_dir / "logs" / attack_type
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / f"id_{system_id}.csv"
    json_path = raw_dir / f"id_{system_id}.json"
    log_path = log_dir / f"id_{system_id}.log"

    cmd = [
        sys.executable,
        str(args.test_script),
        "--run-dir",
        str(run_dir),
        "--prompts-json",
        str(attack_path),
        "--device",
        "cuda",
        "--dtype",
        args.dtype,
        "--gen-max-new",
        str(args.gen_max_new),
        "--inference-alpha",
        str(args.inference_alpha),
        "--csv-out",
        str(csv_path),
        "--output",
        str(json_path),
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if not args.no_compare_original:
        cmd.append("--compare-original")
    if args.compare_baseline:
        cmd.append("--compare-baseline")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[gpu={gpu_id}] {' '.join(cmd)}\n\n")
        log_file.flush()
        subprocess.run(
            cmd,
            cwd=str(args.project_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    return {
        "attack_type": attack_type,
        "system_id": system_id,
        "csv_path": csv_path,
        "json_path": json_path,
        "log_path": log_path,
        "gpu_id": gpu_id,
    }


def run_type_job(gpu_id, job, args):
    finished = []
    attack_type = job["attack_type"]
    system_ids = job["system_ids"]
    print(
        f"[GPU {gpu_id}] Starting attack type {attack_type} for "
        f"id_{system_ids[0]}-id_{system_ids[-1]}",
        flush=True,
    )
    for system_id in system_ids:
        print(
            f"[GPU {gpu_id}] Running {attack_type} on id_{system_id}",
            flush=True,
        )
        finished.append(run_job(attack_type, job["attack_path"], system_id, gpu_id, args))
    return finished


def aggregate_results(result_dir, attack_types):
    for attack_type in attack_types:
        raw_dir = result_dir / "raw" / attack_type
        csv_paths = sorted(raw_dir.glob("id_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
        if not csv_paths:
            raise FileNotFoundError(f"No raw CSV outputs found for {attack_type} under {raw_dir}")

        frames = []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            df.insert(0, "system_id", int(csv_path.stem.split("_")[1]))
            df.insert(1, "attack_type", attack_type)
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(result_dir / f"{attack_type}.csv", index=False, encoding="utf-8")


def main():
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.test_script = args.test_script.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.attack_dir = args.attack_dir.resolve()
    args.result_dir = args.result_dir.resolve()

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided.")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_type_jobs(args)

    assignments = []
    for idx, job in enumerate(jobs):
        assigned_gpu = gpu_ids[idx % len(gpu_ids)]
        assignments.append((assigned_gpu, job))

    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [
            executor.submit(run_type_job, gpu_id, job, args)
            for gpu_id, job in assignments
        ]
        for future in futures:
            future.result()

    aggregate_results(args.result_dir, [p.stem for p in map(Path, ATTACK_FILE_ORDER)])
    print(f"Finished. Aggregated CSVs saved under {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
