"""Queue missing MIMIC runs for recent source-released baselines.

Independent model/seed jobs are assigned one per visible GPU because the
upstream implementations are single-process.  This preserves each official
model's execution semantics while allowing seed-level parallelism.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "wavegnn": ROOT / "experiments" / "recent_baselines" / "run_wavegnn_mimic.py",
    "misstm": ROOT / "experiments" / "recent_baselines" / "run_misstm_mimic.py",
    "ists_plm": ROOT / "experiments" / "recent_baselines" / "run_ists_plm_mimic.py",
    "atenet": ROOT / "experiments" / "recent_baselines" / "run_atenet_mimic.py",
}
TARGET_HOST_ARGS = {
    "wavegnn": ["--batch-size", "4", "--gradient-accumulation-steps", "32"],
    "misstm": ["--batch-size", "16"],
    "ists_plm": ["--batch-size", "6"],
    "atenet": ["--batch-size", "16"],
}
DATASETS = ("mimic_mortality", "mimic_decompensation", "c12", "c19")
SEEDS = (1, 42, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=SCRIPTS, default=list(SCRIPTS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-8gb", action="store_true")
    parser.add_argument(
        "--source-missing-only",
        action="store_true",
        help="Skip WaveGNN mortality, whose source paper already reports MIMIC IHM.",
    )
    return parser.parse_args()


def command_for(model: str, dataset: str, seed: int, args: argparse.Namespace):
    command = [
        args.python,
        str(SCRIPTS[model]),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
    ]
    if not args.local_8gb:
        command.extend(TARGET_HOST_ARGS[model])
    return command


def run_job(job, gpu: str, args: argparse.Namespace) -> tuple[str, int]:
    model, dataset, seed = job
    name = f"{model}_{dataset}_seed{seed}"
    result_path = (
        ROOT
        / "export"
        / "recent_baselines"
        / dataset
        / model
        / f"seed_{seed}"
        / "eval_results.json"
    )
    if result_path.exists() and not args.force:
        return name + " (already complete)", 0
    command = command_for(model, dataset, seed, args)
    rendered = subprocess.list2cmdline(command)
    if args.dry_run:
        print(f"CUDA_VISIBLE_DEVICES={gpu} {rendered}")
        return name, 0

    log_dir = ROOT / "logs" / "recent_baselines"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return name + f" (log: {log_path})", completed.returncode


def run_gpu_queue(gpu: str, jobs, args: argparse.Namespace):
    return [run_job(job, gpu, args) for job in jobs]


def main() -> None:
    args = parse_args()
    jobs = [
        (model, dataset, seed)
        for model in args.models
        for dataset in args.datasets
        for seed in args.seeds
        if not (
            args.source_missing_only
            and model == "wavegnn"
            and dataset == "mimic_mortality"
        )
    ]
    if args.dry_run:
        for index, job in enumerate(jobs):
            run_job(job, args.gpus[index % len(args.gpus)], args)
        return

    queues = {gpu: [] for gpu in args.gpus}
    for index, job in enumerate(jobs):
        queues[args.gpus[index % len(args.gpus)]].append(job)
    failures = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(run_gpu_queue, gpu, queue, args): gpu
            for gpu, queue in queues.items()
        }
        for future in as_completed(futures):
            for name, returncode in future.result():
                print(f"[{returncode}] {name}", flush=True)
                if returncode:
                    failures.append(name)
    if failures:
        raise SystemExit(f"{len(failures)} baseline jobs failed")


if __name__ == "__main__":
    main()
