"""Orchestrate frozen-checkpoint sensor-removal evaluation grids."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sensor_robustness import DEFAULT_KS

DATASETS = ("mimic_mortality", "mimic_decompensation", "c12", "c19")
SEEDS = (1, 42, 3407)
PAPER_MODELS = ("smile", "smart", "ists_plm", "wavegnn", "misstm", "atenet")
MECHANISM_MODELS = (
    "smile",
    "smart",
    "smile_no_mnar_bias",
    "smile_no_density",
    "smile_no_curriculum",
)
MAIN_MODELS = {
    "smile": "smart-smile-lean",
    "smart": "smart",
    "smile_no_mnar_bias": "smart-smile-lean-no-mnar-bias",
    "smile_no_density": "smart-smile-lean-no-density",
    "smile_no_curriculum": "smart-smile-lean-norandom",
}
BASELINE_SCRIPTS = {
    "ists_plm": ("run_ists_plm_mimic.py", ["--batch-size", "8"]),
    "wavegnn": ("run_wavegnn_mimic.py", ["--batch-size", "8"]),
    "misstm": ("run_misstm_mimic.py", ["--batch-size", "16"]),
    "atenet": ("run_atenet_mimic.py", ["--batch-size", "16"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("paper", "mechanism"), default="paper")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--replicates", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--devices", nargs="+", default=None,
        help="Physical GPU IDs. One serial worker is launched per device, e.g. --devices 0 1.",
    )
    parser.add_argument(
        "--gpu", default=None,
        help="Deprecated single-device alias; prefer --devices.",
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "export"
    )
    parser.add_argument(
        "--baseline-checkpoint-root",
        type=Path,
        default=ROOT / "export" / "recent_baselines",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=ROOT / "export" / "sensor_robustness_v1" / "manifests",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "export" / "sensor_robustness_v1",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-incomplete-smoke", action="store_true")
    return parser.parse_args()


def resolved_devices(args: argparse.Namespace) -> tuple[str, ...]:
    if args.devices:
        devices = tuple(str(value) for value in args.devices)
    elif args.gpu is not None:
        devices = (str(args.gpu),)
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        devices = tuple(value.strip() for value in visible.split(",") if value.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(f"GPU devices must be a non-empty unique list: {devices}")
    if any("," in value for value in devices):
        raise ValueError("Pass device IDs separately, for example --devices 0 1")
    return devices


def validate_local_devices(devices: tuple[str, ...]) -> None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi failed; cannot validate requested GPUs") from exc
    available = {line.strip() for line in output.splitlines() if line.strip()}
    missing = sorted(set(devices) - available)
    if missing:
        raise ValueError(f"Requested GPU IDs {missing} are unavailable; visible physical IDs: {sorted(available)}")


def profile_models(args: argparse.Namespace) -> tuple[str, ...]:
    allowed = PAPER_MODELS if args.profile == "paper" else MECHANISM_MODELS
    models = tuple(args.models) if args.models else allowed
    unknown = sorted(set(models) - set(allowed))
    if unknown:
        raise ValueError(f"Models not in {args.profile} profile: {unknown}")
    return models


def manifest_path(args: argparse.Namespace, dataset: str) -> Path:
    return args.manifest_root / f"{dataset}_test_seed{args.split_seed}.npz"


def checkpoint_path(args: argparse.Namespace, dataset: str, model: str, seed: int) -> Path:
    if model in MAIN_MODELS:
        return (
            args.checkpoint_root
            / dataset
            / MAIN_MODELS[model]
            / f"seed_{seed}"
            / "checkpoint-prc.pth"
        )
    return (
        args.baseline_checkpoint_root
        / dataset
        / model
        / f"seed_{seed}"
        / "best_auprc.pt"
    )


def checkpoint_problem(path: Path) -> str | None:
    if not path.exists():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    if path.suffix in (".pt", ".pth") and not zipfile.is_zipfile(path):
        return "not a readable PyTorch zip checkpoint"
    return None


def build_command(
    args: argparse.Namespace, dataset: str, model: str, seed: int
) -> tuple[list[str], Path]:
    manifest = manifest_path(args, dataset)
    runs_root = args.output_root / "runs"
    common_grid = [
        "--sensor-manifest", str(manifest),
        "--sensor-ks", *[str(value) for value in args.ks],
        "--sensor-replicates", *[str(value) for value in args.replicates],
    ]
    if args.resume:
        common_grid.append("--sensor-resume")
    if model in MAIN_MODELS:
        cmd = [
            args.python,
            str(ROOT / "run_all_experiments.py"),
            "--models", MAIN_MODELS[model],
            "--datasets", dataset,
            "--seeds", str(seed),
            "--eval-only",
            "--export-root", str(args.checkpoint_root),
            "--eval-output-root", str(runs_root),
            "--split-seed", str(args.split_seed),
            *common_grid,
        ]
        if args.limit_test:
            cmd.extend(["--limit-test", str(args.limit_test)])
        destination = runs_root / dataset / MAIN_MODELS[model] / f"seed_{seed}"
        return cmd, destination

    script, extras = BASELINE_SCRIPTS[model]
    destination = runs_root / dataset / model / f"seed_{seed}"
    cmd = [
        args.python,
        str(ROOT / "experiments" / "recent_baselines" / script),
        "--dataset", dataset,
        "--seed", str(seed),
        "--split-seed", str(args.split_seed),
        "--output-root", str(args.baseline_checkpoint_root),
        "--eval-only",
        "--eval-output-dir", str(destination),
        *extras,
        *common_grid,
    ]
    if args.limit_test:
        cmd.extend(["--limit-test", str(args.limit_test)])
    return cmd, destination


def main() -> None:
    args = parse_args()
    devices = resolved_devices(args)
    if not args.dry_run:
        validate_local_devices(devices)
    models = profile_models(args)
    planned = [
        (dataset, model, seed)
        for dataset in args.datasets
        for model in models
        for seed in args.seeds
    ]
    problems = []
    for dataset in args.datasets:
        manifest = manifest_path(args, dataset)
        if not manifest.exists():
            problems.append({"dataset": dataset, "manifest": str(manifest), "problem": "missing"})
    for dataset, model, seed in planned:
        path = checkpoint_path(args, dataset, model, seed)
        problem = checkpoint_problem(path)
        if problem:
            problems.append(
                {"dataset": dataset, "model": model, "seed": seed,
                 "checkpoint": str(path), "problem": problem}
            )

    if problems:
        print(json.dumps({"preflight_problems": problems}, indent=2))
        if not args.allow_incomplete_smoke:
            raise SystemExit("Preflight failed; use --allow-incomplete-smoke only for plumbing checks")

    failures = []
    runnable = []
    for dataset, model, seed in planned:
        path = checkpoint_path(args, dataset, model, seed)
        if checkpoint_problem(path) or not manifest_path(args, dataset).exists():
            failures.append({"dataset": dataset, "model": model, "seed": seed, "status": "skipped"})
            continue
        runnable.append((dataset, model, seed))

    if args.dry_run:
        for index, (dataset, model, seed) in enumerate(runnable):
            device = devices[index % len(devices)]
            cmd, _ = build_command(args, dataset, model, seed)
            rendered = subprocess.list2cmdline(cmd)
            print(
                f"[gpu={device} {dataset}/{model}/seed_{seed}] {rendered}",
                flush=True,
            )
        completed_jobs = []
    else:
        task_queue: queue.Queue = queue.Queue()
        for task in runnable:
            task_queue.put(task)
        completed_jobs = []
        lock = threading.Lock()
        stop_event = threading.Event()

        def worker(device: str) -> None:
            while not stop_event.is_set():
                try:
                    dataset, model, seed = task_queue.get_nowait()
                except queue.Empty:
                    return
                cmd, destination = build_command(args, dataset, model, seed)
                rendered = subprocess.list2cmdline(cmd)
                print(
                    f"[gpu={device} {dataset}/{model}/seed_{seed}] {rendered}",
                    flush=True,
                )
                log_path = args.output_root / "logs" / dataset / model / f"seed_{seed}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = device
                started = time.monotonic()
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                record = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "gpu": device,
                    "returncode": completed.returncode,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "log": str(log_path),
                    "destination": str(destination),
                }
                with lock:
                    completed_jobs.append(record)
                    if completed.returncode:
                        failures.append({**record, "status": "failed"})
                task_queue.task_done()
                if completed.returncode and not args.allow_incomplete_smoke:
                    stop_event.set()
                    return

        workers = [threading.Thread(target=worker, args=(device,)) for device in devices]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        while True:
            try:
                dataset, model, seed = task_queue.get_nowait()
            except queue.Empty:
                break
            failures.append(
                {"dataset": dataset, "model": model, "seed": seed, "status": "cancelled"}
            )
            task_queue.task_done()

    # A single parent writes the summary after every GPU worker has finished.
    # This avoids the output races caused by launching independent orchestrators.
    summary = {
        "profile": args.profile,
        "datasets": args.datasets,
        "models": list(models),
        "seeds": args.seeds,
        "ks": args.ks,
        "replicates": args.replicates,
        "devices": list(devices),
        "limit_test": args.limit_test,
        "preflight_problems": problems,
        "completed_jobs": completed_jobs,
        "failures": failures,
        "complete": not problems and not failures,
    }
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        summaries = args.output_root / "execution_summaries"
        summaries.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
        history_path = summaries / f"{run_id}_{args.profile}.json"
        history_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary = args.output_root / f"execution_summary.json.tmp-{os.getpid()}"
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.output_root / "execution_summary.json")
    if failures and not args.allow_incomplete_smoke:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
