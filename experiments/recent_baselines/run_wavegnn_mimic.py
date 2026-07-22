"""Run the official WaveGNN architecture on SMILE's MIMIC cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external" / "wavegnn"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from data.dataloader import collate_fn  # noqa: E402
from experiments.recent_baselines.baseline_utils import (  # noqa: E402
    add_sensor_robustness_args,
    binary_metrics,
    binary_prediction_arrays,
    json_ready_config,
    load_mimic_splits,
    make_loaders,
    maybe_run_sensor_grid,
    set_seed,
    upstream_commit,
    write_result,
)

try:  # noqa: E402
    from model.model import WaveGNN
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in environment setup
    if exc.name == "torch_geometric":
        raise SystemExit(
            "WaveGNN requires torch-geometric; install the baseline dependencies first."
        ) from exc
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("mimic_mortality", "mimic_decompensation", "c12", "c19"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "export" / "recent_baselines"
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load best_auprc.pt and evaluate only.")
    parser.add_argument("--eval-output-dir", type=Path, default=None,
                        help="Write eval result here instead of the checkpoint dir.")
    add_sensor_robustness_args(parser)
    return parser.parse_args()


def relative_timestamps(timestamps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Vectorized form of upstream get_relative_timestamps."""
    delta = torch.zeros_like(mask, dtype=timestamps.dtype)
    reference = timestamps - timestamps.amin(dim=1, keepdim=True)
    delta[:, 0] = reference[:, 0].unsqueeze(-1)
    for index in range(1, timestamps.shape[1]):
        step = (reference[:, index] - reference[:, index - 1]).unsqueeze(-1)
        delta[:, index] = torch.where(
            mask[:, index - 1].bool(), step, step + delta[:, index - 1]
        )
    return delta


def prepare_batch(batch, device: torch.device):
    values = torch.nan_to_num(batch["x"].float()).to(device, non_blocking=True)
    mask = batch["mask"].float().to(device, non_blocking=True)
    values = values * mask
    timestamps = batch["time"].float().to(device, non_blocking=True)
    labels = batch["labels"].float().view(-1, 1).to(device, non_blocking=True)
    relative = relative_timestamps(timestamps, mask).unsqueeze(-1)
    static = torch.empty(values.shape[0], 0, device=device)
    return values.unsqueeze(-1), mask, timestamps, static, relative, labels


@torch.no_grad()
def evaluate(model, loader, device: torch.device, *, return_predictions=False):
    model.eval()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for batch in loader:
        values, mask, timestamps, static, relative, labels = prepare_batch(batch, device)
        logits = model(values, mask, timestamps, static, relative)
        labels_all.append(labels.cpu().numpy().reshape(-1))
        scores_all.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    if return_predictions:
        return binary_prediction_arrays(labels_all, scores_all)
    return binary_metrics(labels_all, scores_all)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("WaveGNN local runs require CUDA.")
    device = torch.device("cuda")
    set_seed(args.seed)
    splits = load_mimic_splits(
        args.dataset,
        args.split_seed,
        (args.limit_train, args.limit_val, args.limit_test),
    )
    train_loader, val_loader, test_loader = make_loaders(
        splits, args.batch_size, collate_fn, args.num_workers
    )
    sample = splits[0][0]
    n_sensors = int(np.asarray(sample["x"]).shape[-1])
    window_size = int(np.asarray(sample["x"]).shape[0])
    model_args = SimpleNamespace(
        n_classes=1,
        window_size=window_size,
        dropout=args.dropout,
        num_attention_heads=args.num_heads,
        observation_dim=1,
        device=str(device),
        positional_encoding="relative_t2v",
        dataset="MIMIC3-IHM" if args.dataset == "mimic_mortality" else "MIMIC3-DECOMP",
    )
    model = WaveGNN(n_sensors, 0, args.hidden, model_args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=1, min_lr=1e-8
    )
    criterion = nn.BCEWithLogitsLoss()
    # torch.amp.GradScaler was introduced after the target server's PyTorch.
    # The CUDA namespace works on both the older and current supported builds.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    output_dir = args.output_root / args.dataset / "wavegnn" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_auprc.pt"
    best_auprc = -float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, (0 if args.eval_only else args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            values, mask, timestamps, static, relative, labels = prepare_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(values, mask, timestamps, static, relative)
                raw_loss = criterion(logits, labels)
                loss = raw_loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(raw_loss.detach()) * labels.numel()
            seen += labels.numel()

        val = evaluate(model, val_loader, device)
        scheduler.step(val["auprc"])
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "val_auroc": val["auroc"],
            "val_auprc": val["auprc"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val["auprc"] > best_auprc:
            best_auprc = val["auprc"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    if maybe_run_sensor_grid(
        args,
        evaluator=lambda loader: evaluate(
            model, loader, device, return_predictions=True
        ),
        test_dataset=splits[2],
        base_collate=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=output_dir,
        model_name="wavegnn",
        checkpoint_path=checkpoint_path,
    ):
        return
    test = evaluate(model, test_loader, device)
    result = {
        "model": "WaveGNN",
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "upstream_commit": upstream_commit(UPSTREAM),
        "upstream_url": "https://github.com/USC-InfoLab/WaveGNN",
        "adapter_notes": [
            "official WaveGNN architecture",
            "SMILE split, normalization, observation mask, and timestamps",
            "checkpoint selected by validation AUPRC",
        ],
        "selection_metric": "validation AUPRC",
        "best_validation_auprc": best_auprc,
        "test": test,
        "history": history,
        "config": json_ready_config(args),
    }
    write_result(result, args.eval_output_dir or output_dir)
    print(json.dumps(test, indent=2), flush=True)


if __name__ == "__main__":
    main()
