"""Run the official MissTSM-MAE classifier on SMILE's MIMIC cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external" / "misstm"
IMTS = UPSTREAM / "classification" / "IMTS"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(IMTS) not in sys.path:
    sys.path.insert(0, str(IMTS))

# Upstream imports wandb in model.py but never uses it in the classifier.  Keep
# experiment tracking optional instead of pulling it into the execution path.
if "wandb" not in sys.modules:
    sys.modules["wandb"] = ModuleType("wandb")

from data.dataloader import collate_fn  # noqa: E402
from experiments.recent_baselines.baseline_utils import (  # noqa: E402
    add_sensor_robustness_args,
    balanced_sampler,
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
    from model import MaskedAutoencoder
    from positional_encodings.torch_encodings import PositionalEncoding2D
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in setup
    raise SystemExit(
        f"MissTSM dependency {exc.name!r} is missing; install baseline dependencies first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("mimic_mortality", "mimic_decompensation", "c12", "c19"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-embed-dim", type=int, default=64)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-num-heads", type=int, default=2)
    parser.add_argument("--decoder-embed-dim", type=int, default=32)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pct-start", type=float, default=0.3)
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


class PositionalHolder:
    def __init__(self, seq_len: int, num_feats: int, embed_dim: int, device):
        encoder = PositionalEncoding2D(embed_dim).to(device)
        seed = torch.zeros(1, seq_len + 1, num_feats, embed_dim, device=device)
        self.pos_embed = encoder(seed).detach()


def model_options(args, seq_len: int):
    # MaskedAutoencoder reads this argparse-like object directly.
    args.task_name = "finetune"
    args.seq_len = seq_len
    args.norm_field_loss = False
    return args


def prepare_batch(batch, device: torch.device):
    values = torch.nan_to_num(batch["x"].float()).to(device, non_blocking=True)
    observed = batch["mask"].float().to(device, non_blocking=True)
    values = values * observed
    labels = batch["labels"].long().view(-1).to(device, non_blocking=True)
    return values, observed, labels


@torch.no_grad()
def evaluate(model, holder, loader, device: torch.device, *, return_predictions=False):
    model.eval()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for batch in loader:
        values, observed, labels = prepare_batch(batch, device)
        logits = model(values, observed, holder)
        labels_all.append(labels.cpu().numpy())
        scores_all.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    if return_predictions:
        return binary_prediction_arrays(labels_all, scores_all)
    return binary_metrics(labels_all, scores_all)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MissTSM local runs require CUDA.")
    device = torch.device("cuda")
    set_seed(args.seed)
    splits = load_mimic_splits(
        args.dataset,
        args.split_seed,
        (args.limit_train, args.limit_val, args.limit_test),
    )
    train_loader, val_loader, test_loader = make_loaders(
        splits,
        args.batch_size,
        collate_fn,
        args.num_workers,
        train_sampler=balanced_sampler(splits[0]),
    )
    sample = splits[0][0]
    seq_len, num_feats = np.asarray(sample["x"]).shape
    options = model_options(args, int(seq_len))
    holder = PositionalHolder(seq_len, num_feats, args.encoder_embed_dim, device)
    model = MaskedAutoencoder(options, num_feats=num_feats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        steps_per_epoch=len(train_loader),
        pct_start=args.pct_start,
        epochs=args.epochs,
        max_lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()
    # Keep compatibility with the target host's pre-torch.amp PyTorch build.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    output_dir = args.output_root / args.dataset / "misstm" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_auprc.pt"
    best_auprc = -float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, (0 if args.eval_only else args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            values, observed, labels = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(values, observed, holder)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * labels.numel()
            seen += labels.numel()

        val = evaluate(model, holder, val_loader, device)
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
            model, holder, loader, device, return_predictions=True
        ),
        test_dataset=splits[2],
        base_collate=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=output_dir,
        model_name="misstm",
        checkpoint_path=checkpoint_path,
    ):
        return
    test = evaluate(model, holder, test_loader, device)
    result = {
        "model": "MissTSM",
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "upstream_commit": upstream_commit(UPSTREAM),
        "upstream_url": "https://github.com/abhilash-neog/SparseTimeSeriesModeling",
        "adapter_notes": [
            "official MissTSM-MAE classification architecture",
            "SMILE split, normalization, and observation mask",
            "class-balanced training sampler for the imbalanced MIMIC endpoint",
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
