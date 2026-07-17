"""Two-process CPU smoke for ODP DDP invariants.

Run with ``python -m torch.distributed.run --nproc_per_node 2``.  This does not
replace the required two-4090 NCCL smoke; it validates rank-independent metric
aggregation and DDP/non-DDP checkpoint compatibility without GPU access.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main_finetune
from models.odp import ODPLateFusionEncoder
from utils.odp_metrics import forecasting_metrics, gather_forecast_records


def args():
    return SimpleNamespace(
        input_dim=3, d_model=8, num_class=2, dropout=0.0, max_len=4,
        e_layers=1, n_heads=2, time_dim=16, obs_density_window=5,
        policy_hidden_dim=64, policy_kernel_size=7, seed=42,
    )


def record(sample_id):
    target = np.asarray([[0, 1, 0], [sample_id % 2, 0, 1],
                         [1 - sample_id % 2, 1, 0]], dtype=np.float32)
    prob = np.clip(0.15 + 0.7 * target, 1e-4, 1 - 1e-4)
    return {"sample_id": sample_id, "target": target, "prob": prob}


def run_worker(rank=None, world=2, port=31979):
    if rank is None:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world = dist.get_world_size()
        token = os.environ.get("MASTER_PORT", "unknown")
    else:
        from datetime import timedelta
        store = dist.TCPStore(
            "127.0.0.1", port, world, rank == 0,
            timeout=timedelta(seconds=30), use_libuv=False,
        )
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world)
        token = str(port)
    if world != 2:
        raise AssertionError(f"expected two ranks, received {world}")

    torch.manual_seed(42)
    ddp_model = DistributedDataParallel(ODPLateFusionEncoder(args()))
    batch = 2
    mask = torch.randint(0, 2, (batch, 4, 3)).float()
    payload = {
        "x": torch.randn(batch, 4, 3), "mask": mask, "original_mask": mask,
        "time": torch.arange(1, 5).float().repeat(batch, 1),
        "lens": torch.tensor([4, 3]),
    }
    output = ddp_model(**payload)
    output[:, :, 0].sum().backward()

    root = os.path.join(tempfile.gettempdir(), f"smile_odp_ddp_smoke_{token}")
    checkpoint = os.path.join(root, "checkpoint.pth")
    result_path = os.path.join(root, "result.json")
    if rank == 0:
        os.makedirs(root, exist_ok=False)
        torch.save({"encoder": ddp_model.state_dict()}, checkpoint)
    dist.barrier()

    state = torch.load(checkpoint, weights_only=False)["encoder"]
    plain = ODPLateFusionEncoder(args())
    main_finetune._load_model_state_dict(plain, state)
    for key, value in ddp_model.module.state_dict().items():
        torch.testing.assert_close(value, plain.state_dict()[key])

    # Deliberately uneven shards with sample 4 repeated as sampler padding.
    ids = [0, 2, 4] if rank == 0 else [1, 3, 4]
    gathered = gather_forecast_records([record(sample_id) for sample_id in ids])
    if [item["sample_id"] for item in gathered] != [0, 1, 2, 3, 4]:
        raise AssertionError("cross-rank sample-id de-duplication failed")
    distributed_metrics = forecasting_metrics(gathered)
    single_metrics = forecasting_metrics([record(i) for i in range(5)])
    if distributed_metrics != single_metrics:
        raise AssertionError("single-process and two-process metrics differ")
    local_ids = torch.tensor(ids)
    local_labels = (local_ids % 2).long()
    local_preds = torch.stack((1.0 - local_labels.float(), local_labels.float()), dim=1)
    task_labels, task_preds = main_finetune._gather_task_records(
        local_labels, local_preds, local_ids)
    if task_labels.shape[0] != 5 or task_preds.shape[0] != 5:
        raise AssertionError("downstream DDP task output de-duplication failed")

    if rank == 0:
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump({
                "world_size": world,
                "unique_samples": len(gathered),
                "metrics_equal": True,
                "rank0_only_checkpoint": True,
            }, handle, indent=2)
        print(json.dumps(json.load(open(result_path, encoding="utf-8")), sort_keys=True))
    dist.barrier()
    if rank == 0:
        os.remove(result_path)
        os.remove(checkpoint)
        os.rmdir(root)
    dist.destroy_process_group()


def main():
    if "RANK" in os.environ:
        run_worker()
    else:
        # Windows fallback for PyTorch builds whose torchrun TCPStore requires
        # unavailable libuv.  The official Linux/NCCL command remains torchrun.
        mp.spawn(run_worker, args=(2, 31979), nprocs=2, join=True)


if __name__ == "__main__":
    main()
