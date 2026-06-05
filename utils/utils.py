import os
import sys
import logging
import random
import numpy as np
import torch
import torch.distributed as dist


def configure_distributed_runtime():
    """Apply conservative NCCL defaults before process-group init.

    Some multi-GPU Linux stacks segfault during DDP/NCCL setup even before the
    first forward pass. Default to safer transport settings unless the user
    explicitly opts out with SMART_SAFE_NCCL=0.
    """
    if not torch.cuda.is_available():
        return
    if os.environ.get('SMART_SAFE_NCCL', '1') != '1':
        return
    safe_env = {
        'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
        'TORCH_NCCL_BLOCKING_WAIT': '1',
        'NCCL_P2P_DISABLE': '1',
        'NCCL_IB_DISABLE': '1',
    }
    for key, value in safe_env.items():
        os.environ.setdefault(key, value)


def init_logging(log_root, models_root=None):
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    if models_root is not None:
        handler_file = logging.FileHandler(
            os.path.join(models_root, "training.log"))
        handler_file.setFormatter(formatter)
        log_root.addHandler(handler_file)
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_stream)


def distributed_init(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', getattr(args, 'local_rank', 0)))
        args.distributed = True
        if torch.cuda.is_available():
            configure_distributed_runtime()
            visible_gpus = torch.cuda.device_count()
            if args.local_rank >= visible_gpus:
                raise RuntimeError(
                    f"LOCAL_RANK={args.local_rank} but only {visible_gpus} CUDA device(s) are visible. "
                    "Check CUDA_VISIBLE_DEVICES/DEVICES and --nproc-per-node."
                )
            args.gpu = args.local_rank
            torch.cuda.set_device(args.gpu)
            backend = 'nccl'
        else:
            args.gpu = None
            backend = 'gloo'
        args.dist_backend = backend
        dist.init_process_group(backend=backend, init_method='env://',
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = getattr(args, 'local_rank', 0)
        if torch.cuda.is_available():
            args.gpu = getattr(args, 'gpu', args.local_rank)
            torch.cuda.set_device(args.gpu)
        else:
            args.gpu = None
        args.distributed = False
        args.dist_backend = None


def configure_torch_runtime():
    """Prefer stable attention kernels over the fastest CUDA SDPA variants.

    This codebase uses scaled_dot_product_attention with custom masks/shapes in
    several blocks. On some multi-GPU CUDA stacks, flash or mem-efficient SDPA
    can segfault. Force the math kernel unless the user explicitly opts out.
    """
    if not torch.cuda.is_available():
        return
    disable_fast_sdpa = os.environ.get('SMART_DISABLE_FAST_SDPA', '1') == '1'
    if not disable_fast_sdpa:
        return
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
        torch.backends.cuda.enable_math_sdp(True)


def resolve_dataloader_workers(args):
    """Choose a conservative worker count that improves throughput by default.

    The default aims to speed up host-side collation/loading without oversubscribing
    CPU cores when running multi-process DDP. Users can override it with
    `--num-workers` or `SMART_NUM_WORKERS`.
    """
    env_workers = os.environ.get('SMART_NUM_WORKERS')
    if env_workers is not None:
        return max(0, int(env_workers))
    user_workers = getattr(args, 'num_workers', None)
    if user_workers is not None:
        return max(0, user_workers)
    cpu_count = os.cpu_count() or 1
    world_size = max(1, getattr(args, 'world_size', 1))
    per_rank_budget = max(1, cpu_count // world_size)
    return min(4, per_rank_budget)


def build_dataloader_kwargs(args):
    """Return safe DataLoader kwargs that improve throughput without changing results."""
    num_workers = resolve_dataloader_workers(args)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return kwargs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    device = length.device if device is None else device
    mask = torch.arange(max_len,
                        device=device, dtype=length.dtype).expand(
                            len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask
