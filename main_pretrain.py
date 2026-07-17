import argparse
import copy
import json
import os
import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm

from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import load_mimic_iii_mortality, load_mimic_iii_phenotyping, load_mimic_iii_decompensation, load_mimic_iii_lengthofstay
from data.dataloader import collate_fn, DeterministicShuffledHistoryDataset
from data.feature_registry import REGISTRY_VERSION, registry_fingerprint
from models.smart import EmbeddingDecoder
from models.odp import causal_policy_bce, POLICY_LOSS_WEIGHT
from utils.odp_metrics import (
    forecasting_metrics,
    gather_forecast_records,
    make_forecast_records,
    prior_records,
    training_variable_prior,
)
from utils.utils import (
    set_seed,
    distributed_init,
    init_logging,
    configure_torch_runtime,
    build_dataloader_kwargs,
)
from utils.variable_order import get_variable_order


def random_masking(x, original_mask, min_mask_ratio, max_mask_ratio):
    """
    Perform per-sample random masking.
    """
    N, L, V = x.shape  # batch, length, var

    # Calculate mask ratios and lengths to keep for each sample in the batch
    mask_ratios = torch.rand(N, device=x.device) * \
        (max_mask_ratio - min_mask_ratio) + min_mask_ratio

    mask = torch.rand_like(x) < mask_ratios.view(-1, 1, 1)
    x = x * (~mask)  # True for reconstruction, False for original
    return x, original_mask * (~mask),  original_mask * mask


def apply_pretrain_checkpoint_rule(args, uses_curriculum):
    """Keep legacy behavior while locking ODP to validation joint-loss selection."""
    if uses_curriculum and not args.save_last and args.odp_model == 'none':
        args.save_last = True
    if args.odp_model != 'none':
        args.save_last = False


# ── PMAE v1: Variable-level proportional masking ──────────────────────────────

def compute_variable_density(dataloader):
    """One-time scan of training set to compute per-variable observation fraction.
    Returns p_obs: (V,) float tensor on CPU, values in [0, 1].
    """
    total_obs = None
    total_steps = 0
    for batch in dataloader:
        m = batch['mask'].float()          # (B, T, V)
        obs = m.sum(dim=(0, 1)).cpu()      # (V,)
        total_obs = obs if total_obs is None else total_obs + obs
        total_steps += m.shape[0] * m.shape[1]
    return total_obs / (total_steps + 1e-8)


def compute_var_max_ratios(p_obs, ratio_min, ratio_max, temperature=1.0):
    """Maps per-variable observation density to per-variable max masking ratio.
    temperature=1.0: linear  max_ratio_v = ratio_min + (ratio_max - ratio_min) * p_obs_v
    temperature>1:   sigmoid-like, amplifies contrast between high/low density vars.
    Returns var_max_ratios: (V,) in [ratio_min, ratio_max]
    """
    if temperature == 1.0:
        f = p_obs.clamp(0.0, 1.0)
    else:
        raw = torch.sigmoid(temperature * (2.0 * p_obs - 1.0))
        lo = torch.sigmoid(torch.full_like(p_obs, -temperature))
        hi = torch.sigmoid(torch.full_like(p_obs, temperature))
        f = (raw - lo) / (hi - lo + 1e-8)
    return ratio_min + (ratio_max - ratio_min) * f


def proportional_random_masking(x, original_mask, min_mask_ratio, max_mask_ratio,
                                 var_max_ratios):
    """PMAE-style per-variable proportional masking.
    Drop-in replacement for random_masking() when --pretrain-mask-mode=proportional_var.

    Difference from random_masking:
      random_masking:          mask_ratios ~ (B,)   -> same ratio for all variables
      proportional_random_masking: sampled_ratios ~ (B, V) -> each variable sampled independently

    Args:
        x              : (B, T, V) float32
        original_mask  : (B, T, V) bool, 1=observed
        min_mask_ratio : float
        max_mask_ratio : float  (unused directly; upper bound encoded in var_max_ratios)
        var_max_ratios : (V,) float tensor from compute_var_max_ratios()

    Returns same triple as random_masking:
        x_masked, visible_mask, pretrain_mask
    """
    B, T, V = x.shape
    device = x.device
    rand_bv = torch.rand(B, V, device=device)                                    # (B, V)
    sampled_ratios = (min_mask_ratio +
                      (var_max_ratios.to(device) - min_mask_ratio) * rand_bv)   # (B, V)
    rand_btv = torch.rand(B, T, V, device=device)                                # (B, T, V)
    mask = rand_btv < sampled_ratios.unsqueeze(1)                                # (B, T, V)
    x = x * (~mask)
    return x, original_mask * (~mask), original_mask * mask


# ── Variable groupings (dynamically built from feature names) ─────────────────

def uses_structured_masking(args):
    """Return whether a run can invoke the audited system-masking branch."""
    uses_smile = (
        args.use_smile or args.use_smile_film or args.use_smile_v2
        or args.use_smile_v2_film or args.use_smile_lean or args.use_smile_lean_v2
        or args.odp_model != 'none'
    )
    if not uses_smile or args.use_mnar or args.use_smile_lean_samepretrain:
        return False
    if args.smile_no_curriculum:
        return False
    return args.smile_stratified or args.smile_mask_type != 'temporal'


def get_mask_system_groups(args):
    """Read and validate selected system groups from the audit artifact."""
    if not uses_structured_masking(args):
        return {}, None
    if not args.mask_group_config:
        raise ValueError(
            '--mask-group-config is required when structured/system masking is enabled.'
        )
    try:
        with open(args.mask_group_config, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f'Unable to load mask-group config {args.mask_group_config!r}: {exc}'
        ) from exc

    if payload.get('registry_version') != REGISTRY_VERSION:
        raise ValueError(
            f'Mask-group registry_version mismatch: config={payload.get("registry_version")!r}, '
            f'runtime={REGISTRY_VERSION!r}.'
        )
    if payload.get('selection_rule') != 'delta > 0.5':
        raise ValueError(
            f'Unsupported mask-group selection_rule: {payload.get("selection_rule")!r}.'
        )
    if payload.get('split') != 'train':
        raise ValueError(
            f'Mask-group config must be derived from split "train", got {payload.get("split")!r}.'
        )
    if payload.get('split_seed') != args.split_seed:
        raise ValueError(
            f'Mask-group split_seed mismatch: config={payload.get("split_seed")!r}, '
            f'runtime={args.split_seed!r}.'
        )

    task = payload.get('tasks', {}).get(args.dataset)
    if not isinstance(task, dict):
        raise ValueError(f'Mask-group config does not contain task {args.dataset!r}.')
    fingerprint = registry_fingerprint(args.dataset)
    if task.get('registry_fingerprint') != fingerprint:
        raise ValueError(
            f'Mask-group registry_fingerprint mismatch for {args.dataset}: '
            f'config={task.get("registry_fingerprint")!r}, runtime={fingerprint!r}.'
        )

    selected_groups = task.get('selected_groups')
    if not isinstance(selected_groups, dict):
        raise ValueError(f'Mask-group selected_groups for {args.dataset} must be an object.')
    system_groups = {}
    for group_name, indices in selected_groups.items():
        if not isinstance(group_name, str) or not isinstance(indices, list):
            raise ValueError(f'Invalid selected group entry for {group_name!r}.')
        checked_indices = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(
                    f'Mask-group index for {group_name!r} must be an integer: {index!r}.'
                )
            if index < 0 or index >= args.input_dim:
                raise ValueError(
                    f'Mask-group index out of bounds for {args.dataset}/{group_name}: '
                    f'{index} not in [0, {args.input_dim}).'
                )
            checked_indices.append(index)
        if checked_indices:
            system_groups[group_name] = checked_indices
    metadata = {
        'registry_version': REGISTRY_VERSION,
        'registry_fingerprint': fingerprint,
        'split': payload['split'],
        'split_seed': payload['split_seed'],
        'selection_rule': payload['selection_rule'],
        'selected_groups': system_groups,
    }
    return system_groups, metadata


# ── Masking strategy functions ────────────────────────────────────────────────

def temporal_block_masking(x, original_mask, min_ratio, max_ratio):
    """Randomly select a contiguous time interval per sample, mask all variables.
    Fully vectorized (Bug fix review #4: no Python-level for loop over batch).
    """
    N, L, V = x.shape
    ratios = torch.rand(N, device=x.device) * (max_ratio - min_ratio) + min_ratio
    block_lens = (ratios * L).clamp(min=1).long()                           # (N,)
    max_starts = (L - block_lens).clamp(min=0)                              # (N,)
    starts = (torch.rand(N, device=x.device) * (max_starts + 1)).long().clamp(max=max_starts)
    t_idx = torch.arange(L, device=x.device).unsqueeze(0)                  # (1, L)
    time_mask = (t_idx >= starts.unsqueeze(1)) & \
                (t_idx < (starts + block_lens).unsqueeze(1))                # (N, L)
    mask = time_mask.unsqueeze(2).expand(-1, -1, V)                         # (N, L, V)
    x = x * (~mask)
    return x, original_mask * (~mask), original_mask * mask


def system_level_masking(x, original_mask, min_ratio, max_ratio, system_groups):
    """Randomly select a physiological system and mask all its variables in a time block.
    Fully vectorized over batch dimension; loop only over groups (typically 5-10).
    """
    group_names = list(system_groups.keys())
    N, L, V = x.shape
    device = x.device
    group_idx = torch.randint(0, len(group_names), (N,), device=device)
    ratios = torch.rand(N, device=device) * (max_ratio - min_ratio) + min_ratio
    block_lens = (ratios * L).clamp(min=1).long()
    max_starts = (L - block_lens).clamp(min=0)
    starts = (torch.rand(N, device=device) * (max_starts + 1)).long().clamp(max=max_starts)
    t_idx = torch.arange(L, device=device)
    mask = torch.zeros(N, L, V, dtype=torch.bool, device=device)
    for g, group_name in enumerate(group_names):
        var_idx = torch.tensor(system_groups[group_name], dtype=torch.long, device=device)
        sel_i = (group_idx == g).nonzero(as_tuple=True)[0]
        if sel_i.numel() == 0:
            continue
        time_in_block = (t_idx[None] >= starts[sel_i, None]) & \
                        (t_idx[None] < (starts[sel_i] + block_lens[sel_i]).unsqueeze(1))
        chunk = mask[sel_i]
        chunk[:, :, var_idx] = time_in_block.unsqueeze(-1).expand(-1, -1, len(var_idx))
        mask[sel_i] = chunk
    x = x * (~mask)
    return x, original_mask * (~mask), original_mask * mask


def get_masking_fn(epoch, total_epochs, system_groups):
    """
    Smooth probability interpolation curriculum schedule.
    Gradually increases complex masking probability as training progresses.
    Avoids hard cutpoints; random masking is always retained (~28% at end).

    progress=0:   P(system)=0,    P(temporal)=0,    P(random)=1.00
    progress=0.5: P(system)=0.18, P(temporal)=0.14, P(random)=0.68
    progress=1.0: P(system)=0.48, P(temporal)=0.24, P(random)=0.28

    Bug fix (review #1): original used `rand_prob < system_prob and system_groups`
    short-circuit, causing temporal_prob=0.72 when system_groups={} (3x inflation).
    Fix: independent if check; system_groups={} falls back to random.
    """
    progress = epoch / total_epochs
    rand_prob = torch.rand(1).item()
    system_prob   = max(0.0, (progress - 0.2) * 0.6)   # 0 -> 0.48
    temporal_prob = max(0.0, (progress - 0.1) * 0.8)   # 0 -> 0.72 (includes system range)

    if rand_prob < system_prob:
        if system_groups:
            return lambda x, m, mn, mx: system_level_masking(x, m, mn, mx, system_groups)
        else:
            return random_masking
    elif rand_prob < temporal_prob:
        return temporal_block_masking
    else:
        return random_masking


# Dataset-adaptive fixed ratios (Scheme F) with epoch-level stratified sampling (Scheme D)
# Rationale: 25 epochs too short for curriculum warm-up; fixed ratios give full
# strategy exposure from epoch 1. Per-dataset ratios reflect system_groups coverage.

_DATASET_MASK_RATIOS = {
    # C12: 7 groups, 75.7% coverage, 9 ungrouped vars -> more random
    'c12':   (0.30, 0.25, 0.45),
    # C19: 7 groups, 88.2% coverage, strict stat validation -> more system
    'c19':   (0.35, 0.25, 0.40),
    # MIMIC: 4 groups, 82.4% coverage, strong temporal patterns -> more temporal
    'mimic': (0.25, 0.30, 0.45),
}


def _build_stratified_schedule(n_batches, p_sys, p_temp, p_rand, epoch_seed):
    """Build a deterministic per-batch strategy assignment for one epoch.

    Assigns exactly floor(n*p) batches to each strategy (remainder to random),
    then shuffles with a fixed seed for reproducibility across runs.

    Returns list[str] of length n_batches, each element in
    {'system', 'temporal', 'random'}.
    """
    n_sys = int(n_batches * p_sys)
    n_temp = int(n_batches * p_temp)
    n_rand = n_batches - n_sys - n_temp  # remainder goes to random

    schedule = ['system'] * n_sys + ['temporal'] * n_temp + ['random'] * n_rand
    rng = np.random.RandomState(epoch_seed)
    rng.shuffle(schedule)
    return schedule


def get_stratified_masking_fn(strategy, system_groups):
    """Return the masking function for a given strategy name."""
    if strategy == 'system' and system_groups:
        return lambda x, m, mn, mx: system_level_masking(x, m, mn, mx, system_groups)
    elif strategy == 'temporal':
        return temporal_block_masking
    else:
        return random_masking


def apply_mnar_dropout(original_mask, dropout_rate=0.05):
    """
    Lightweight dropout on observed positions. Prevents shortcut learning from
    (original_mask - input_mask) diff signal; mitigates pretrain->finetune shift.
    """
    drop = torch.rand_like(original_mask.float()) < dropout_rate
    return original_mask * (~drop)


def get_mnar_dropout_rate(epoch, total_epochs, max_rate):
    """Linear ramp: near-0 at epoch 1, max_rate at epoch total_epochs.
    Early training stays stable; regularization increases as model matures.
    """
    return max_rate * (epoch / total_epochs)


def test(args, checkpoint_path, test_dataloader):
    checkpoint = torch.load(os.path.join(args.save_dir, checkpoint_path), weights_only=False)
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    target_encoder.load_state_dict(checkpoint['target_encoder'])
    encoder.eval()
    predictor.eval()
    target_encoder.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            # Test uses clean mask: no dropout, consistent with finetune test()
            # samepretrain: always None (no MNAR encoder, same as training)
            policy_mask_clean = None
            if (args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2
                    or args.use_smile_v2_film or args.use_smile_lean or args.use_smile_lean_samepretrain
                    or args.use_smile_lean_v2 or args.odp_model != 'none'):
                policy_mask_clean = batch['mask'].clone()
            original_mask = None if (args.smile_no_mnar or args.use_smile_lean_samepretrain) else policy_mask_clean
            with torch.no_grad():
                h = target_encoder(**batch, original_mask=original_mask)
            batch['labels'] = batch['x']
            batch['x'], batch['mask'], pretrain_mask = random_masking(batch['x'], batch['mask'], args.min_mask_ratio, args.max_mask_ratio)
            z = encoder(**batch, original_mask=original_mask)
            z = predictor(z)
            test_loss += criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:])).item() * batch['x'].shape[0]
    log(logger, 'Test Loss %.4f' % (test_loss / len(test_dataset)))


def smooth_l1_loss(pred, target, pad_mask, beta=1.0):
    diff = torch.abs(pred - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    combined_mask = pad_mask.bool()
    loss = (loss * combined_mask).sum() / (combined_mask.sum() + 1e-6)
    return loss


def log(logger, msg):
    if logger is not None:
        logger.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic_decompensation', choices=['c12', 'c19', 'mimic_mortality', 
                            'mimic_phenotyping', 'mimic_decompensation', 'mimic_lengthofstay'])
    parser.add_argument('--data_dropout', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--run-tag', type=str, default=None,
                        help='Optional suffix appended to the model_name/save dir, '
                             'e.g. "pmatch" for the parameter-matched backbone control.')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader workers per process. Defaults to a conservative auto setting.')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_dir', '--save-dir', dest='save_dir', type=str, default='./export/')
    parser.add_argument('--mask-group-config', type=str, default=None,
                        help='Audited selected-mask-groups JSON required by structured/system masking.')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Fixed patient split seed associated with the mask-group audit.')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--min_mask_ratio', type=float, default=0.15)
    parser.add_argument('--max_mask_ratio', type=float, default=0.75)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--time_dim', type=int, default=16)
    parser.add_argument('--use-film', action='store_true', default=False)
    parser.add_argument('--use-smile', action='store_true', default=False)
    parser.add_argument('--use-smile-film', action='store_true', default=False,
                        help='Use SMILEFiLMEncoder (MNAR + FiLM joint modulation)')
    parser.add_argument('--use-smile-v2', action='store_true', default=False,
                        help='Use SMILEv2Encoder (MNAR attn bias + obs density + cross-attn fusion)')
    parser.add_argument('--use-smile-v2-film', action='store_true', default=False,
                        help='Use SMILEv2FiLMEncoder (SMILEv2 + time-conditional FiLM)')
    parser.add_argument('--use-smile-lean', action='store_true', default=False,
                        help='Use SMILELeanEncoder (MNAR cooccur bias + VarAtt FiLM + local obs density)')
    parser.add_argument('--use-smile-lean-samepretrain', action='store_true', default=False,
                        help='Use SMILELeanEncoder with same pretrain as smart (random masking, no MNAR)')
    parser.add_argument('--use-smile-lean-v2', action='store_true', default=False,
                        help='Use SMILELeanV2Encoder (dynamic MNAR bias + policy embeddings)')
    parser.add_argument('--odp-model', choices=['none', 'nocomiss-base', 'nocomiss-density',
                        'late', 'late-shuffled', 'nocomiss-pmatch'], default='none',
                        help='Explicit bounded ODP redesign/control architecture.')
    parser.add_argument('--model-id', type=str, default=None,
                        help='Explicit experiment directory name supplied by the runner.')
    parser.add_argument('--forecast-mode', choices=['none', 'prior', 'time', 'own', 'full',
                        'shuffled'], default='none')
    parser.add_argument('--policy-hidden-dim', type=int, default=64)
    parser.add_argument('--policy-kernel-size', type=int, default=7)
    parser.add_argument('--policy-loss-weight', type=float, default=POLICY_LOSS_WEIGHT)
    parser.add_argument('--skip-test', action='store_true',
                        help='Do not create, iterate or evaluate the held-out test loader.')
    parser.add_argument('--use-mnar', action='store_true', default=False,
                        help='Use simplified MNAREncoder (no curriculum masking)')
    parser.add_argument('--save-last', action='store_true', default=False,
                        help='Save final epoch checkpoint instead of best val loss. Pass for non-LoS datasets where curriculum masking causes monotonic loss increase.')
    # SMILE ablation control flags
    parser.add_argument('--smile-no-mnar', action='store_true',
                        help='Disable MissingPatternEncoder (curriculum mask only)')
    parser.add_argument('--smile-no-curriculum', action='store_true',
                        help='Disable curriculum masking (MNAR encoder only, random mask)')
    parser.add_argument('--smile-mask-type', choices=['all', 'temporal', 'system'], default='all',
                        help='Which masking types to include in curriculum')
    parser.add_argument('--smile-mnar-dropout', type=float, default=0.05,
                        help='MNAR dropout rate (default 0.05)')
    parser.add_argument('--obs-density-window', type=int, default=5,
                        help='Sliding window size for observation density embedding (must be odd)')
    # SMILE-v2 / SMILE-Lean ablation switches
    parser.add_argument('--abl-no-density', action='store_true', default=False,
                        help='Ablation: disable observation-density pathway')
    parser.add_argument('--abl-no-mnar-bias', action='store_true', default=False,
                        help='Ablation: disable MNAR co-occurrence attention bias')
    parser.add_argument('--abl-no-film', action='store_true', default=False,
                        help='Ablation: disable time-conditional FiLM on VarAtt')
    parser.add_argument('--abl-no-time-mnar', action='store_true', default=False,
                        help='Ablation: disable time-dynamic MNAR scaling only')
    parser.add_argument('--abl-no-time-pe', action='store_true', default=False,
                        help='Ablation: disable physical-time positional encoding')
    parser.add_argument('--abl-no-cross-attn', action='store_true', default=False,
                        help='Ablation: disable per-block MNAR cross-attention fusion')
    parser.add_argument('--abl-no-mnar-cls', action='store_true', default=False,
                        help='Ablation: disable global MNAR -> CLS injection')
    parser.add_argument('--abl-no-policy', action='store_true', default=False,
                        help='Ablation: disable policy tokens in SMILE-Lean v2 embedder')
    parser.add_argument('--abl-no-dynamic-mnar', action='store_true', default=False,
                        help='Ablation: replace dynamic MNAR co-occurrence with static global co-occurrence')
    parser.add_argument('--abl-random-bias', action='store_true', default=False,
                        help='Control: replace per-sample co-missingness matrix with a '
                             'magnitude-matched random-permuted bias (structure destroyed)')
    parser.add_argument('--abl-global-comiss', action='store_true', default=False,
                        help='Control: replace per-sample co-missingness matrix with a '
                             'running cohort-prior matrix (a momentum EMA estimate, '
                             'cross-rank aggregated under DDP; not the exact '
                             'training-set average)')
    parser.add_argument('--abl-no-dual-head', action='store_true', default=False,
                        help='Ablation: finetune with standard classifier instead of dual-head classifier')
    parser.add_argument('--pretrain-mask-mode', type=str, default='fixed',
                        choices=['fixed', 'proportional_var'],
                        help='fixed=uniform ratio across variables (default); '
                             'proportional_var=PMAE per-variable ratio scaled by obs density')
    parser.add_argument('--ratio-temperature', type=float, default=1.0,
                        help='Steepness of density->ratio mapping; 1.0=linear (recommended)')
    parser.add_argument('--smile-stratified', action='store_true', default=False,
                        help='Use dataset-adaptive fixed-ratio masking with epoch-level '
                             'stratified sampling (Scheme F+D). Replaces curriculum schedule.')
    parser.add_argument('--smile-mask-weights', type=float, nargs=3, default=None,
                        metavar=('P_SYS', 'P_TEMP', 'P_RAND'),
                        help='Custom fixed masking mix (system temporal random), must sum to 1. '
                             'Overrides dataset-adaptive defaults when --smile-stratified is set.')
    args = parser.parse_args()
    if args.dataset in ('c12', 'c19') and args.split_seed != 42:
        raise ValueError(f'{args.dataset} loaders currently expose only the fixed split seed 42.')
    # Build ablation suffix for architecture variants
    _abl_flags = {
        'no-density': args.abl_no_density,
        'no-mnar-bias': args.abl_no_mnar_bias,
        'no-film': args.abl_no_film,
        'no-time-mnar': args.abl_no_time_mnar,
        'no-time-pe': args.abl_no_time_pe,
        'no-cross-attn': args.abl_no_cross_attn,
        'no-mnar-cls': args.abl_no_mnar_cls,
        'no-policy': args.abl_no_policy,
        'no-dynamic-mnar': args.abl_no_dynamic_mnar,
        'random-bias': args.abl_random_bias,
        'global-comiss': args.abl_global_comiss,
    }
    _abl_suffix = '-'.join(k for k, v in _abl_flags.items() if v)
    if args.policy_hidden_dim != 64 or args.policy_kernel_size != 7:
        raise ValueError('ODP redesign locks policy_hidden_dim=64 and policy_kernel_size=7.')
    if args.policy_loss_weight != POLICY_LOSS_WEIGHT:
        raise ValueError('ODP redesign locks policy_loss_weight=0.1.')
    if args.odp_model != 'none':
        if args.odp_model == 'nocomiss-base':
            from models.odp import NoCoMissClinicalEncoder
            Encoder = lambda ns: NoCoMissClinicalEncoder(ns, use_density=False)
        elif args.odp_model == 'nocomiss-density':
            from models.odp import NoCoMissClinicalEncoder
            Encoder = lambda ns: NoCoMissClinicalEncoder(ns, use_density=True)
        elif args.odp_model == 'late':
            from models.odp import ODPLateFusionEncoder
            Encoder = lambda ns: ODPLateFusionEncoder(
                ns, forecast_mode=ns.forecast_mode if ns.forecast_mode != 'none' else 'full')
        elif args.odp_model == 'late-shuffled':
            from models.odp import ODPLateFusionEncoder
            Encoder = lambda ns: ODPLateFusionEncoder(ns, shuffled=True)
        else:
            from models.odp import ParameterMatchedAdapterEncoder as Encoder
        model_name = args.model_id or {
            'nocomiss-base': 'smart-smile-lean-nocomiss-base',
            'nocomiss-density': 'smart-smile-lean-nocomiss-density',
            'late': 'smart-smile-lean-odp-late',
            'late-shuffled': 'smart-smile-lean-odp-late-shuffled',
            'nocomiss-pmatch': 'smart-smile-lean-nocomiss-pmatch',
        }[args.odp_model]
    elif args.forecast_mode == 'prior':
        Encoder = None
        model_name = args.model_id or 'odp-forecast-prior'
    elif args.use_smile_lean_samepretrain:
        from models.smart import SMILELeanEncoder as Encoder
        model_name = 'smart-smile-lean-samepretrain'
    elif args.use_smile_lean_v2:
        from models.smart import SMILELeanV2Encoder as Encoder
        model_name = 'smart-smile-lean-v2'
        if _abl_suffix:
            model_name = 'smart-smile-lean-v2-' + _abl_suffix
    elif args.use_smile_lean:
        from models.smart import SMILELeanEncoder as Encoder
        model_name = 'smart-smile-lean'
        if _abl_suffix:
            model_name = 'smart-smile-lean-' + _abl_suffix
        elif args.smile_no_curriculum:
            # Clean w/o-curriculum control: full SMILE-Lean encoder (MNAR structural
            # signal preserved) with random pretrain masking instead of the curriculum.
            model_name = 'smart-smile-lean-norandom'
        elif getattr(args, 'obs_density_window', 5) != 5:
            # Density-window sweep: each non-default window is a distinct encoder
            # (different local receptive field) and MUST get its own checkpoint dir,
            # otherwise it collides with (and overwrites) plain smart-smile-lean.
            model_name = f'smart-smile-lean-dw{args.obs_density_window}'
    elif args.use_smile_v2_film:
        from models.smart import SMILEv2FiLMEncoder as Encoder
        model_name = 'smart-smile-v2-film'
        if _abl_suffix:
            model_name = 'smart-smile-v2-film-' + _abl_suffix
    elif args.use_smile_v2:
        from models.smart import SMILEv2Encoder as Encoder
        model_name = 'smart-smile-v2'
        if _abl_suffix:
            model_name = 'smart-smile-v2-' + _abl_suffix
    elif args.use_mnar:
        from models.smart import MNAREncoder as Encoder
        model_name = 'smart-mnar'
        if not args.save_last:
            args.save_last = True  # default to save-last for mnar
    elif args.use_smile_film:
        from models.smart import SMILEFiLMEncoder as Encoder
        model_name = 'smart-smile-film'
    elif args.use_smile:
        from models.smart import SMILEEncoder as Encoder
        model_name = 'smart-smile'
        if args.smile_no_mnar:
            model_name = 'smart-smile-nomnar'
        elif args.smile_no_curriculum:
            model_name = 'smart-smile-norandom'
        elif args.smile_stratified:
            model_name = 'smart-smile-stratified'
        elif args.smile_mask_type == 'temporal':
            model_name = 'smart-smile-temporal-only'
        elif args.smile_mask_type == 'system':
            model_name = 'smart-smile-system-only'
    elif args.use_film:
        from models.smart import TimeFiLMEncoder as Encoder
        model_name = 'smart-film'
    else:
        from models.smart import Encoder
        model_name = 'smart'
    # Auto-enable save_last for curriculum masking models to avoid
    # monotonic val-loss increase causing epoch-1 checkpoint selection
    
    _uses_curriculum = (
        not args.use_mnar
        and not args.use_smile_lean_samepretrain
        and not args.smile_no_curriculum
        and not args.smile_stratified
        and (args.use_smile or args.use_smile_film or args.use_smile_lean
             or args.use_smile_v2 or args.use_smile_v2_film or args.use_smile_lean_v2
             or args.odp_model != 'none')
    )
    # All learned forecasting conditions use the same validation joint
    # pretraining loss checkpoint rule; never inherit the legacy curriculum
    # save-last exception.
    apply_pretrain_checkpoint_rule(args, _uses_curriculum)
    if getattr(args, 'pretrain_mask_mode', 'fixed') == 'proportional_var':
        model_name = model_name + '-pmae'
    if getattr(args, 'run_tag', None):
        model_name = f'{model_name}-{args.run_tag}'
    if args.model_id:
        model_name = args.model_id
    args.save_dir = os.path.join(args.save_dir, args.dataset, model_name, f'seed_{args.seed}')
    distributed_init(args)
    configure_torch_runtime()
    if args.local_rank == 0 and args.save_model and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.local_rank == 0:
        logger = logging.getLogger()
        init_logging(logger, args.save_dir if args.save_model else None)
    else:
        logger = None
    log(logger, json.dumps(vars(args), indent=4))
    set_seed(args.seed)

    if args.dataset == 'c12':
        args.input_dim = 37
        args.demo_dim = 4
        args.num_class = 2
        args.max_len = 48
        train_dataset, val_dataset, test_dataset = load_challenge_2012()
    elif args.dataset == 'c19':
        args.input_dim = 34
        args.demo_dim = 5
        args.num_class = 2
        args.max_len = 60
        train_dataset, val_dataset, test_dataset = load_challenge_2019()
    elif args.dataset == 'mimic_mortality':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 2
        args.max_len = 48
        train_dataset, val_dataset, test_dataset = load_mimic_iii_mortality(split_seed=args.split_seed)
    elif args.dataset == 'mimic_phenotyping':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 25
        args.max_len = 60
        train_dataset, val_dataset, test_dataset = load_mimic_iii_phenotyping(split_seed=args.split_seed)
    elif args.dataset == 'mimic_decompensation':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 2
        args.max_len = 24
        train_dataset, val_dataset, test_dataset = load_mimic_iii_decompensation(split_seed=args.split_seed)
    elif args.dataset == 'mimic_lengthofstay':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 1
        args.max_len = 24
        args.max_mask_ratio = 0.75
        train_dataset, val_dataset, test_dataset = load_mimic_iii_lengthofstay(
            task='regression',
            label_unit='auto',
            split_seed=args.split_seed,
        )
    else:
        raise Exception("Dataset not exist!")
    if args.forecast_mode == 'shuffled' or args.odp_model == 'late-shuffled':
        train_dataset = DeterministicShuffledHistoryDataset(
            train_dataset, 'train', args.seed, train=True)
        val_dataset = DeterministicShuffledHistoryDataset(
            val_dataset, 'validation', args.seed, train=False)
        if not args.skip_test:
            test_dataset = DeterministicShuffledHistoryDataset(
                test_dataset, 'test', args.seed, train=False)
    if args.forecast_mode == 'prior':
        prior = training_variable_prior(train_dataset)
        records = prior_records(val_dataset, prior)
        result = {
            'mode': 'prior',
            'primary': forecasting_metrics(records, include_t0=False),
            'secondary': forecasting_metrics(records, include_t0=True),
            'source': 'training split t>=1 prevalence only',
        }
        if args.local_rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, 'forecasting_validation.json'), 'w', encoding='utf-8') as handle:
                json.dump(result, handle, indent=2)
        log(logger, '[ODP] analytic prior validation complete; held-out test not touched.')
        raise SystemExit(0)
    if args.data_dropout > 0:
        train_dataset.dropout_data(args.data_dropout)
        val_dataset.dropout_data(args.data_dropout)
        test_dataset.dropout_data(args.data_dropout)
    log(logger, 'Dataset Loaded.')
    system_groups, mask_group_metadata = get_mask_system_groups(args)
    if mask_group_metadata is not None:
        log(logger, '[Mask Groups] registry_version={registry_version} '
                    'registry_fingerprint={registry_fingerprint} split={split} '
                    'split_seed={split_seed} threshold={selection_rule}'.format(
                        **mask_group_metadata))
        log(logger, f'[Mask Groups] selected_groups={mask_group_metadata["selected_groups"]}')
    elif args.mask_group_config:
        log(logger, '[Mask Groups] structured masking disabled; supplied config is not consumed.')
    dataloader_kwargs = build_dataloader_kwargs(args)
    log(logger, f'DataLoader kwargs: {dataloader_kwargs}')
    if args.dataset != 'all':
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True)
            val_sampler = SequentialSampler(val_dataset)
            test_sampler = None if args.skip_test else SequentialSampler(test_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = SequentialSampler(val_dataset)
            test_sampler = None if args.skip_test else SequentialSampler(test_dataset)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            collate_fn=collate_fn, **dataloader_kwargs
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=val_sampler,
            collate_fn=collate_fn, **dataloader_kwargs
        )
        test_dataloader = None
        if not args.skip_test:
            test_dataloader = DataLoader(
                test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                collate_fn=collate_fn, **dataloader_kwargs
            )

    var_order_idx, inv_order_idx = get_variable_order(
        args.dataset.split('_')[0] if args.dataset.startswith('mimic') else args.dataset
    )
    log(logger, 'Runtime init: variable order resolved.')
    args.var_order_idx = var_order_idx.cuda()
    args.inv_order_idx = inv_order_idx.cuda()
    log(logger, 'Runtime init: variable order moved to CUDA.')

    encoder = Encoder(args).cuda()
    log(logger, 'Runtime init: encoder moved to CUDA.')
    predictor = EmbeddingDecoder(args).cuda()
    log(logger, 'Runtime init: predictor moved to CUDA.')
    target_encoder = copy.deepcopy(encoder)
    log(logger, 'Runtime init: target encoder copied.')
    
    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
        predictor = torch.nn.parallel.DistributedDataParallel(
            predictor, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
        log(logger, 'Runtime init: DDP wrap complete.')
    for p in target_encoder.parameters():
        p.requires_grad = False
        
    ema = [0.996, 1]
    ipe = len(train_dataloader)
    ipe_scale = 1.0
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*args.epochs*ipe_scale)
                          for i in range(int(ipe*args.epochs*ipe_scale)+1))
    
    param_groups = [
        {
            'params': encoder.parameters(),
        }, 
        {
            'params': predictor.parameters()
        }
    ]
    optimizer = torch.optim.Adam(param_groups, args.lr)
    criterion = smooth_l1_loss

    # PMAE v1: pre-compute per-variable observation density (one-time scan)
    PROP_MODE = getattr(args, 'pretrain_mask_mode', 'fixed')
    var_max_ratios = None
    if PROP_MODE == 'proportional_var':
        log(logger, '[PMAE] Computing variable-level observation density...')
        p_obs = compute_variable_density(train_dataloader)
        var_max_ratios = compute_var_max_ratios(
            p_obs, args.min_mask_ratio, args.max_mask_ratio,
            temperature=getattr(args, 'ratio_temperature', 1.0)).cuda()
        log(logger, f'[PMAE] p_obs  range: [{p_obs.min():.3f}, {p_obs.max():.3f}]  mean={p_obs.mean():.3f}')
        log(logger, f'[PMAE] var_max_ratios: [{var_max_ratios.min():.3f}, {var_max_ratios.max():.3f}]')

    best_auc = 0
    best_prc = 0
    best_mse = float('inf')
    ema_val = float('inf')
    # Scheme F+D: resolve dataset-adaptive ratios once before training
    if args.smile_stratified:
        if args.smile_mask_weights is not None:
            _p_sys, _p_temp, _p_rand = args.smile_mask_weights
        else:
            _ds_key = 'mimic' if args.dataset.startswith('mimic') else args.dataset
            _p_sys, _p_temp, _p_rand = _DATASET_MASK_RATIOS.get(
                _ds_key, (0.30, 0.25, 0.45))
        log(logger, f'[Stratified] P(sys)={_p_sys:.2f} P(temp)={_p_temp:.2f} '
                     f'P(rand)={_p_rand:.2f}  batches/epoch={len(train_dataloader)}')

    epoch_bar = tqdm(range(1, args.epochs + 1), desc='[Pretrain]', unit='epoch')
    for i in epoch_bar:
        train_loss = 0
        val_loss = 0
        forecast_result = None
        encoder.train()
        predictor.train()
        target_encoder.eval()  # EMA target: eval mode prevents dropout noise in targets
        if args.distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(i - 1)
        if hasattr(train_dataset, 'set_epoch'):
            train_dataset.set_epoch(i - 1)
        # Scheme D: build per-epoch stratified schedule (deterministic shuffle)
        if args.smile_stratified:
            _strat_schedule = _build_stratified_schedule(
                len(train_dataloader), _p_sys, _p_temp, _p_rand,
                epoch_seed=args.seed * 10000 + i)
        batch_bar = tqdm(train_dataloader, desc=f'  Ep{i:>3}', leave=False, unit='batch')
        for step, batch in enumerate(batch_bar, 1):
            for key in batch:
                batch[key] = batch[key].cuda(non_blocking=True)
            # Clean policy mask is never corrupted; input visibility mask may be.
            policy_mask_clean = batch['mask'].clone()
            mnar_drop = get_mnar_dropout_rate(i, args.epochs, args.smile_mnar_dropout)
            batch['mask'] = apply_mnar_dropout(batch['mask'], dropout_rate=mnar_drop)
            # When ablating MNAR encoder, pass None so SMILEEncoder skips it
            enc_original_mask = None if args.smile_no_mnar else policy_mask_clean
            # smart-smile-lean-samepretrain: same pretrain as smart (random mask, no MNAR)
            if args.use_smile_lean_samepretrain:
                enc_original_mask = None
            # smart-mnar: always random masking, no curriculum
            if args.use_mnar or args.use_smile_lean_samepretrain:
                masking_fn = random_masking
            elif args.smile_no_curriculum:
                masking_fn = random_masking
            elif args.smile_stratified:
                # Scheme F+D: deterministic strategy from pre-built schedule
                masking_fn = get_stratified_masking_fn(
                    _strat_schedule[step - 1], system_groups)
            else:
                mask_sg = {} if args.smile_mask_type == 'temporal' else system_groups
                if args.smile_mask_type == 'system':
                    masking_fn = (lambda x, m, mn, mx: system_level_masking(x, m, mn, mx, system_groups)
                                  if system_groups else random_masking)
                else:
                    masking_fn = get_masking_fn(i, args.epochs, mask_sg)
            with torch.no_grad():
                h = target_encoder(**batch, original_mask=enc_original_mask)
            batch['labels'] = batch['x']
            # PMAE v1: replace random branch with proportional masking; temporal/system unchanged
            if var_max_ratios is not None and masking_fn is random_masking:
                batch['x'], batch['mask'], pretrain_mask = proportional_random_masking(
                    batch['x'], batch['mask'],
                    args.min_mask_ratio, args.max_mask_ratio, var_max_ratios)
            else:
                batch['x'], batch['mask'], pretrain_mask = masking_fn(
                    batch['x'], batch['mask'], args.min_mask_ratio, args.max_mask_ratio)
            policy_output = None
            if args.odp_model in ('late', 'late-shuffled'):
                z, policy_output = encoder(
                    **batch, original_mask=policy_mask_clean, return_policy=True)
            else:
                z = encoder(**batch, original_mask=enc_original_mask)
            z = predictor(z)
            recon_loss = criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:]))
            policy_loss = (causal_policy_bce(
                policy_output['logits'], policy_mask_clean, batch['lens'])
                if policy_output is not None else recon_loss.new_zeros(()))
            loss = recon_loss + args.policy_loss_weight * policy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
            train_loss += loss.item() * batch['x'].shape[0]
            batch_bar.set_postfix(loss=f'{loss.item():.4f}')

        encoder.eval()
        predictor.eval()
        target_encoder.eval()
        forecast_records = []
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch:
                    batch[key] = batch[key].cuda(non_blocking=True)
                # Val uses clean mask: stable metric consistent with finetune val
                # samepretrain: always None (no MNAR encoder, same as training)
                policy_mask_clean = None
                if (args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2
                        or args.use_smile_v2_film or args.use_smile_lean or args.use_smile_lean_samepretrain
                        or args.use_smile_lean_v2 or args.odp_model != 'none'):
                    policy_mask_clean = batch['mask'].clone()
                enc_original_mask = None if (args.smile_no_mnar or args.use_smile_lean_samepretrain) else policy_mask_clean
                h = target_encoder(**batch, original_mask=enc_original_mask)
                batch['labels'] = batch['x']
                if var_max_ratios is not None:
                    batch['x'], batch['mask'], pretrain_mask = proportional_random_masking(
                        batch['x'], batch['mask'],
                        args.min_mask_ratio, args.max_mask_ratio, var_max_ratios)
                else:
                    batch['x'], batch['mask'], pretrain_mask = random_masking(
                        batch['x'], batch['mask'], args.min_mask_ratio, args.max_mask_ratio)
                policy_output = None
                if args.odp_model in ('late', 'late-shuffled'):
                    z, policy_output = encoder(
                        **batch, original_mask=policy_mask_clean, return_policy=True)
                else:
                    z = encoder(**batch, original_mask=enc_original_mask)
                z = predictor(z)
                recon_loss = criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:]))
                policy_loss = (causal_policy_bce(
                    policy_output['logits'], policy_mask_clean, batch['lens'])
                    if policy_output is not None else recon_loss.new_zeros(()))
                joint_loss = recon_loss + args.policy_loss_weight * policy_loss
                val_loss += joint_loss.item() * batch['x'].shape[0]
                if policy_output is not None:
                    forecast_records.extend(make_forecast_records(
                        policy_output['prob'], policy_mask_clean, batch['lens'], batch['sample_id']))
        t_loss = train_loss / len(train_dataset) * args.world_size
        v_loss = val_loss / len(val_dataset)
        ema_val = v_loss if i == 1 else 0.3 * v_loss + 0.7 * ema_val
        epoch_bar.set_postfix(train=f'{t_loss:.4f}', val=f'{v_loss:.4f}', ema=f'{ema_val:.4f}')
        log(logger, 'Epoch %d: Train Loss %.4f, Valid Loss %.4f, EMA Val %.4f' % (i, t_loss, v_loss, ema_val))
        if forecast_records:
            global_records = gather_forecast_records(forecast_records)
            forecast_result = {
                'mode': args.forecast_mode if args.forecast_mode != 'none' else (
                    'shuffled' if args.odp_model == 'late-shuffled' else 'full'),
                'epoch': i,
                'checkpoint_selection_loss': v_loss,
                'primary': forecasting_metrics(global_records, include_t0=False),
                'secondary': forecasting_metrics(global_records, include_t0=True),
            }
            if args.local_rank == 0:
                with open(os.path.join(args.save_dir, f'forecasting_validation_epoch_{i}.json'), 'w', encoding='utf-8') as handle:
                    json.dump(forecast_result, handle, indent=2)
        cur_mse = ema_val
        if args.save_last:
            # Save at final epoch (ensures model benefits from full training)
            if i == args.epochs and args.local_rank == 0:
                state = {
                    'encoder': encoder.state_dict(),
                    'predictor': predictor.state_dict(),
                    'target_encoder': target_encoder.state_dict(),
                    'epoch': i,
                    'forecasting_validation': forecast_result,
                }
                log(logger, '----- Save last epoch model - L1: %.4f -----' % cur_mse)
                torch.save(state, os.path.join(args.save_dir, 'checkpoint-mse.pth'))
                if forecast_result is not None:
                    with open(os.path.join(args.save_dir, 'forecasting_validation.json'), 'w', encoding='utf-8') as handle:
                        json.dump(forecast_result, handle, indent=2)
        elif cur_mse < best_mse:
            best_mse = cur_mse
            if args.local_rank == 0:
                state = {
                    'encoder': encoder.state_dict(),
                    'predictor': predictor.state_dict(),
                    'target_encoder': target_encoder.state_dict(),
                    'epoch': i,
                    'forecasting_validation': forecast_result,
                }
                log(logger, '----- Save best model - L1: %.4f -----' % cur_mse)
                torch.save(state, os.path.join(args.save_dir, 'checkpoint-mse.pth'))
                if forecast_result is not None:
                    with open(os.path.join(args.save_dir, 'forecasting_validation.json'), 'w', encoding='utf-8') as handle:
                        json.dump(forecast_result, handle, indent=2)
        if args.distributed:
            dist.barrier()

    if args.distributed:
        dist.barrier()
    if args.skip_test:
        log(logger, 'Held-out test skipped by --skip-test; no test metrics were computed.')
    else:
        test(args, 'checkpoint-mse.pth', test_dataloader)
