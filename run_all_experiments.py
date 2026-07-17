"""
Run the public SMILE-Lean experiment grid.

Usage examples:
    # Inspect commands first
    python run_all_experiments.py --dry-run

    # Run SMART, SMILE-Lean, and the three main ablations on C12/C19
    python run_all_experiments.py --datasets c12 c19

    # Run one ablation
    python run_all_experiments.py --models smart-smile-lean-no-film --datasets c12

    # Only run specific seeds
    python run_all_experiments.py --seeds 42

    # Skip pretrain if checkpoint already exists, redo finetune
    python run_all_experiments.py --finetune-only

    # Re-evaluate existing finetune checkpoints with validation-selected thresholds
    python run_all_experiments.py --eval-only

    # Density-window sensitivity sweep (window=5 is the default smart-smile-lean)
    python run_all_experiments.py \
        --models smart-smile-lean smart-smile-lean-dw3 smart-smile-lean-dw7 smart-smile-lean-dw9 \
        --datasets c12 c19
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

SMART_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MASK_GROUP_CONFIG = os.path.join(
    'experiments', 'bibm_smile', 'configs', 'selected_mask_groups.json'
)

ALL_DATASETS = [
    'c12',
    'c19',
    'mimic_mortality',
    'mimic_phenotyping',
    'mimic_decompensation',
    'mimic_lengthofstay',
]
ALL_MODELS = [
    'smart',
    'smart-smile-lean',
]
# Public SMILE-Lean architecture ablations used in the paper tables.
ABLATION_MODELS = [
    'smart-smile-lean-no-density',
    'smart-smile-lean-no-mnar-bias',
    'smart-smile-lean-no-film',
]
# Co-missingness diagnostic controls (P0-B): isolate structure from generic bias
# capacity and cohort priors. Both keep the bias pathway on SMILE-Lean.
CONTROL_MODELS = [
    'smart-smile-lean-random-bias',
    'smart-smile-lean-global-comiss',
]
# Parameter-matched backbone control: plain SMART backbone widened to strictly
# exceed the SMILE-Lean parameter count (d_model 40 vs 32), same training budget
# as the SMART baseline. Isolates capacity from the structured-missingness modules.
PMATCH_MODELS = {
    'smart-pmatch': 40,
}
# Density-window sensitivity sweep. Each variant is plain SMILE-Lean run with a
# non-default --obs-density-window (window=5 is the default smart-smile-lean).
# Window 1 degenerates to pure pointwise observation (lower bound); 3/7/9 probe
# robustness to the local-density receptive field. Values must be odd.
DENSITY_WINDOW_SWEEP = {
    'smart-smile-lean-dw1': 1,
    'smart-smile-lean-dw3': 3,
    'smart-smile-lean-dw7': 7,
    'smart-smile-lean-dw9': 9,
}
# Clean "w/o Curriculum" control. Runs the full SMILE-Lean architecture with the
# MNAR structural signal preserved (the MNAR encoder still receives the observation
# mask at every stage), replacing ONLY the pretraining structured-masking curriculum
# with plain random masking (--smile-no-curriculum). This isolates the curriculum
# factor alone. Contrast with smart-smile-lean-samepretrain, which additionally
# zeroes out the pretrain structural signal (original_mask=None) and therefore
# cannot attribute a change to the curriculum by itself.
NOCURRICULUM_MODELS = ['smart-smile-lean-norandom']
ALL_SEEDS = [1, 42, 3407]
# Lean models use batch_size=64 and finetune_epochs=25 (same as smart baseline)
# and save_best instead of save_last for pretrain checkpointing.
_LEAN_MODELS = {'smart-smile-lean'}
# Ablation models also use lean settings
_LEAN_V1_ABLATION_MODELS = set(ABLATION_MODELS) | set(CONTROL_MODELS)
_LEAN_V2_ABLATION_MODELS = set()
_LEAN_ABLATION_MODELS = set(ABLATION_MODELS) | set(CONTROL_MODELS)
_LEAN_MODELS.update(_LEAN_ABLATION_MODELS)
# Density-window sweep variants share all SMILE-Lean training settings.
_LEAN_MODELS.update(DENSITY_WINDOW_SWEEP)
# The w/o-curriculum control shares all SMILE-Lean training settings (batch 64,
# finetune 25 epochs); only the pretrain masking schedule differs.
_LEAN_MODELS.update(NOCURRICULUM_MODELS)

# Map ablation model name -> list of --abl-* CLI flags
_ABLATION_FLAGS = {
    'smart-smile-lean-no-density':                   ['--abl-no-density'],
    'smart-smile-lean-no-mnar-bias':                 ['--abl-no-mnar-bias'],
    'smart-smile-lean-no-film':                      ['--abl-no-film'],
    'smart-smile-lean-random-bias':                  ['--abl-random-bias'],
    'smart-smile-lean-global-comiss':                ['--abl-global-comiss'],
}


def output_root_path(export_root):
    if os.path.isabs(export_root):
        return export_root
    return os.path.join(SMART_DIR, export_root)


def mask_group_config_path(mask_group_config):
    if os.path.isabs(mask_group_config):
        return mask_group_config
    return os.path.join(SMART_DIR, mask_group_config)


def models_require_mask_group_config(models):
    return [model for model in models if model in _LEAN_MODELS]


def pretrain_ckpt(export_root, dataset, model_name, seed):
    return os.path.join(
        output_root_path(export_root), dataset, model_name, f'seed_{seed}', 'checkpoint-mse.pth'
    )


def finetune_ckpt(export_root, dataset, model_name, seed):
    return os.path.join(
        output_root_path(export_root), dataset, model_name, f'seed_{seed}', 'checkpoint-prc.pth'
    )


def pretrain_source_model(model_name):
    """Return the model name whose pretrain checkpoint should be reused."""
    if model_name == 'smart-smile-lean-v2-no-dual-head':
        return 'smart-smile-lean-v2'
    return model_name


def launch_prefix(args, run_idx):
    """Build the process launcher prefix for a worker script."""
    if not args.use_torchrun:
        return [args.python_executable]
    master_port = args.master_port_base + (run_idx - 1)
    return [
        args.python_executable, '-m', 'torch.distributed.run',
        '--standalone',
        '--nnodes=1',
        '--nproc_per_node', str(args.nproc_per_node),
        '--master_port', str(master_port),
    ]


def build_launch_env(args):
    env = os.environ.copy()
    devices = args.devices
    if devices:
        env['CUDA_VISIBLE_DEVICES'] = devices
    if args.use_torchrun and env.get('SMART_SAFE_NCCL', '1') == '1':
        safe_env = {
            'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
            'TORCH_NCCL_BLOCKING_WAIT': '1',
            'NCCL_P2P_DISABLE': '1',
            'NCCL_IB_DISABLE': '1',
        }
        for key, value in safe_env.items():
            env.setdefault(key, value)
    return env


def run_cmd(cmd, tag, dry_run, env=None):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'\n{"="*70}')
    print(f'[{ts}] {tag}')
    print(f'CMD: {" ".join(cmd)}')
    if env is not None and env.get('CUDA_VISIBLE_DEVICES'):
        print(f'CUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]}')
    if env is not None:
        env_keys = [
            'SMART_SAFE_NCCL',
            'TORCH_NCCL_ASYNC_ERROR_HANDLING',
            'TORCH_NCCL_BLOCKING_WAIT',
            'NCCL_P2P_DISABLE',
            'NCCL_IB_DISABLE',
        ]
        applied = [f'{key}={env[key]}' for key in env_keys if key in env]
        if applied:
            print('DIST_ENV:', ' '.join(applied))
    print('='*70, flush=True)
    if dry_run:
        print('[DRY RUN] skipped')
        return True
    result = subprocess.run(cmd, cwd=SMART_DIR, env=env)
    return result.returncode == 0


def los_finetune_flags(dataset):
    """Use the ROC-style LoS protocol consistently across all model variants."""
    if dataset != 'mimic_lengthofstay':
        return []
    return [
        '--los-task', 'classification',
        '--los-label-unit', 'auto',
        '--los-save-metric', 'auc_micro',
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running them')
    parser.add_argument('--python-executable', type=str,
                        default=os.environ.get('SMART_PYTHON_EXECUTABLE', sys.executable),
                        help='Python executable used to launch training workers. '
                             'Defaults to SMART_PYTHON_EXECUTABLE env or the current interpreter.')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS + ABLATION_MODELS + CONTROL_MODELS
                                + list(PMATCH_MODELS) + list(DENSITY_WINDOW_SWEEP)
                                + NOCURRICULUM_MODELS,
                        metavar='MODEL',
                        help='Models to run. Available: ' + ', '.join(
                            ALL_MODELS + ABLATION_MODELS + CONTROL_MODELS
                            + list(PMATCH_MODELS) + list(DENSITY_WINDOW_SWEEP)
                            + NOCURRICULUM_MODELS))
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS,
                        choices=ALL_DATASETS, metavar='DATASET')
    parser.add_argument('--seeds', nargs='+', type=int, default=ALL_SEEDS,
                        metavar='SEED')
    parser.add_argument('--pretrain-epochs', type=int, default=25)
    parser.add_argument('--finetune-epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size per GPU. Default is 64.')
    parser.add_argument('--export-root', type=str, default='./export',
                        help='Root directory for checkpoints and logs generated by both stages.')
    parser.add_argument('--mask-group-config', type=str, default=DEFAULT_MASK_GROUP_CONFIG,
                        help='Audited selected-mask-groups JSON forwarded to pretraining. '
                             f'Default: {DEFAULT_MASK_GROUP_CONFIG}')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Fixed data split seed forwarded to pretraining and finetuning.')
    parser.add_argument('--use-torchrun', action='store_true',
                        help='Launch pretrain/finetune with torchrun for multi-GPU DDP.')
    parser.add_argument('--nproc-per-node', type=int, default=2,
                        help='Processes per node when --use-torchrun is enabled.')
    parser.add_argument('--master-port-base', type=int, default=29500,
                        help='Base master port for torchrun; each run uses base + run_idx - 1.')
    parser.add_argument('--devices', type=str,
                        default=os.environ.get('DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES'),
                        help='Visible CUDA devices, e.g. "0,1". Defaults to DEVICES or CUDA_VISIBLE_DEVICES env.')
    parser.add_argument('--pretrain-only', action='store_true',
                        help='Only run pretraining, skip finetuning')
    parser.add_argument('--finetune-only', action='store_true',
                        help='Only run finetuning (pretrain checkpoint must exist)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing checkpoint-prc.pth files')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if checkpoint already exists')
    args = parser.parse_args()
    exclusive_modes = [args.pretrain_only, args.finetune_only, args.eval_only]
    if sum(exclusive_modes) > 1:
        parser.error('--pretrain-only, --finetune-only, and --eval-only are mutually exclusive')
    structured_models = models_require_mask_group_config(args.models)
    if structured_models and not (args.finetune_only or args.eval_only):
        if not args.mask_group_config:
            parser.error(
                '--mask-group-config is required for structured masking models: '
                + ', '.join(structured_models)
            )
        if not args.dry_run and not os.path.exists(mask_group_config_path(args.mask_group_config)):
            parser.error(f'--mask-group-config not found: {args.mask_group_config}')

    plan = []
    for model in args.models:
        for dataset in args.datasets:
            for seed in args.seeds:
                plan.append((model, dataset, seed))

    total = len(plan)
    failed = []
    skipped_pre = 0
    skipped_ft = 0

    print(f'Total experiments: {total}')
    print(f'Models:   {args.models}')
    print(f'Datasets: {args.datasets}')
    print(f'Seeds:    {args.seeds}')
    print(f'Pretrain epochs: {args.pretrain_epochs}  |  Finetune epochs: {args.finetune_epochs}')
    print(f'Export root: {args.export_root}  |  Split seed: {args.split_seed}')
    if args.mask_group_config:
        print(f'Mask group config: {args.mask_group_config}')
    print(f'Worker python: {args.python_executable}')
    if args.devices:
        print(f'Visible devices: {args.devices}')
    if args.use_torchrun:
        print(f'Launch mode: torchrun ({args.nproc_per_node} proc/node), master_port_base={args.master_port_base}')
    else:
        print('Launch mode: python (single process)')
    launch_env = build_launch_env(args)

    for idx, (model, dataset, seed) in enumerate(plan, 1):
        use_film_flag          = ['--use-film']          if model == 'smart-film'          else []
        use_smile_film_flag    = ['--use-smile-film']    if model == 'smart-smile-film'    else []
        _is_v2_film_ablation = model.startswith('smart-smile-v2-film-') and model in _ABLATION_FLAGS
        _is_v2_ablation = model.startswith('smart-smile-v2-') and not _is_v2_film_ablation and model in _ABLATION_FLAGS
        use_smile_v2_film_flag = ['--use-smile-v2-film'] if model == 'smart-smile-v2-film' or _is_v2_film_ablation else []
        use_smile_v2_flag      = ['--use-smile-v2']      if model == 'smart-smile-v2' or _is_v2_ablation else []
        _is_lean_v2_ablation = model in _LEAN_V2_ABLATION_MODELS
        use_smile_lean_v2_flag = ['--use-smile-lean-v2'] if model == 'smart-smile-lean-v2' or _is_lean_v2_ablation else []
        _is_lean_v1_ablation = model in _LEAN_V1_ABLATION_MODELS
        _is_density_window_sweep = model in DENSITY_WINDOW_SWEEP
        _is_nocurriculum = model in NOCURRICULUM_MODELS
        use_smile_lean_flag              = ['--use-smile-lean']             if model in ('smart-smile-lean', 'smart-smile-lean-pmae') or _is_lean_v1_ablation or _is_density_window_sweep or _is_nocurriculum else []
        # Clean w/o-curriculum control: full SMILE-Lean run with random pretrain masking.
        smile_no_curriculum_flag = ['--smile-no-curriculum'] if _is_nocurriculum else []
        # Non-default density-window forwarded to pretrain, finetune, and eval so
        # the encoder is rebuilt with a matching receptive field at every stage.
        density_window_flag = (['--obs-density-window', str(DENSITY_WINDOW_SWEEP[model])]
                               if _is_density_window_sweep else [])
        # Parameter-matched backbone control: widen plain SMART and tag its runs so
        # checkpoints do not collide with the d_model=32 SMART baseline.
        _is_pmatch = model in PMATCH_MODELS
        pmatch_flag = (['--d_model', str(PMATCH_MODELS[model]), '--run-tag', 'pmatch']
                       if _is_pmatch else [])
        use_smile_lean_samepretrain_flag = ['--use-smile-lean-samepretrain'] if model == 'smart-smile-lean-samepretrain' else []
        pmae_pretrain_flag               = ['--pretrain-mask-mode', 'proportional_var'] if model == 'smart-smile-lean-pmae' else []
        pretrain_model = pretrain_source_model(model)
        pretrain_dir_flag = [
            '--pretrain-dir',
            os.path.join(args.export_root, dataset, pretrain_model, f'seed_{seed}'),
        ]
        _lean_exclude = {'smart-smile-film', 'smart-smile-v2', 'smart-smile-v2-film',
                         'smart-smile-lean-v2',
                         'smart-smile-lean', 'smart-smile-lean-samepretrain', 'smart-smile-lean-pmae'}
        _lean_exclude.update(_ABLATION_FLAGS.keys())
        _lean_exclude.update(DENSITY_WINDOW_SWEEP)
        _lean_exclude.update(NOCURRICULUM_MODELS)
        use_smile_flag         = ['--use-smile']         if (model.startswith('smart-smile')
                                                             and model not in _lean_exclude) else []
        use_mnar_flag          = ['--use-mnar']          if model == 'smart-mnar'          else []
        # Ablation extra flags for smile variants
        smile_extra = []
        if model == 'smart-smile-nomnar':
            smile_extra = ['--smile-no-mnar']
        elif model == 'smart-smile-norandom':
            smile_extra = ['--smile-no-curriculum']
        elif model == 'smart-smile-temporal-only':
            smile_extra = ['--smile-mask-type', 'temporal']
        elif model == 'smart-smile-system-only':
            smile_extra = ['--smile-mask-type', 'system']
        elif model == 'smart-smile-stratified':
            smile_extra = ['--smile-stratified']
        # Architecture ablation flags
        arch_abl_extra = _ABLATION_FLAGS.get(model, [])
        tag_prefix = f'[{idx:>2}/{total}] {model:12s} | {dataset:25s} | seed={seed}'
        los_ft_flags = los_finetune_flags(dataset)

        # ---- Pretrain ----
        # Lean models: batch_size=64, save_best (same setup as smart baseline)
        cur_batch_size = 64 if model in _LEAN_MODELS else args.batch_size
        cur_ft_epochs = 25 if model in _LEAN_MODELS else args.finetune_epochs
        if args.eval_only:
            ft_ckpt = finetune_ckpt(args.export_root, dataset, model, seed)
            if not args.dry_run and not os.path.exists(ft_ckpt):
                print(f'[WARN] finetune checkpoint missing for {model}/{dataset}/seed_{seed}, skipping evaluation')
                failed.append(f'{tag_prefix} evaluation (no finetune ckpt)')
                continue
            cmd = launch_prefix(args, idx) + [
                'main_finetune.py',
                '--dataset', dataset,
                '--seed', str(seed),
                '--batch_size', str(cur_batch_size),
                '--save_dir', args.export_root,
                '--split-seed', str(args.split_seed),
                '--eval-only',
            ] + los_ft_flags + use_film_flag + use_smile_film_flag + use_smile_v2_film_flag + use_smile_v2_flag + use_smile_lean_v2_flag + use_smile_lean_flag + use_smile_lean_samepretrain_flag + use_smile_flag + use_mnar_flag + smile_extra + arch_abl_extra + density_window_flag + pmatch_flag + smile_no_curriculum_flag
            ok = run_cmd(cmd, f'{tag_prefix} | EVALUATE', args.dry_run, env=launch_env)
            if not ok:
                failed.append(f'{tag_prefix} evaluation')
            continue

        if not args.finetune_only:
            pre_ckpt = pretrain_ckpt(args.export_root, dataset, pretrain_model, seed)
            if model == 'smart-smile-lean-v2-no-dual-head':
                print(f'{tag_prefix} | pretrain: REUSE ({pretrain_model})')
                skipped_pre += 1
            elif not args.force and os.path.exists(pre_ckpt):
                print(f'{tag_prefix} | pretrain: SKIP (exists)')
                skipped_pre += 1
            else:
                # LoS/Decomp: save best (curriculum loss stays low); other non-lean: save last
                # Lean models: save best (random masking; val loss is monotone-friendly)
                save_last_flag = ([]
                    if dataset in ('mimic_lengthofstay', 'mimic_decompensation') or model in _LEAN_MODELS
                    else ['--save-last'])
                cmd = launch_prefix(args, idx) + [
                    'main_pretrain.py',
                    '--dataset', dataset,
                    '--seed', str(seed),
                    '--epochs', str(args.pretrain_epochs),
                    '--batch_size', str(cur_batch_size),
                    '--save_dir', args.export_root,
                    '--split-seed', str(args.split_seed),
                ] + save_last_flag + use_film_flag + use_smile_film_flag + use_smile_v2_film_flag + use_smile_v2_flag + use_smile_lean_v2_flag + use_smile_lean_flag + use_smile_lean_samepretrain_flag + use_smile_flag + use_mnar_flag + smile_extra + pmae_pretrain_flag + arch_abl_extra + density_window_flag + pmatch_flag + smile_no_curriculum_flag
                if args.mask_group_config:
                    cmd.extend(['--mask-group-config', args.mask_group_config])
                ok = run_cmd(cmd, f'{tag_prefix} | PRETRAIN', args.dry_run, env=launch_env)
                if not ok:
                    failed.append(f'{tag_prefix} pretrain')
                    print(f'[WARN] pretrain failed, skipping finetune for this experiment')
                    continue

        # ---- Finetune ----
        if not args.pretrain_only:
            ft_ckpt = finetune_ckpt(args.export_root, dataset, model, seed)
            if not args.force and os.path.exists(ft_ckpt):
                print(f'{tag_prefix} | finetune: SKIP (exists)')
                skipped_ft += 1
                continue
            pre_ckpt = pretrain_ckpt(args.export_root, dataset, pretrain_model, seed)
            if not args.dry_run and not os.path.exists(pre_ckpt):
                print(f'[WARN] pretrain checkpoint missing for {pretrain_model}/{dataset}/seed_{seed}, skipping finetune')
                failed.append(f'{tag_prefix} finetune (no pretrain ckpt)')
                continue
            cmd = launch_prefix(args, idx) + [
                'main_finetune.py',
                '--dataset', dataset,
                '--seed', str(seed),
                '--epochs', str(cur_ft_epochs),
                '--batch_size', str(cur_batch_size),
                '--save_dir', args.export_root,
                '--split-seed', str(args.split_seed),
            ] + los_ft_flags + use_film_flag + use_smile_film_flag + use_smile_v2_film_flag + use_smile_v2_flag + use_smile_lean_v2_flag + use_smile_lean_flag + use_smile_lean_samepretrain_flag + use_smile_flag + use_mnar_flag + smile_extra + pretrain_dir_flag + arch_abl_extra + density_window_flag + pmatch_flag + smile_no_curriculum_flag
            ok = run_cmd(cmd, f'{tag_prefix} | FINETUNE', args.dry_run, env=launch_env)
            if not ok:
                failed.append(f'{tag_prefix} finetune')

    print(f'\n{"="*70}')
    print(f'Finished. Total={total}, Skipped pretrain={skipped_pre}, Skipped finetune={skipped_ft}')
    if failed:
        print(f'Failed ({len(failed)}):')
        for f in failed:
            print(f'  - {f}')
        raise SystemExit(1)
    else:
        print('All experiments completed successfully.')


if __name__ == '__main__':
    main()
