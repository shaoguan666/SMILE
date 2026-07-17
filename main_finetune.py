import argparse
from collections import OrderedDict
import json
import os
import logging
import torch
import torch.distributed as dist
import numpy as np
from sklearn import metrics as sk_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import load_mimic_iii_mortality, load_mimic_iii_phenotyping, load_mimic_iii_decompensation, load_mimic_iii_lengthofstay
from data.dataloader import collate_fn
from models.smart import Classifier
from utils.metrics import print_metrics_binary, print_metrics_multilabel, print_metrics_regression
from utils.utils import (
    set_seed,
    distributed_init,
    init_logging,
    configure_torch_runtime,
    build_dataloader_kwargs,
)
from utils.variable_order import get_variable_order


def compute_class_weights(dataset, num_classes=2):
    """Compute inverse-frequency class weights from dataset labels."""
    from collections import Counter
    counts = Counter()
    for sample in dataset.data:
        label = sample['labels']
        counts[int(label)] += 1
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = total / (num_classes * max(counts[c], 1))
    return weights


def apply_mnar_dropout(original_mask, dropout_rate=0.05):
    """Apply fixed random dropout to original_mask during training.
    Reduces pretrain->finetune distribution shift by mimicking pretrain's mnar dropout.
    """
    drop = torch.rand_like(original_mask.float()) < dropout_rate
    return original_mask * (~drop)


def _collect_predictions(args, dataloader):
    preds_all = []
    labels_all = []
    loss_total = 0
    with torch.no_grad():
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            if (args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2
                    or args.use_smile_v2_film or args.use_smile_lean or args.use_smile_lean_samepretrain
                    or args.use_smile_lean_v2):
                policy_mask_clean = batch['mask'].clone()
            else:
                policy_mask_clean = None
            h = encoder(**batch, original_mask=policy_mask_clean)
            preds = classifier(h, original_mask=policy_mask_clean, **batch)
            loss_total += criterion(preds, batch['labels']).item() * batch['x'].shape[0]
            preds_all.append(preds.cpu())
            labels_all.append(batch['labels'].cpu())
    return torch.cat(labels_all), torch.cat(preds_all), loss_total


def _best_f1_threshold(y_true, probs):
    precisions, recalls, thresholds = sk_metrics.precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5, 0.0
    denom = precisions[:-1] + recalls[:-1]
    f1s = np.where(denom > 0, 2 * precisions[:-1] * recalls[:-1] / denom, 0.0)
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(f1s[idx])


def _best_minpse_threshold(y_true, probs):
    precisions, recalls, thresholds = sk_metrics.precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5, 0.0
    minpse = np.minimum(precisions[:-1], recalls[:-1])
    idx = int(np.argmax(minpse))
    return float(thresholds[idx]), float(minpse[idx])


def _binary_metrics_at_threshold(y_true, preds, threshold):
    probs = np.asarray(preds)[:, 1]
    y = np.asarray(y_true)
    y_hat = (probs >= threshold).astype(int)
    precision = sk_metrics.precision_score(y, y_hat, zero_division=0)
    recall = sk_metrics.recall_score(y, y_hat, zero_division=0)
    f1 = sk_metrics.f1_score(y, y_hat, zero_division=0)
    auroc = sk_metrics.roc_auc_score(y, probs)
    precision_curve, recall_curve, _ = sk_metrics.precision_recall_curve(y, probs)
    auprc = sk_metrics.auc(recall_curve, precision_curve)
    minpse = min(precision, recall)
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "minpse": float(minpse),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def _load_model_state_dict(model, state_dict, allow_unused_mnar_bias_mismatch=False):
    """Load weights across DDP prefixes while permitting only disabled legacy CoMiss scales."""
    target_state = model.state_dict()
    source_has_module = any(key.startswith("module.") for key in state_dict)
    target_has_module = any(key.startswith("module.") for key in target_state)
    aligned_state = OrderedDict()
    for key, value in state_dict.items():
        if source_has_module and not target_has_module and key.startswith("module."):
            aligned_key = key[len("module."):]
        elif target_has_module and not source_has_module:
            aligned_key = "module." + key
        else:
            aligned_key = key
        aligned_state[aligned_key] = value

    ignored = []
    for key in list(aligned_state):
        if key not in target_state or aligned_state[key].shape == target_state[key].shape:
            continue
        if allow_unused_mnar_bias_mismatch and key.endswith("mnar_bias_scale"):
            ignored.append(key)
            del aligned_state[key]

    incompatible = model.load_state_dict(aligned_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if key not in ignored]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the requested encoder. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )
    return ignored


def test(args, checkpoint_path, test_dataloader, val_dataloader=None):
    checkpoint = torch.load(os.path.join(args.save_dir, checkpoint_path), weights_only=False)
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    ignored = _load_model_state_dict(
        encoder,
        checkpoint['encoder'],
        allow_unused_mnar_bias_mismatch=args.abl_no_mnar_bias,
    )
    if ignored:
        log(logger, "Ignored disabled legacy CoMiss parameters: {}".format(", ".join(ignored)))
    _load_model_state_dict(classifier, checkpoint['classifier'])
    encoder.eval()
    classifier.eval()
    labels_all, preds_all, test_loss = _collect_predictions(args, test_dataloader)
    test_metrics = print_metrics(labels_all, preds_all, args.local_rank == 0)
    threshold_metrics = None
    if args.num_class == 2 and val_dataloader is not None:
        val_labels, val_preds, _ = _collect_predictions(args, val_dataloader)
        val_probs = np.asarray(val_preds)[:, 1]
        val_labels_np = np.asarray(val_labels)
        f1_threshold, val_f1 = _best_f1_threshold(val_labels_np, val_probs)
        minpse_threshold, val_minpse = _best_minpse_threshold(val_labels_np, val_probs)
        f1_metrics = _binary_metrics_at_threshold(labels_all, preds_all, f1_threshold)
        minpse_metrics = _binary_metrics_at_threshold(labels_all, preds_all, minpse_threshold)
        threshold_metrics = {
            "protocol": "validation_selected_per_metric_v2",
            "f1": f1_metrics["f1"],
            "f1_threshold": f1_threshold,
            "f1_precision": f1_metrics["precision"],
            "f1_recall": f1_metrics["recall"],
            "minpse": minpse_metrics["minpse"],
            "minpse_threshold": minpse_threshold,
            "minpse_precision": minpse_metrics["precision"],
            "minpse_recall": minpse_metrics["recall"],
        }
        log(logger, "Validation-selected F1 threshold = {:.4f}".format(f1_threshold))
        log(logger, "Val best F1 = {:.4f}".format(val_f1))
        log(logger, "F1 at validation-selected threshold = {:.4f}".format(f1_metrics["f1"]))
        log(logger, "Validation-selected minPSE threshold = {:.4f}".format(minpse_threshold))
        log(logger, "Val best minPSE = {:.4f}".format(val_minpse))
        log(logger, "minPSE at validation-selected threshold = {:.4f}".format(
            minpse_metrics["minpse"]))
    if args.local_rank == 0:
        result_path = os.path.join(args.save_dir, "eval_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": checkpoint_path,
                "checkpoint_epoch": int(save_epoch),
                "test_metrics": {k: float(v) for k, v in test_metrics.items() if not hasattr(v, "__len__")},
                "validation_threshold_metrics": threshold_metrics,
            }, f, indent=2)
    log(logger, 'Test Loss %.4f' % (test_loss / len(test_dataset)))


def log(logger, msg):
    if logger is not None:
        logger.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic_decompensation', choices=['c12', 'c19', 'mimic_mortality', 
                            'mimic_phenotyping', 'mimic_decompensation', 'mimic_lengthofstay'])
    parser.add_argument('--data_dropout', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--freeze_epochs', type=int, default=5)
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
    parser.add_argument('--pretrain-dir', type=str, default=None,
                        help='Directory containing pretrained checkpoint-mse.pth. Defaults to save_dir.')
    parser.add_argument('--eval-only', action='store_true',
                        help='Evaluate save_dir/checkpoint-prc.pth without pretraining or finetuning.')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Fixed patient split seed used consistently with pretraining.')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--time_dim', type=int, default=16)
    parser.add_argument('--use-film', action='store_true', default=False)
    parser.add_argument('--use-smile', action='store_true', default=False)
    parser.add_argument('--use-smile-film', action='store_true', default=False,
                        help='Use SMILEFiLMEncoder (MNAR + FiLM joint modulation)')
    parser.add_argument('--use-mnar', action='store_true', default=False,
                        help='Use simplified MNAREncoder (no curriculum masking)')
    parser.add_argument('--use-smile-v2', action='store_true', default=False,
                        help='Use SMILEv2Encoder (MNAR attn bias + obs density + cross-attn fusion)')
    parser.add_argument('--use-smile-v2-film', action='store_true', default=False,
                        help='Use SMILEv2FiLMEncoder (SMILEv2 + time-conditional FiLM)')
    parser.add_argument('--use-smile-lean', action='store_true', default=False,
                        help='Use SMILELeanEncoder (MNAR cooccur bias + VarAtt FiLM + local obs density)')
    parser.add_argument('--use-smile-lean-samepretrain', action='store_true', default=False,
                        help='Use SMILELeanEncoder pretrained with same strategy as smart (random masking)')
    parser.add_argument('--use-smile-lean-v2', action='store_true', default=False,
                        help='Use SMILELeanV2Encoder (dynamic MNAR bias + policy embeddings + dual head)')
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
                        help='Ablation: use standard classifier instead of dual-head classifier')
    parser.add_argument('--smile-no-mnar', action='store_true', default=False)
    parser.add_argument('--smile-no-curriculum', action='store_true', default=False)
    parser.add_argument('--smile-mask-type', choices=['all', 'temporal', 'system'], default='all')
    parser.add_argument('--smile-mnar-dropout', type=float, default=0.05,
                        help='MNAR dropout rate (base value, default 0.05)')
    parser.add_argument('--smile-mnar-dropout-initial', type=float, default=0.0,
                        help='Initial MNAR dropout for progressive decay schedule. '
                             'When > 0, linearly decays from this value to 0 over all epochs. '
                             'When 0 (default), uses constant smile-mnar-dropout instead.')
    parser.add_argument('--smile-stratified', action='store_true', default=False,
                        help='Use pretrained model from stratified masking (Scheme F+D).')
    parser.add_argument('--los-task', choices=['classification', 'regression'], default='classification',
                        help='Length-of-stay protocol: classification matches ROC-style reporting; '
                             'regression matches SMART-original open-source code.')
    parser.add_argument('--los-label-unit', choices=['auto', 'hours', 'days'], default='auto',
                        help='Interpretation of raw LoS labels in lengthofstay_normalized.pkl.')
    parser.add_argument('--los-use-class-weights', action='store_true', default=False,
                        help='Use inverse-frequency class weights for LoS classification.')
    parser.add_argument('--los-save-metric', choices=['auc_micro', 'auc_macro'], default='auc_micro',
                        help='Model selection metric for LoS classification.')
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
        'no-dual-head': args.abl_no_dual_head,
    }
    _abl_suffix = '-'.join(k for k, v in _abl_flags.items() if v)
    if args.use_smile_lean_samepretrain:
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
            # Clean w/o-curriculum control: matches main_pretrain naming so the
            # finetune stage loads from and saves to the norandom checkpoint dir.
            model_name = 'smart-smile-lean-norandom'
        elif getattr(args, 'obs_density_window', 5) != 5:
            # Density-window sweep: mirror main_pretrain naming so finetune loads
            # from and saves to the matching smart-smile-lean-dw{N} checkpoint dir.
            model_name = f'smart-smile-lean-dw{args.obs_density_window}'
    elif args.use_mnar:
        from models.smart import MNAREncoder as Encoder
        model_name = 'smart-mnar'
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
    if getattr(args, 'run_tag', None):
        model_name = f'{model_name}-{args.run_tag}'
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
        args.num_class = 10 if args.los_task == 'classification' else 1
        args.max_len = 24
        train_dataset, val_dataset, test_dataset = load_mimic_iii_lengthofstay(
            task=args.los_task,
            label_unit=args.los_label_unit,
            split_seed=args.split_seed,
        )
    else:
        raise Exception("Dataset not exist!")
    if args.data_dropout > 0:
        train_dataset.dropout_data(args.data_dropout)
        val_dataset.dropout_data(args.data_dropout)
        test_dataset.dropout_data(args.data_dropout)
    log(logger, 'Dataset Loaded.')
    dataloader_kwargs = build_dataloader_kwargs(args)
    log(logger, f'DataLoader kwargs: {dataloader_kwargs}')
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collate_fn, **dataloader_kwargs
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        collate_fn=collate_fn, **dataloader_kwargs
    )
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
    if args.use_smile_lean_v2 and not args.abl_no_dual_head:
        from models.smart import DualHeadClassifier
        classifier = DualHeadClassifier(args).cuda()
    else:
        classifier = Classifier(args).cuda()
    log(logger, 'Runtime init: classifier moved to CUDA.')
    
    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
        log(logger, 'Runtime init: DDP wrap complete.')
    
    param_groups = [
        {
            'params': encoder.parameters(),
        }, 
        {
            'params': classifier.parameters()
        }
    ]
    optimizer = torch.optim.Adam(param_groups, args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    if args.dataset == 'mimic_phenotyping':
        criterion = torch.nn.BCEWithLogitsLoss()
        print_metrics = print_metrics_multilabel
        save_metric = 'auc_macro'
    elif args.dataset == 'mimic_lengthofstay':
        if args.los_task == 'classification':
            if args.los_use_class_weights:
                class_weights = compute_class_weights(train_dataset, num_classes=10).cuda()
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                log(logger, f'LoS class weights: {class_weights.tolist()}')
            else:
                criterion = torch.nn.CrossEntropyLoss()
            print_metrics = print_metrics_multilabel
            save_metric = args.los_save_metric
        else:
            criterion = torch.nn.MSELoss()
            print_metrics = print_metrics_regression
            save_metric = 'mse'
    elif args.dataset in ('mimic_decompensation', 'mimic_mortality'):
        # 与原论文一致，不加权重，极端权重会导致 AUPRC 崩溃
        criterion = torch.nn.CrossEntropyLoss()
        print_metrics = print_metrics_binary
        save_metric = 'auprc'
    else:
        # c12/c19：类别不平衡明显，加权有益
        class_weights = compute_class_weights(train_dataset, num_classes=2).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        log(logger, f'Class weights: {class_weights.tolist()}')
        print_metrics = print_metrics_binary
        save_metric = 'auprc'
    
    pretrain_dir = args.pretrain_dir if args.pretrain_dir else args.save_dir
    if args.eval_only:
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint-prc.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Finetune checkpoint not found: {checkpoint_path}')
        test(args, 'checkpoint-prc.pth', test_dataloader, val_dataloader)
        if args.distributed:
            dist.barrier()
        raise SystemExit(0)

    checkpoint = torch.load(os.path.join(pretrain_dir, 'checkpoint-mse.pth'), weights_only=False)
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    ignored = _load_model_state_dict(
        encoder,
        checkpoint['encoder'],
        allow_unused_mnar_bias_mismatch=args.abl_no_mnar_bias,
    )
    if ignored:
        log(logger, "Ignored disabled legacy CoMiss parameters: {}".format(", ".join(ignored)))

    best_prc = 0
    best_mse = float('inf')
    # Progressive MNAR dropout schedule: linear decay from initial to 0 over epochs
    _mnar_initial = args.smile_mnar_dropout_initial if args.smile_mnar_dropout_initial > 0 \
        else args.smile_mnar_dropout
    _mnar_progressive = args.smile_mnar_dropout_initial > 0
    epoch_bar = tqdm(range(1, args.epochs + 1), desc='[Finetune]', unit='epoch')
    for i in epoch_bar:
        train_loss = 0
        val_loss = 0
        encoder.train()
        classifier.train()
        if args.distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(i - 1)
        # Current MNAR dropout: linear decay if progressive schedule, else constant
        if _mnar_progressive:
            current_mnar_drop = _mnar_initial * max(0.0, 1.0 - (i - 1) / args.epochs)
        else:
            current_mnar_drop = args.smile_mnar_dropout
        batch_bar = tqdm(train_dataloader, desc=f'  Ep{i:>3}', leave=False, unit='batch')
        for step, batch in enumerate(batch_bar, 1):
            for key in batch:
                batch[key] = batch[key].cuda(non_blocking=True)
            policy_mask_clean = None
            if (args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2
                    or args.use_smile_v2_film or args.use_smile_lean or args.use_smile_lean_samepretrain
                    or args.use_smile_lean_v2):
                policy_mask_clean = batch['mask'].clone()
                batch['mask'] = apply_mnar_dropout(batch['mask'], current_mnar_drop)
            if i <= args.freeze_epochs:
                with torch.no_grad():
                    h = encoder(**batch, original_mask=policy_mask_clean)
            else:
                h = encoder(**batch, original_mask=policy_mask_clean)
            preds = classifier(h, original_mask=policy_mask_clean, **batch)
            loss = criterion(preds, batch['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch['x'].shape[0]
            batch_bar.set_postfix(loss=f'{loss.item():.4f}')

        encoder.eval()
        classifier.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch:
                    batch[key] = batch[key].cuda(non_blocking=True)
                policy_mask_clean = None
                if (args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2
                        or args.use_smile_v2_film or args.use_smile_lean or args.use_smile_lean_samepretrain
                        or args.use_smile_lean_v2):
                    policy_mask_clean = batch['mask'].clone()  # no dropout: val uses clean mask
                h = encoder(**batch, original_mask=policy_mask_clean)
                preds = classifier(h, original_mask=policy_mask_clean, **batch)
                val_loss += criterion(preds, batch['labels']).item() * batch['x'].shape[0]
                preds_all.append(preds.cpu())
                labels_all.append(batch['labels'].cpu())
        metrics = print_metrics(torch.cat(labels_all), torch.cat(preds_all), args.local_rank == 0)
        t_loss = train_loss / len(train_dataset) * args.world_size
        v_loss = val_loss / len(val_dataset)
        scheduler.step()
        epoch_bar.set_postfix(train=f'{t_loss:.4f}', val=f'{v_loss:.4f}')
        log(logger, 'Epoch %d: Train Loss %.4f, Valid Loss %.4f' % (i, t_loss, v_loss))
        current_metric = metrics[save_metric]
        should_save = False
        if save_metric == 'mse':
            if current_metric < best_mse:
                best_mse = current_metric
                should_save = True
        else:
            if current_metric > best_prc:
                best_prc = current_metric
                should_save = True
        if should_save and args.local_rank == 0:
            state = {
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': i
            }
            log(logger, f'----- Save best model - {save_metric}: %.4f -----' % current_metric)
            torch.save(state, os.path.join(args.save_dir, 'checkpoint-prc.pth'))
        if args.distributed:
            dist.barrier()

    if args.distributed:
        dist.barrier()
    test(args, 'checkpoint-prc.pth', test_dataloader, val_dataloader)
