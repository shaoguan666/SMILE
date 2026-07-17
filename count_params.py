"""
Parameter-count report for the capacity-control comparison.

Reports trainable parameter counts for three variants on every dataset:
  * Backbone           : plain SMART encoder            (Encoder,          d_model=32)
  * SMILE              : SMILE-Lean encoder             (SMILELeanEncoder, d_model=32)
  * Capacity control   : widened SMART backbone         (Encoder,          d_model=40, "smart-pmatch")

For each variant we count the pretrained encoder alone and the encoder plus the
finetune classification head (Classifier), so the appendix table can report either
the representation size or the full finetuned model.

The counts depend only on the fixed per-dataset dimensions and the architecture
defaults in main_pretrain.py / main_finetune.py; no data or checkpoints are loaded.

Usage:
    python count_params.py
    python count_params.py --datasets c12 c19 mimic_mortality mimic_decompensation
"""

import argparse
from types import SimpleNamespace

import torch

from models.smart import Encoder, SMILELeanEncoder, Classifier
from utils.variable_order import get_variable_order

# Fixed per-dataset dimensions, mirrored from main_pretrain.py (lines ~578-612).
DATASET_DIMS = {
    'c12':                  dict(input_dim=37, demo_dim=4, num_class=2,  max_len=48),
    'c19':                  dict(input_dim=34, demo_dim=5, num_class=2,  max_len=60),
    'mimic_mortality':      dict(input_dim=17, demo_dim=0, num_class=2,  max_len=48),
    'mimic_phenotyping':    dict(input_dim=17, demo_dim=0, num_class=25, max_len=60),
    'mimic_decompensation': dict(input_dim=17, demo_dim=0, num_class=2,  max_len=24),
    'mimic_lengthofstay':   dict(input_dim=17, demo_dim=0, num_class=1,  max_len=24),
}

# Architecture defaults shared by SMART / SMILE-Lean (main_pretrain.py argparse).
ARCH_DEFAULTS = dict(
    dropout=0.1,
    e_layers=2,
    n_heads=4,
    time_dim=16,
    obs_density_window=5,
)

# Backbone width used by the parameter-matched capacity control (smart-pmatch).
PMATCH_D_MODEL = 40
BACKBONE_D_MODEL = 32


def build_args(dataset, d_model):
    dims = DATASET_DIMS[dataset]
    args = SimpleNamespace(d_model=d_model, **dims, **ARCH_DEFAULTS)
    # Variable-order buffers are consumed by the MNAR encoder; CPU tensors suffice
    # for construction and do not contribute trainable parameters.
    var_order_idx, inv_order_idx = get_variable_order(
        dataset.split('_')[0] if dataset.startswith('mimic') else dataset
    )
    args.var_order_idx = var_order_idx
    args.inv_order_idx = inv_order_idx
    return args


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def report(dataset):
    variants = [
        ('Backbone (smart)',       Encoder,          BACKBONE_D_MODEL),
        ('SMILE (smile-lean)',     SMILELeanEncoder, BACKBONE_D_MODEL),
        ('Capacity (smart-pmatch)', Encoder,         PMATCH_D_MODEL),
    ]
    rows = []
    for label, encoder_cls, d_model in variants:
        args = build_args(dataset, d_model)
        encoder = encoder_cls(args)
        classifier = Classifier(args)
        enc_params = count_trainable(encoder)
        head_params = count_trainable(classifier)
        rows.append((label, d_model, enc_params, head_params, enc_params + head_params))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_DIMS),
                        choices=list(DATASET_DIMS), metavar='DATASET')
    cli = parser.parse_args()

    torch.manual_seed(0)
    header = f'{"Variant":24s} {"d_model":>7s} {"Encoder":>12s} {"Head":>10s} {"Total":>12s}'
    for dataset in cli.datasets:
        print('=' * 70)
        print(f'Dataset: {dataset}  (input_dim={DATASET_DIMS[dataset]["input_dim"]}, '
              f'num_class={DATASET_DIMS[dataset]["num_class"]})')
        print('-' * 70)
        print(header)
        for label, d_model, enc, head, total in report(dataset):
            print(f'{label:24s} {d_model:7d} {enc:12,d} {head:10,d} {total:12,d}')
    print('=' * 70)


if __name__ == '__main__':
    main()
