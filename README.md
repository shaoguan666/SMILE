# SMILE

Official research code release for the AAAI paper version of SMILE, centered on
the `smart-smile-lean` model and its main ablations for irregular clinical time
series.

This repository is derived from SMART and keeps the public release surface small:
the SMART backbone, SMILE-Lean, data preprocessing/loading utilities, audited
missingness analysis, and reproducible experiment runners.

## Setup

```bash
pip install -r requirements.txt
```

The code expects Python 3.10+ and PyTorch. Install the CUDA-enabled PyTorch build
that matches your local driver if you plan to train on GPU.

## Data

Raw and processed datasets are not included. Obtain access from the original
sources and place processed files under `data/`:

- PhysioNet Challenge 2012: https://physionet.org/content/challenge-2012/1.0.0/
- PhysioNet Challenge 2019: https://physionet.org/content/challenge-2019/1.0.0/
- MIMIC-III: https://physionet.org/content/mimiciii/1.4/

Supported dataset names:

```text
c12
c19
mimic_mortality
mimic_phenotyping
mimic_decompensation
mimic_lengthofstay
```

Expected processed files are ignored by Git:

```text
data/Challenge2012/data_normalized.pkl
data/Challenge2019/data_normalized.pkl
data/MIMIC-III/mortality_normalized.pkl
data/MIMIC-III/phenotyping_normalized.pkl
data/MIMIC-III/decompensation_normalized.pkl
data/MIMIC-III/lengthofstay_normalized.pkl
```

Do not commit raw data, processed pickles, checkpoints, logs, or any files copied
from another workspace data directory such as `SMART/data/`.

## Models And Ablations

Public experiment ids for `run_all_experiments.py`:

```text
smart
smart-smile-lean
smart-smile-lean-no-density
smart-smile-lean-no-mnar-bias
smart-smile-lean-no-film
```

Main ablation semantics:

- `smart-smile-lean-no-density`: removes local observation density `rho`.
- `smart-smile-lean-no-mnar-bias`: removes the sample-level co-missingness attention-logit bias.
- `smart-smile-lean-no-film`: removes only the post-attention Time-Affine/FiLM transform; `time_enc` is still available to the attention block for time-dependent MNAR gating.

The active time-gated CoMiss pathway uses a learnable per-head/per-pair scale
with shape `(H, V, V)` (normally `H=4`). A sample-level time gate with shape
`(B, H)` modulates it before broadcasting the bias to `(B, H, V, V)`.

The direct CLI also contains diagnostic switches such as `--abl-no-time-mnar`,
but these are not part of the main public ablation table.

## Quick Start

Inspect the generated commands before launching training:

```bash
python run_all_experiments.py --dry-run
```

Run SMILE-Lean on one dataset and one seed:

```bash
python run_all_experiments.py --models smart-smile-lean --datasets c12 --seeds 42
```

Run the public ablation set:

```bash
python run_all_experiments.py \
  --models smart smart-smile-lean smart-smile-lean-no-density smart-smile-lean-no-mnar-bias smart-smile-lean-no-film \
  --datasets c12 c19 \
  --seeds 1 42 3407
```

Re-evaluate existing checkpoints without retraining:

```bash
python run_all_experiments.py --eval-only \
  --models smart smart-smile-lean smart-smile-lean-no-mnar-bias \
  --datasets c12 c19 mimic_mortality mimic_decompensation \
  --seeds 1 42 3407
```

Legacy `no-density` and `no-film` checkpoints that used scalar `(H,)` CoMiss
scales are intentionally rejected and must be retrained under the current
`(H, V, V)` implementation.

One-off pretraining and finetuning:

```bash
python main_pretrain.py --dataset c12 --use-smile-lean \
  --mask-group-config experiments/bibm_smile/configs/selected_mask_groups.json
python main_finetune.py --dataset c12 --use-smile-lean
```

Outputs are written under:

```text
export/<dataset>/<model>/seed_<seed>/
```

## Audited Reproduction Runner

Generate structured missingness audit inputs from the training split:

```bash
python analysis/mnar_verification.py --split train --split-seed 42 --output-dir analysis/results/bibm_audit_fixed
```

Then inspect the AAAI/BIBM-style command grid:

```bash
python experiments/bibm_smile/run_bibm_experiments.py --dry-run
```

Important: `experiments/bibm_smile/run_bibm_experiments.py --models` expects
display names from the JSON config, for example:

```bash
python experiments/bibm_smile/run_bibm_experiments.py --dry-run \
  --models Backbone SMILE-Full "SMILE w/o Density" "SMILE w/o CoMiss Bias" "SMILE w/o Time-Affine" \
  --datasets c12 --seeds 42
```

Aggregate completed logs:

```bash
python experiments/bibm_smile/aggregate_results.py
```

Validation-selected F1/minPSE is the default aggregation protocol. Use
`--threshold-protocol benchmark` only when explicitly reproducing the older
test-optimized reporting protocol.

Generated results are ignored by Git:

```text
analysis/results/
export/
experiments/bibm_smile/results/
figs/
```

## Citation

```bibtex
@inproceedings{smile2026,
  title = {SMILE: Missingness-Aware Learning for Irregular Clinical Time Series},
  author = {Anonymous},
  booktitle = {AAAI},
  year = {2026}
}
```

This code builds on SMART:

```bibtex
@inproceedings{yu2024smart,
  title = {SMART: Towards Pre-trained Missing-Aware Model for Patient Health Status Prediction},
  author = {Yu, Zhihao and Chu, Xu and Jin, Yujie and Wang, Yasha and Zhao, Junfeng},
  booktitle = {NeurIPS},
  year = {2024}
}
```
