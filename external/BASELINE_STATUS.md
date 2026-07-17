# Recent baseline source status

Checked on 2026-07-16.

| Method | Paper-advertised repository | Local status | MIMIC integration |
|---|---|---|---|
| ATENet | <https://github.com/shlee-labs/ATENet> | Cloned at `external/atenet` (base commit `dc3149c`) | Adapter and seed-42 full-cohort runs completed; see below |
| Sequence--Image SSL | <https://github.com/zju-d3/AAAI25-Irregular-Medical-Time-Series> | Cloned at `external/sequence_image_ssl` (commit `cc9f6ad`), but upstream contains only a 38-byte README and no implementation | Blocked on upstream source release |
| SPECTRA | <https://github.com/qinxin8021/SPECTRA> | `git clone` returns `Repository not found`; no local source was fabricated | Blocked on upstream repository availability |

The paper table labels original P12/P19 numbers from these methods as
source-reported. They are not treated as matched-protocol results.

## ATENet MIMIC adaptation

The adapter preserves the official ATENet architecture and objectives while
using SMILE's data splits, normalization, observation masks, and threshold-free
evaluation. The local full-cohort runs use seeds 1, 42, and 3407,
full-precision training, and validation AUPRC for checkpoint selection:

Two local compatibility patches remain visible in `external/atenet/models.py`:
random masks follow the input device instead of calling `.cuda()` directly,
and the attention-mask sentinel follows the tensor dtype so AMP does not
overflow. The architecture and loss definitions are otherwise unchanged.

| Task | AUROC (mean ± sample std) | AUPRC (mean ± sample std) | Test n | Artifact |
|---|---:|---:|---:|---|
| MIMIC mortality | 0.85287 ± 0.00291 | 0.49689 ± 0.01146 | 2,114 | `export/recent_baselines/mimic_mortality/atenet/aggregate_seeds_1_42_3407.json` |
| MIMIC decompensation | 0.90957 ± 0.00404 | 0.50678 ± 0.00827 | 6,251 | `export/recent_baselines/mimic_decompensation/atenet/aggregate_seeds_1_42_3407.json` |

These are three-seed local adaptation results, not source-paper values. They
populate only the MIMIC columns of a separate local row in the matched-protocol
block; the paper's C12/C19 values remain daggered source-paper values.
