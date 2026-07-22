# Recent baseline source status

Checked on 2026-07-16.

| Method | Paper-advertised repository | Local status | MIMIC integration |
|---|---|---|---|
| ATENet | <https://github.com/shlee-labs/ATENet> | Cloned at `external/atenet` (base commit `dc3149c`) | Adapter and seed-42 full-cohort runs completed; see below |
| Sequence--Image SSL | <https://github.com/zju-d3/AAAI25-Irregular-Medical-Time-Series> | Cloned at `external/sequence_image_ssl` (commit `cc9f6ad`), but upstream contains only a 38-byte README and no implementation | Blocked on upstream source release |
| SPECTRA | <https://github.com/qinxin8021/SPECTRA> | `git clone` returns `Repository not found`; no local source was fabricated | Blocked on upstream repository availability |
| WaveGNN | <https://github.com/USC-InfoLab/WaveGNN> | Cloned at `external/wavegnn` (commit `2bd2c3c`) | SMILE MIMIC adapter and GPU smoke test complete; full matched runs pending target host |
| MissTSM | <https://github.com/abhilash-neog/SparseTimeSeriesModeling> | Cloned at `external/misstm` (commit `3d78abf`) | SMILE MIMIC adapter and GPU smoke test complete; full matched runs pending target host |
| ISTS-PLM | <https://github.com/usail-hkust/ISTS-PLM> | Cloned at `external/ists-plm` (commit `5c55fc9`) | Official GPT-2/BERT weights downloaded; SMILE MIMIC adapter and GPU smoke test complete; full matched runs pending target host |

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

## WaveGNN, MissTSM, and ISTS-PLM source results

The following values are copied directly from the original papers and appear
only in the daggered, source-reported paper block. They use the authors' data
partitions/training protocols and are excluded from matched-protocol ranking.

| Method | C12 AUPRC / AUROC | C19 AUPRC / AUROC | MIMIC mortality AUPRC / AUROC | Source protocol |
|---|---:|---:|---:|---|
| WaveGNN | 49.4 +/- 1.5 / 83.9 +/- 1.2 | 57.1 +/- 4.7 / 88.0 +/- 0.9 | 47.8 +/- 1.3 / not reported | 3 independent runs; fixed author splits |
| MissTSM | 43.8 +/- 1.1 / 82.2 +/- 0.5 | 56.5 +/- 1.2 / 88.8 +/- 1.3 | not reported | source classification protocol |
| ISTS-PLM | 57.6 +/- 3.3 / 87.6 +/- 1.4 | 56.9 +/- 5.0 / 89.4 +/- 2.2 | not reported | 5 fixed 80/10/10 partitions |

WaveGNN's MIMIC experiment reports F1 and AUPRC, not AUROC. Neither WaveGNN,
MissTSM, nor ISTS-PLM reports MIMIC decompensation classification. MissTSM and
ISTS-PLM do not report MIMIC mortality classification. These absent tasks must
therefore come from the local matched adapters rather than interpolation or
cross-paper substitution.

All three adapters passed end-to-end CUDA smoke tests on 2026-07-18. Smoke
artifacts under `tmp/baseline_smoke/` are diagnostic only and must never be
copied into the manuscript. The local laptop exposes one 8 GB RTX 4060, so the
full `1/42/3407` queue is reserved for the two-RTX-4090 experiment host.
