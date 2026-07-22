#!/usr/bin/env bash
# ============================================================================
# Server-side: full 6-model sensor-robustness grid on c12 (PhysioNet-2012)
# and c19 (PhysioNet-2019). Retrains the one bad SMILE checkpoint and all
# plain-SMART checkpoints under the same fixed split/training budget, trains the
# 4 recent baselines, then evaluates all 6 models. Transfer the two runs
# subtrees and refreshed SMART checkpoints back (printed at the end).
#
# REQUIRES the updated code on the server first (see file list in chat):
#   main_finetune.py, main_pretrain.py, run_all_experiments.py,
#   data/challenge2012.py, data/challenge2019.py,
#   experiments/generate_sensor_manifest.py, experiments/run_sensor_robustness.py,
#   experiments/plot_sensor_robustness.py, experiments/recent_baselines/baseline_utils.py,
#   run_{ists_plm,wavegnn,misstm,atenet}_mimic.py, run_recent_mimic_baselines.py
#
# Run from repo root:  bash run_c12c19_server.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

require_cli_options() {
  local program="$1"
  shift
  local help
  help="$(python "$program" --help)"
  local option
  for option in "$@"; do
    if [[ "$help" != *"$option"* ]]; then
      echo "Required option $option is absent from $program."
      echo "This server checkout is stale; sync the updated evaluation files before running."
      exit 1
    fi
  done
}

C12_SHA=335144ab928bb048544d574902bdf9b7ab926cada74bd637a4f14c915c868456
C19_SHA=27c7d499074e34070bf5668cb296e08b2ca87d278bac4031bf66b70de151b9d6
SPLIT_SEED=42
TRAIN_DEVICES=0,1
TRAIN_NPROC=2
# Per-process batch 32 x 2 DDP processes preserves the single-GPU global batch 64.
TRAIN_BATCH_PER_GPU=32

# Validate both launch layers before deleting a checkpoint or starting jobs.
# A stale main_finetune.py exits immediately with argparse return code 2.
require_cli_options main_finetune.py --eval-only --eval-output-dir --sensor-manifest --sensor-ks --sensor-replicates --sensor-resume --split-seed
require_cli_options run_all_experiments.py --eval-only --eval-output-root --sensor-manifest --sensor-ks --sensor-replicates --sensor-resume --split-seed
python - <<'PY'
import inspect
from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019

for loader in (load_challenge_2012, load_challenge_2019):
    if "split_seed" not in inspect.signature(loader).parameters:
        raise SystemExit(
            f"{loader.__module__}.{loader.__name__} must accept split_seed; "
            "sync data/challenge2012.py and data/challenge2019.py before running."
        )
PY

echo "==== [0] 依赖 ===="
# ists_plm 需要 transformers (py3.13 走 --no-deps 绕过 tokenizers 编译)
if ! python -c "import transformers" 2>/dev/null; then
  # transformers 4.30.1 requires tokenizers < 0.14.  Always force the
  # compatible pair here: an unpinned tokenizers install selects newer builds.
  echo "Repairing transformers/tokenizers compatibility..."
  python -m pip install -q --upgrade --force-reinstall \
    "transformers==4.30.1" "tokenizers==0.13.3" \
    "huggingface_hub==0.16.4" safetensors regex
fi
# wavegnn 需要 torch_geometric (服务器之前缺这个)
if ! python -c "import torch_geometric" 2>/dev/null; then
  echo "安装 torch_geometric..."; pip install -q torch_geometric
fi
python -c "import transformers,torch_geometric; print('deps OK', transformers.__version__, torch_geometric.__version__)" \
  || { echo "依赖缺失，终止"; exit 1; }
# ists_plm 需要 PLM 权重
for f in external/ists-plm/PLMs/gpt2/config.json external/ists-plm/PLMs/bert-base-uncased/config.json; do
  [ -f "$f" ] || { echo "缺 PLM 权重: $f"; exit 1; }
done

echo "==== [1] 生成 c12/c19 manifest 并校验 sha ===="
python experiments/generate_sensor_manifest.py --datasets c12 c19
python - "$C12_SHA" "$C19_SHA" <<'PY'
import hashlib, sys
exp={"c12":sys.argv[1],"c19":sys.argv[2]}
bad=0
for ds,e in exp.items():
    p=f"export/sensor_robustness_v1/manifests/{ds}_test_seed42.npz"
    got=hashlib.sha256(open(p,'rb').read()).hexdigest()
    ok=(got==e); print(f"{ds}: {got[:16]} {'OK' if ok else 'MISMATCH'}"); bad=bad or (0 if ok else 1)
sys.exit(bad)
PY
[ $? -eq 0 ] || { echo "manifest sha 与本地不一致，请 ToDesk 覆盖本地 c12/c19 manifest 后重跑"; exit 1; }

echo "==== [2] 重训坏掉的 c12 smart-smile-lean seed_42 (embed 32x4 bug) ===="
rm -f export/c12/smart-smile-lean/seed_42/checkpoint-mse.pth \
      export/c12/smart-smile-lean/seed_42/checkpoint-prc.pth
python run_all_experiments.py --models smart-smile-lean --datasets c12 --seeds 42 \
  --split-seed "$SPLIT_SEED" \
  --use-torchrun --nproc-per-node "$TRAIN_NPROC" --devices "$TRAIN_DEVICES" \
  --batch-size "$TRAIN_BATCH_PER_GPU" --export-root export --force
python -c "import torch;s=torch.load('export/c12/smart-smile-lean/seed_42/checkpoint-prc.pth',map_location='cpu',weights_only=False)['encoder'];k=[x for x in s if x.endswith('embedder.embed.0.weight')][0];import sys;sys.exit(0 if tuple(s[k].shape)==(32,3) else 1)" \
  && echo "c12 smile seed_42 重训 OK (embed 32x3)" || { echo "c12 smile seed_42 仍异常"; exit 1; }

echo "==== [3] 同口径重训 plain SMART: c12/c19 x seeds 1/42/3407 ===="
SMART_RETRAIN_STARTED="$(date +%s)"
export SMART_RETRAIN_STARTED SPLIT_SEED TRAIN_NPROC TRAIN_BATCH_PER_GPU
python run_all_experiments.py --models smart --datasets c12 c19 --seeds 1 42 3407 \
  --split-seed "$SPLIT_SEED" \
  --use-torchrun --nproc-per-node "$TRAIN_NPROC" --devices "$TRAIN_DEVICES" \
  --batch-size "$TRAIN_BATCH_PER_GPU" --export-root export --force

# Fail before sensor evaluation if any legacy SMART checkpoint survived or if
# the new logs do not record the fixed split. Persist a sidecar so transferred
# checkpoints retain auditable split/batch/hash provenance.
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

started = int(os.environ["SMART_RETRAIN_STARTED"])
split_seed = int(os.environ["SPLIT_SEED"])
nproc = int(os.environ["TRAIN_NPROC"])
batch_per_gpu = int(os.environ["TRAIN_BATCH_PER_GPU"])

for dataset in ("c12", "c19"):
    for seed in (1, 42, 3407):
        run_dir = Path("export") / dataset / "smart" / f"seed_{seed}"
        checkpoint = run_dir / "checkpoint-prc.pth"
        log_path = run_dir / "training.log"
        if not checkpoint.is_file() or checkpoint.stat().st_mtime < started:
            raise SystemExit(f"SMART checkpoint was not refreshed: {checkpoint}")
        if not log_path.is_file() or f'"split_seed": {split_seed}' not in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            raise SystemExit(f"SMART log lacks split_seed={split_seed}: {log_path}")

        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        provenance = {
            "protocol": "matched-c12c19-smart-retrain-v1",
            "dataset": dataset,
            "model": "smart",
            "train_seed": seed,
            "split_seed": split_seed,
            "world_size": nproc,
            "batch_size_per_process": batch_per_gpu,
            "global_batch_size": nproc * batch_per_gpu,
            "checkpoint": checkpoint.name,
            "checkpoint_epoch": int(checkpoint_payload["epoch"]),
            "checkpoint_sha256": digest,
        }
        (run_dir / "training_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"SMART provenance OK: {dataset}/seed_{seed} "
            f"split={split_seed} sha={digest[:12]}"
        )
PY

echo "==== [4] 训练 4 个基线 x c12/c19 x 3 seed (24 jobs, 2 卡). 最耗时 ===="
python experiments/recent_baselines/run_recent_mimic_baselines.py \
  --models ists_plm wavegnn misstm atenet \
  --datasets c12 c19 --seeds 1 42 3407 --gpus 0 1

echo "==== [5] 6 模型 sensor 评测 (c12/c19) ===="
# A changed checkpoint cannot safely resume its old condition files. Preserve
# the previous grid as a recoverable archive, then build one clean matched grid
# for all methods so aggregation cannot mix old and new SMART artifacts.
SENSOR_ARCHIVE="export/sensor_robustness_v1/archive/c12c19_before_matched_$(date -u +%Y%m%dT%H%M%SZ)"
export SENSOR_ARCHIVE
python - <<'PY'
import os
import shutil
from pathlib import Path

archive = Path(os.environ["SENSOR_ARCHIVE"])
for dataset in ("c12", "c19"):
    source = Path("export/sensor_robustness_v1/runs") / dataset
    if not source.exists():
        continue
    destination = archive / dataset
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    print(f"Archived {source} -> {destination}")
PY
python experiments/run_sensor_robustness.py --profile paper \
  --models smile smart ists_plm wavegnn misstm atenet \
  --datasets c12 c19 --seeds 1 42 3407 --ks 0 4 6 10 14 16 \
  --split-seed "$SPLIT_SEED" --devices 0 1

echo "==== [6] 自检 ===="
python - <<'PY'
import collections
import glob
import json
from pathlib import Path

conds=collections.Counter(); seeds=collections.defaultdict(set)
for f in glob.glob("export/sensor_robustness_v1/runs/c1*/**/run.json",recursive=True):
    d=json.load(open(f,encoding="utf-8")); k=(d["dataset"],d["model"])
    conds[k]+=1; seeds[k].add(d["train_seed"])
bad_grids = []
for ds in ("c12","c19"):
    for m in ("smart-smile-lean","smart","ists_plm","wavegnn","misstm","atenet"):
        k=(ds,m)
        ok = conds[k] == 78 and seeds[k] == {1, 42, 3407}
        print(f"  [{'OK' if ok else 'CHECK'}] {ds} {m:18s} conds={conds[k]} seeds={sorted(seeds[k])}")
        if not ok:
            bad_grids.append(k)
print("期望每格 conds=78 (3 seed x 26 = 1 clean + 5 k x 5 rep)")
if bad_grids:
    raise SystemExit(f"Incomplete sensor grids: {bad_grids}")

# The SMART sensor artifacts must point to exactly the newly retrained,
# split-matched checkpoints recorded above.
for ds in ("c12", "c19"):
    for seed in (1, 42, 3407):
        provenance_path = Path("export") / ds / "smart" / f"seed_{seed}" / "training_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        clean_path = (
            Path("export/sensor_robustness_v1/runs") / ds / "smart"
            / f"seed_{seed}" / "k_0" / "clean" / "run.json"
        )
        clean = json.loads(clean_path.read_text(encoding="utf-8"))
        if clean["split_seed"] != provenance["split_seed"]:
            raise SystemExit(f"SMART split mismatch: {clean_path}")
        if clean["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
            raise SystemExit(f"SMART checkpoint hash mismatch: {clean_path}")
        print(
            f"  [OK] {ds} smart seed_{seed}: "
            f"split={clean['split_seed']} sha={clean['checkpoint_sha256'][:12]}"
        )
PY

echo ""
echo "==== 完成. 用 ToDesk 传回本地同路径 ===="
echo "  export/sensor_robustness_v1/runs/c12/"
echo "  export/sensor_robustness_v1/runs/c19/"
echo "  export/c12/smart/"
echo "  export/c19/smart/"
