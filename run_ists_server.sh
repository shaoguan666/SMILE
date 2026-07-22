#!/usr/bin/env bash
# ============================================================================
# Server-side: produce the complete, manifest-consistent ists_plm sensor grid.
# Trains all 6 ists_plm (2 datasets x 3 seeds) at the matched batch size (6)
# on 2 GPUs, then runs the sensor-robustness eval. Transfer the two output
# subtrees (printed at the end) back to the local repo via ToDesk.
#
# Run from the repo root:  bash run_ists_server.sh
# (activate the python env that has transformers, or let stage 0 install it)
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

# Expected manifest hashes (must match the local grid so results merge in).
MORT_SHA=6ff935b4d6b95c2de97acce35f691ef7f9b9237cffa70b6c8c36ab673acd4d9b
DECOMP_SHA=30aef0b11de835f31c7f73624ac80fd5744dbc433dc8f4be1140d61227137646

check_sha() {
  python - "$MORT_SHA" "$DECOMP_SHA" <<'PY'
import hashlib, sys
exp={"mimic_mortality":sys.argv[1],"mimic_decompensation":sys.argv[2]}
bad=0
for ds,e in exp.items():
    p=f"export/sensor_robustness_v1/manifests/{ds}_test_seed42.npz"
    try:
        got=hashlib.sha256(open(p,'rb').read()).hexdigest()
    except FileNotFoundError:
        print(f"{ds}: MISSING"); bad=1; continue
    ok=(got==e); print(f"{ds}: {got[:16]} {'OK' if ok else 'MISMATCH(exp '+e[:16]+')'}")
    bad=bad or (0 if ok else 1)
sys.exit(bad)
PY
}

echo "==== [0] transformers 环境 ===="
if ! python -c "import transformers" 2>/dev/null; then
  echo "未检测到 transformers，安装中 (py3.13 走 --no-deps 绕过 tokenizers 编译)..."
  pip install -q tokenizers huggingface_hub==0.16.4 safetensors regex
  pip install -q transformers==4.30.1 --no-deps
fi
python -c "from experiments.recent_baselines.run_ists_plm_mimic import load_upstream_transformers_extensions as f; f(); import transformers; print('transformers OK', transformers.__version__)" || {
  echo "transformers 注入失败，终止"; exit 1; }

echo "==== [0b] 检查 PLM 权重 ===="
for f in external/ists-plm/PLMs/gpt2/config.json external/ists-plm/PLMs/bert-base-uncased/config.json; do
  [ -f "$f" ] || { echo "缺 PLM 权重: $f  (按 external/ists-plm/README.md 下载 gpt2 与 bert-base-uncased 到 PLMs/)"; exit 1; }
done
echo "PLM 权重就位"

echo "==== [1] manifest 一致性 ===="
if ! check_sha; then
  echo "manifest sha 不符，尝试重新生成..."
  python experiments/generate_sensor_manifest.py
fi
if ! check_sha; then
  echo "重新生成后仍与本地不一致。请用 ToDesk 把本地这两个文件覆盖到服务器同路径后重跑："
  echo "  export/sensor_robustness_v1/manifests/mimic_mortality_test_seed42.npz"
  echo "  export/sensor_robustness_v1/manifests/mimic_decompensation_test_seed42.npz"
  exit 1
fi
echo "manifest 与本地一致"

echo "==== [2] 训练 6 个 ists_plm (matched batch=6, 2 卡). 预计数小时~十几小时 ===="
python experiments/recent_baselines/run_recent_mimic_baselines.py \
  --models ists_plm \
  --datasets mimic_mortality mimic_decompensation \
  --seeds 1 42 3407 \
  --gpus 0 1 --force

echo "==== [3] 清理旧 ists 结果 + sensor 评测 ===="
rm -rf export/sensor_robustness_v1/runs/mimic_mortality/ists_plm \
       export/sensor_robustness_v1/runs/mimic_decompensation/ists_plm
python experiments/run_sensor_robustness.py --profile paper --models ists_plm \
  --datasets mimic_mortality mimic_decompensation --seeds 1 42 3407 \
  --devices 0 1 --resume --allow-incomplete-smoke

echo "==== [4] 自检 ===="
python - <<'PY'
import json,glob,collections
conds=collections.Counter(); seeds=collections.defaultdict(set); man=collections.defaultdict(set)
npat=collections.Counter(); status=collections.Counter()
for f in glob.glob("export/sensor_robustness_v1/runs/mimic_*/ists_plm/**/run.json",recursive=True):
    d=json.load(open(f,encoding="utf-8")); ds=d["dataset"]
    conds[ds]+=1; seeds[ds].add(d["train_seed"]); man[ds].add(d["manifest_sha256"][:12])
    npat[(ds,d["metrics"]["n_patients"])]+=1; status[d["status"]]+=1
for ds in ("mimic_mortality","mimic_decompensation"):
    print(f"  {ds}: conds={conds[ds]} seeds={sorted(seeds[ds])} manifest={sorted(man[ds])}")
print("  n_patients:",dict(npat)," status:",dict(status))
print("  期望: 每 dataset conds=78, seeds=[1,42,3407], 单一 manifest, mortality n=2114 / decomp n=6251")
PY

echo ""
echo "==== 完成. 用 ToDesk 把下面两个目录传回本地同路径 (覆盖) ===="
echo "  export/sensor_robustness_v1/runs/mimic_mortality/ists_plm/"
echo "  export/sensor_robustness_v1/runs/mimic_decompensation/ists_plm/"
echo "回传后本地全网即 6 模型 x 2 数据集 x 3 seed 全齐且同一套 manifest。"
