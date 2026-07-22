#!/usr/bin/env bash
# ============================================================================
# Server-side c12/c19 FULL 6-model sensor-robustness grid.
# Everything runs on THIS server against THIS server's own c12/c19 data and
# manifest, so it is internally self-consistent. (MIMIC is finalized locally
# and is NOT touched here — this server's MIMIC split differs from local.)
#
# Produces the two deltas to ToDesk back to the local repo:
#   export/sensor_robustness_v1/runs/c12/  and  .../runs/c19/
#
# Usage (from repo root, after unzip -o smile_c12c19_update.zip):
#   bash run_server_all.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 用哪些 GPU (逗号分隔). 显存被别的进程占了就设成空闲卡, 例如:  GPUS=1 bash run_server_all.sh
GPUS="${GPUS:-0,1}"
GPU_SPACE="${GPUS//,/ }"
NPROC=$(printf '%s' "$GPUS" | tr ',' '\n' | grep -c .)
echo "使用 GPU: $GPUS (nproc=$NPROC)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

echo "############ [0] 依赖 ############"
if ! python -c "import transformers" 2>/dev/null; then
  pip install -q tokenizers huggingface_hub==0.16.4 safetensors regex
  pip install -q transformers==4.30.1 --no-deps
fi
if ! python -c "import torch_geometric" 2>/dev/null; then
  echo "安装 torch_geometric..."; pip install -q torch_geometric
fi
python -c "import transformers,torch_geometric;print('deps OK',transformers.__version__,torch_geometric.__version__)" \
  || { echo "依赖缺失，终止"; exit 1; }
for f in external/ists-plm/PLMs/gpt2/config.json external/ists-plm/PLMs/bert-base-uncased/config.json; do
  [ -f "$f" ] || { echo "[FATAL] 缺 PLM 权重: $f"; exit 1; }
done

echo "############ [1] 生成 c12/c19 manifest (本服务器自有数据, 全 6 模型统一用它) ############"
python experiments/generate_sensor_manifest.py --datasets c12 c19
for ds in c12 c19; do
  python -c "import hashlib;print('  $ds manifest sha',hashlib.sha256(open('export/sensor_robustness_v1/manifests/${ds}_test_seed42.npz','rb').read()).hexdigest()[:16])"
done

echo "############ [2] 重训坏掉的 c12 smart-smile-lean seed_42 (embed 32x4 bug) ############"
if [ -f export/c12/smart-smile-lean/seed_42/checkpoint-prc.pth ]; then
  BAD=$(python -c "import torch;s=torch.load('export/c12/smart-smile-lean/seed_42/checkpoint-prc.pth',map_location='cpu',weights_only=False)['encoder'];k=[x for x in s if x.endswith('embedder.embed.0.weight')][0];print(0 if tuple(s[k].shape)==(32,3) else 1)")
else
  BAD=1
fi
if [ "$BAD" = "1" ]; then
  rm -f export/c12/smart-smile-lean/seed_42/checkpoint-mse.pth \
        export/c12/smart-smile-lean/seed_42/checkpoint-prc.pth
  python run_all_experiments.py --models smart-smile-lean --datasets c12 --seeds 42 \
    --use-torchrun --nproc-per-node "$NPROC" --devices "$GPUS" --batch-size 32 --export-root export --force
  python -c "import torch,sys;s=torch.load('export/c12/smart-smile-lean/seed_42/checkpoint-prc.pth',map_location='cpu',weights_only=False)['encoder'];k=[x for x in s if x.endswith('embedder.embed.0.weight')][0];sys.exit(0 if tuple(s[k].shape)==(32,3) else 1)" \
    && echo "  c12 smile seed_42 OK (32x3)" || { echo "[FATAL] c12 smile seed_42 仍异常"; exit 1; }
else
  echo "  c12 smile seed_42 已是 32x3, 跳过重训"
fi

echo "############ [3] 训练 4 基线 x c12/c19 x 3 seed = 24 jobs (2 卡, 最耗时) ############"
python experiments/recent_baselines/run_recent_mimic_baselines.py \
  --models ists_plm wavegnn misstm atenet \
  --datasets c12 c19 --seeds 1 42 3407 --gpus $GPU_SPACE

echo "############ [4] 6 模型 sensor 评测 (c12/c19) ############"
rm -rf export/sensor_robustness_v1/runs/c12 export/sensor_robustness_v1/runs/c19
python experiments/run_sensor_robustness.py --profile paper \
  --models smile smart ists_plm wavegnn misstm atenet \
  --datasets c12 c19 --seeds 1 42 3407 --ks 0 4 6 10 14 16 \
  --devices $GPU_SPACE --resume --allow-incomplete-smoke

echo "############ [5] 自检 ############"
python - <<'PY'
import json,glob,collections
conds=collections.Counter(); seeds=collections.defaultdict(set); man=collections.defaultdict(set)
for f in glob.glob("export/sensor_robustness_v1/runs/c1*/**/run.json",recursive=True):
    d=json.load(open(f,encoding="utf-8")); k=(d["dataset"],d["model"])
    conds[k]+=1; seeds[k].add(d["train_seed"]); man[d["dataset"]].add(d["manifest_sha256"][:12])
for ds in ("c12","c19"):
    for m in ("smart-smile-lean","smart","ists_plm","wavegnn","misstm","atenet"):
        k=(ds,m); print(f"  [{'OK' if conds[k]==78 else 'CHECK'}] {ds} {m:18s} conds={conds[k]} seeds={sorted(seeds[k])}")
    print(f"    {ds} manifest sha: {sorted(man[ds])}  (每 dataset 应只有一个)")
print("期望每格 conds=78 (1 clean + 5 k x 5 rep)")
PY

echo ""
echo "############ 完成. 用 ToDesk 传回本地同路径 ############"
echo "  export/sensor_robustness_v1/runs/c12/"
echo "  export/sensor_robustness_v1/runs/c19/"
