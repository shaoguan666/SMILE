# 交接文档：图 A（Leave-Random-Sensor-Out 鲁棒性曲线）

面向 AAAI 投稿的"缺失鲁棒性"实验图（Raindrop / ViTST 那类 "AUPRC/AUROC vs 缺失率" 曲线）。
本文记录已搭好的基础设施、如何复现、当前结论，以及把 smoke test 变成正式图 A 还差什么。

最后更新：2026-07-20

---

## 1. 背景与决策

- **目标**：一张 "Ours 一直在最上面、且降解最平缓" 的鲁棒性曲线，回应审稿人对标准
  leave-random-sensor-out 协议的预期。
- **叙事注意点**：本图用的是 **MCAR**（随机丢整条传感器通道），而论文主线是 **MNAR /
  observation-policy**。曾担心随机丢弃会抹平 SMILE 的信息性缺失优势 —— **smoke test 证明没有，
  优势反而随缺失加重而扩大**（见 §5）。所以此图安全，可铺。定位成对主结果（audit alignment）
  的**防御性补充实验**，不是核心卖点。
- **决策**：smoke test 通过，方向强利好，**值得铺成完整图 A**。

---

## 2. 已构建的基础设施（keystone 设计）

核心思想：**丢弃逻辑只在一处**（`data/dataloader.py` 的 `collate_fn`），由环境变量控制。
SMILE 和 4 个 baseline **共用同一个 `collate_fn`**，因此一处注入即覆盖全部 5 个模型，
且对同一 `(seed, ratio)` 丢**完全相同的通道**，保证公平；不设环境变量时为 no-op，
训练行为字节级不变。

丢弃是 **channel-wise（leave-random-sensor-out）**：每个样本随机丢 `round(ratio * V)` 条
传感器通道，按 `sample_id` 确定性选取（与 batch 顺序/大小无关）。区别于代码里已有的
`apply_mnar_dropout`（那是 element-wise 逐点丢弃，训练用）。

### 环境变量

| 变量 | 含义 | 默认 |
|---|---|---|
| `SMILE_EVAL_SENSOR_DROP` | 丢弃比例，(0,1]；0 或未设 = 不丢 | `0` |
| `SMILE_EVAL_SENSOR_DROP_SEED` | 通道选取的随机种子 | `0` |

### 改动/新增文件

| 文件 | 改动 |
|---|---|
| `data/dataloader.py` | 新增 `_eval_sensor_dropout()`，`collate_fn` 末尾调用；env-gated，默认 no-op |
| `experiments/recent_baselines/run_missing_sweep.py` | **新增**。sweep 编排器：对每个 (model, ratio, seed) 起一个 eval-only 子进程，汇总 AUROC/AUPRC |
| `experiments/recent_baselines/run_{atenet,ists_plm,misstm,wavegnn}_mimic.py` | 各加 `--eval-only` + `--eval-output-dir`：加载 `best_auprc.pt` 直接评测，不重训 |

> 注：`main_finetune.py`、`run_all_experiments.py` 在 git 里也显示 modified，但那是本工作**之前**
> 已有的改动，非本次所加。SMILE 评测走 `run_all_experiments.py --eval-only`，sweep 脚本通过
> **备份/还原** `eval_results.json` 来避免污染 export 目录（已验证目录无残留）。

---

## 3. 如何复现 / 扩展

### 跑当前 smoke（已完成）
```bash
python experiments/recent_baselines/run_missing_sweep.py \
  --models smile ists_plm wavegnn atenet misstm \
  --ratios 0.0 0.3 0.5 \
  --seeds 1 42 3407
```
输出：`export/missing_sweep/mimic_decompensation_sweep.json` + 控制台表格 +
每次子进程日志在 `export/missing_sweep/logs/`。

### run_missing_sweep.py 参数
- `--dataset`（默认 `mimic_decompensation`；单数据集，扩到 mortality 时改这里）
- `--ratios`（默认 `0.0 0.3 0.5`）
- `--seeds`（默认 `1 42 3407`）
- `--models`（`smile ists_plm wavegnn atenet misstm` 任意子集）
- `--limit-test N`（快速冒烟用，只评测前 N 个样本）
- `--dry-run`（只打印将要执行的命令，不跑）

### 手动单点复现（调试用）
```bash
# baseline，例如 wavegnn，ratio=0.5，seed=42
SMILE_EVAL_SENSOR_DROP=0.5 SMILE_EVAL_SENSOR_DROP_SEED=42 \
  python experiments/recent_baselines/run_wavegnn_mimic.py \
  --dataset mimic_decompensation --seed 42 --eval-only \
  --eval-output-dir export/missing_sweep/tmp

# SMILE，同上
SMILE_EVAL_SENSOR_DROP=0.5 SMILE_EVAL_SENSOR_DROP_SEED=42 \
  python run_all_experiments.py --models smart-smile-lean \
  --datasets mimic_decompensation --seeds 42 --eval-only
```

### 输出 JSON 结构
```
{ dataset, seeds, ratios,
  models: { <model>: { "<ratio>": {auroc_mean, auroc_std, auprc_mean, auprc_std, n} } } }
```

---

## 4. checkpoint 现状

- **SMILE**：`export/mimic_decompensation/smart-smile-lean/seed_{1,42,3407}/checkpoint-prc.pth` 全在。
  该模型此前**没有 eval_results.json**（也是 density-window 曲线 dw5 中心点缺的那个数）；
  ratio=0 的 sweep 结果可顺便补上这个数。
- **4 个 baseline**：`export/recent_baselines/mimic_decompensation/<model>/seed_{1,42,3407}/best_auprc.pt` 全在。
- **已知损坏**：`export/recent_baselines/mimic_decompensation/ists_plm/seed_1/best_auprc.pt`
  只有 3.6 MB（正常 ~591 MB），torch.load 报 "failed finding central directory"。
  sweep 里自动跳过，ISTS-PLM 当前只有 2 seed。**需重训/重存该点以拿回第 3 seed。**

---

## 5. 当前结果（mimic_decompensation，AUPRC mean±std）

| 模型 | 0% | 30% | 50% | 掉幅 |
|---|---|---|---|---|
| **SMILE (ours)** | **0.781**±.012 | **0.674**±.025 | **0.606**±.025 | −0.175 |
| ISTS-PLM (2 seed) | 0.739±.008 | 0.621±.005 | 0.521±.028 | −0.218 |
| WaveGNN | 0.722±.006 | 0.499±.024 | 0.360±.038 | −0.362 |
| MissTSM | 0.631±.021 | 0.417±.034 | 0.324±.008 | −0.307 |
| ATeNet | 0.507±.007 | 0.122±.022 | 0.078±.009 | −0.429 |

结论：SMILE 每个缺失率都居上；领先次优从干净时 +0.042 扩大到 50% 时 +0.085；降解最平缓。
AUROC 同趋势（见 JSON）。

---

## 6. 变成正式图 A 还差什么（TODO）

1. **补缺失率点**：加 `0.1 0.2 0.4`，凑成 5 点平滑曲线（改 `--ratios`）。
2. **补第二个数据集**：`mimic_mortality`（4 个 baseline 的 checkpoint 也在，SMILE 同）。
   改 `--dataset mimic_mortality` 即可；确认 mortality 下 SMILE 模型名/路径一致。
3. **多指标 panel**：至少 AUROC + AUPRC 双 panel；可再加 minPSE/F1（`test()` 已算 threshold 指标）。
4. **修 ISTS-PLM seed_1**：重存 checkpoint，拿回第 3 seed。
5. **画图脚本**：读 sweep JSON → 出 2×N 网格图（SMILE 加粗高亮，baseline 灰/虚线，见预览图配色）。
6. **（可选）显著性**：seed 间做配对检验，标注 SMILE 领先是否显著。

---

## 7. 注意事项 / 坑

- **只用于 eval**：env 变量一旦设置会作用于 collate_fn 的**所有** loader（含 train/val）。
  sweep 全程 eval-only 无训练，故安全；**切勿在训练时残留该 env 变量**。
- SMILE eval 会写 `eval_results.json` 到 export 目录，sweep 脚本靠备份/还原防污染 ——
  若脚本中途被 kill，检查 `export/mimic_decompensation/smart-smile-lean/seed_*/` 有无残留
  `eval_results.json`（正常应无）。
- ISTS-PLM 每次加载 BERT+GPT2（~591MB），全测试集评测慢；全量 sweep 以它为瓶颈
  （3 ratio × 3 seed 约占大头，整轮 1.5–2.5h，单卡 RTX 4060）。
- 显存：sweep 串行起子进程，同时只有一个模型在 GPU，8GB 够。
- 相关联的其它图：density-window 扫描（dw1/3/5/7/9）是**另一回事**（训练期超参敏感度，
  非测试期缺失强度），当前判断价值低，见项目记忆 `smile-clean-nocurriculum-control`。
