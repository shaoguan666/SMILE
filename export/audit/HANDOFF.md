# Observation Policy Audit — 交接文档 (v4, 2026-07-14)

## 0. 一句话现状

无泄漏 Audit v4 已完成，**冻结 go/no-go 判定 4/4 数据集 GO**：临床缺失中的"跨变量共观测结构"包含超越时间/近因/run-length/单变量历史的预测信息。下一步是 D2 hidden-state FiLM 注入下游（尚未开始）。

---

## 1. 背景 / 为什么做这个 audit

- SMILE (`smart-smile-lean`) 主线是静态共缺失矩阵 `MNARCooccurrenceEncoder`。三个诊断对照 `abl_global_comiss` / `abl_random_bias` / pmatch 不掉点 => 静态共缺失学到的主要是队列级固定检查套餐 + 边际频率，不是病人特异结构。
- 拟议新方向：Observation Policy Innovation（预测观测策略、用残差/表示注入）。
- 上游 go/no-go 判据：**动态观测策略预测能否稳定击败静态/浅层基线**。若不能，整条线是噪声，应止损并转 diagnostic 叙事。

---

## 2. 交付的文件

| 文件 | 作用 |
|---|---|
| `audit_observation_policy.py` | 单机诊断脚本（无下游）。严格因果预测自然 mask，跑一个 (dataset, seed) 配置 |
| `aggregate_audit.py` | 多 seed 聚合 + 冻结 go/no-go（D2 主假设、配对 t、患者 bootstrap） |
| `tests/test_observation_policy_audit.py` | 13 个单测：6 模型反事实因果 + 翻转公式 + run-length 向量化 + 静态 prior 约定 |
| `export/audit/<ds>_seed<0..4>.json` | 20 份逐 run 结果（validation，test 封存） |
| `export/audit/scores_<ds>_seed<0..4>.npz` | 20 份 per-record 分数（供患者 bootstrap） |
| `export/audit/_aggregate_v4.json` | 冻结聚合结论 |

---

## 3. 方法（v4，修正了 v1-v3 的问题）

**v1-v3 作废原因**：dynamic predictor 有因果泄漏（`(B*V,H,T)` 直接 reshape 把 hidden 轴与 time 轴错位混合），AUPRC 被虚高到 0.937-0.999。已修复（左 pad + `transpose(1,2)` 重排），并有单测守护。

**预测器阶梯**（全部严格因果，位置 t 只用 `<t` 信息）：
- S1 `P(m_v)` / S2 `P(m_{t,v})` 协议 / S3 persistence `m_{t-1}`
- S4 浅层 logistic（recent-k / run-length / TSLO，无时间、线性）
- **S4+** 浅层特征 + TimeEncoder + 与 D1 同款 MLP head（公平浅层基线）
- D1 单变量因果卷积 `m_{<t,v}`
- **D1-wide** 单变量，宽度自动匹配 D2 参数量（误差<5%，容量控制）
- **D2** 跨变量因果卷积 `m_{<t,:}`（**唯一预注册主方案**）
- **D2-shuffled** D2 架构/参数不变，用卷积线性性把非目标通道跨患者打乱（结构控制）
- D3 跨变量 + 临床值 `(m,m·x)_{<t,:}`（exploratory，非主方案）

**两个预测任务**：
- (A) mask nowcast：预测 `m_{t,v}`；maAUPRC/micro/Brier/NLL/ECE
- (B) transition event：预测翻转 `z=1[m_t!=m_{t-1}]`，score `q_flip=m_prev+pi-2*m_prev*pi`，在全部 valid `t>=1` 上算。这问"能否提前预判翻转"，不是"翻转已发生后猜新状态"

**聚合的严谨性**：D2 定为唯一主假设（不事后选 champion）；配对差值 + t_4=2.776（非 1.96）；强制配置校验（恰好 4 数据集 × 5 唯一 seed、同配置、`audit_version==4`、`test_evaluated==False`、无部分 GO）；患者级 paired bootstrap（300 次、按记录重采样、seed 平均预测）。

---

## 4. 冻结判定规则（预注册）

某数据集 SUPPORT <=> validation 上五项全部成立：
1. `D2-S4+` 均值 >= 0.02 且配对 t95 CI 下界 > 0（nowcast）
2. `D2-D1wide` CI 下界 > 0（容量控制）
3. `D2-D2shuffled` CI 下界 > 0（结构控制）
4. `D2-S4+` transition 均值 > 0 且 D2 transition-Brier <= S4+
5. 患者 bootstrap 的 `D2-S4+` 95% 百分位 CI 下界 > 0

整体 GO <=> >= 3/4 数据集 SUPPORT。

---

## 5. 最终结果：4/4 GO

| 数据集 | D2-S4+ (t95 CI) | D2-D1wide | D2-D2shuf | transition | 患者bootstrap CI | 判定 |
|---|---|---|---|---|---|---|
| c12 | +0.044 [.041,.048] | [+.041,+.044] | [+.060,+.064] | +0.048 | [+.045,+.054] | SUPPORT |
| c19 | +0.049 [.044,.055] | [+.048,+.050] | [+.068,+.069] | +0.039 | [+.051,+.059] | SUPPORT |
| mimic_mortality | +0.036 [.035,.037] | [+.016,+.018] | [+.033,+.036] | +0.043 | [+.036,+.041] | SUPPORT |
| mimic_decompensation | +0.041 [.039,.043] | [+.026,+.030] | [+.038,+.043] | +0.051 | [+.040,+.047] | SUPPORT |

D2 同时打赢 S4+/D1-wide/D2-shuffled => 增益是真实的跨变量共观测结构，不是时间、不是容量、不是自相关。患者 bootstrap CI 全部 > 0 => 纳入患者划分不确定性后结论依然稳。

---

## 6. 冻结的论文措辞

CAN write:
> "Historical cross-variable co-observation patterns contain predictive information beyond time, recency, run length, and per-variable history."

CANNOT write:
> "Patient clinical state drives observation policy."（D3 打不过 D2，只 c19 有微弱提升）

---

## 7. 已知局限

- c12/c19 只有单一固定患者划分；mimic 固定 `split_seed=42`。患者 bootstrap 已捕捉患者级不确定性，但**跨划分（不同 split_seed）不确定性**未覆盖；如需可对 mimic 变 split_seed 复跑（decomp 的 pickle 内含固定 split_sizes，loader 忽略 split_seed，需重划或患者 bootstrap）。
- transition AUPRC 对高 prevalence 变量偏乐观；已用 Brier/NLL 并行报告。
- CUDA teardown 偶发段错误（exit 139）出现在结果写盘之后，不影响结果；sweep 用幂等 runner（跳过已存在 JSON、单 run 段错误不中止）规避。

---

## 8. 复现命令

```bash
# 单配置
python audit_observation_policy.py --dataset c12 --seed 0 --epochs 20 --batch_size 64

# 全 sweep（4 数据集 x 5 seed；幂等，跳过已存在）
for ds in c12 c19 mimic_mortality mimic_decompensation; do
  for s in 0 1 2 3 4; do
    [ -f export/audit/${ds}_seed${s}.json ] || python audit_observation_policy.py --dataset $ds --seed $s --epochs 20
  done
done

# 聚合 + 冻结判定（含患者 bootstrap，约 15-20 min）
python aggregate_audit.py --datasets c12 c19 mimic_mortality mimic_decompensation
# 快速版（跳过 bootstrap，秒级，只验条件 1-4）
python aggregate_audit.py --datasets c12 c19 mimic_mortality mimic_decompensation --no-bootstrap

# 测试
PYTHONPATH=. python tests/test_observation_policy_audit.py
```

---

## 9. 下一步（未开始）：D2 hidden-state FiLM 注入

Audit v4 已 GO，方可进入下游注入。冻结的实现约束：
- policy encoder（D2）只读**自然 mask**，不能读人工 corruption mask
- 第一版**预训练并冻结 D2**，再注入
- 注入 D2 的 **hidden state**（不是 raw 残差；因数据显示策略结构而非数值驱动）
- FiLM **末层零初始化**，保证起点等价于 baseline
- **bounded residual `m-pi`** 仅作消融（不用可能爆值的 Pearson 标准化残差）
- **D3 不作主方法**，仅 audit ablation
- 下游结构/阈值在 **validation 冻结后**才开 test

首轮实验矩阵（冻结）：
1. 原始 SMILE
2. + time-only FiLM（参数控制）
3. + valid-density FiLM
4. + D2 pi-FiLM
5. **+ D2 hidden-FiLM（主方法）**
6. + bounded residual (m-pi)（消融）
7. + patient-shuffled D2 hidden（负对照）

集成点（代码）：`SMILELeanEncoder`（`models/smart.py`），注入在 `DensityMLPEmbedder` 输出之后；`policy_mask_clean` 已在 `main_pretrain.py:748` / `main_finetune.py:63` 备好。
