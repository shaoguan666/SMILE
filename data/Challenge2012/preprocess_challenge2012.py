"""
Preprocess Challenge 2012 (PhysioNet 2012) data.
Output: data_normalized.pkl, stat.pkl

Run from this directory:
    cd data/Challenge2012
    python preprocess_challenge2012.py

Required directory layout:
    data/Challenge2012/
        raw/           <-- 12000 .txt patient files (set-a + set-b + set-c)
        Outcomes-a.txt
        Outcomes-b.txt
        Outcomes-c.txt

Data source: https://physionet.org/content/challenge-2012/1.0.0/
"""
import os
import csv
import pickle
import numpy as np


# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------
STATIC_NAMES = ['Age', 'Gender', 'Height', 'ICUType']
DYNAMIC_NAMES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
    'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent',
    'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH',
    'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI',
    'TroponinT', 'Urine', 'WBC', 'Weight'
]
STATIC_NUM  = len(STATIC_NAMES)
DYNAMIC_NUM = len(DYNAMIC_NAMES)
STATIC_DICT  = {n: i for i, n in enumerate(STATIC_NAMES)}
DYNAMIC_DICT = {n: i for i, n in enumerate(DYNAMIC_NAMES)}

print(f'Static variables: {STATIC_NUM},  Dynamic variables: {DYNAMIC_NUM}')

# ---------------------------------------------------------------------------
# Read labels
# ---------------------------------------------------------------------------
y_dict = {}
for fname in ['Outcomes-a.txt', 'Outcomes-b.txt', 'Outcomes-c.txt']:
    with open(fname, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            y_dict[row[0]] = int(row[-1])
print(f'Labels loaded: {len(y_dict)}')

# ---------------------------------------------------------------------------
# Read patient time series
# ---------------------------------------------------------------------------
training_files = sorted(f for f in os.listdir('./raw/') if f.endswith('.txt'))
print(f'Patient files found: {len(training_files)}')

x = []
static = []
y = []
name = []
mask = []
static_mask = []

for file in training_files:
    x_one = []
    mask_one = []
    static_one = np.zeros(STATIC_NUM)
    static_mask_one = np.ones(STATIC_NUM)

    with open(os.path.join('raw', file), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        x_row = np.zeros(DYNAMIC_NUM)
        mask_row = np.zeros(DYNAMIC_NUM)
        cur_time = 0
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if row[1] == 'RecordID':
                continue
            time = int(row[0][:2])
            while time > cur_time:
                cur_time += 1
                x_one.append(
                    np.divide(x_row, mask_row,
                              out=np.zeros_like(x_row), where=mask_row != 0)
                )
                mask_one.append(mask_row.copy())
                x_row = np.zeros(DYNAMIC_NUM)
                mask_row = np.zeros(DYNAMIC_NUM)
            if row[1] in STATIC_DICT:
                if float(row[2]) == -1:
                    static_mask_one[STATIC_DICT[row[1]]] = 0
                static_one[STATIC_DICT[row[1]]] = float(row[2])
            elif row[1] in DYNAMIC_DICT:
                if float(row[2]) != -1:
                    x_row[DYNAMIC_DICT[row[1]]] += float(row[2])
                    mask_row[DYNAMIC_DICT[row[1]]] += 1

    if len(x_one) >= 1:
        x.append(x_one)
        mask.append(mask_one)
        static.append(static_one)
        y.append(y_dict[file.split('.')[0]])
        name.append(int(file.split('.')[0]))
        static_mask.append(static_mask_one)
    else:
        print(f'Skipped (no time steps): {file}')

print(f'Kept patients: {len(x)}')

# ---------------------------------------------------------------------------
# Normalization statistics (computed on observed values only)
# ---------------------------------------------------------------------------
static_masked = np.ma.masked_array(static, np.array(static_mask) == 0)
static_mean = np.mean(static_masked, 0)
static_std  = np.std(static_masked, 0)

x_flat    = [row for patient in x for row in patient]
mask_flat = [row for patient in mask for row in patient]
x_masked  = np.ma.masked_array(x_flat, np.array(mask_flat) == 0)
x_mean = np.mean(x_masked, 0)
x_std  = np.std(x_masked, 0)

# MechVent (index 19) is binary with std=0: avoid divide-by-zero
print(f'MechVent mean={x_mean[19]:.4f}, std={x_std[19]:.4f}')
x_mean[19] = 0.0
x_std[19]  = 1.0

# ---------------------------------------------------------------------------
# Build normalized arrays
# ---------------------------------------------------------------------------
x_normalize      = []
static_normalize = []
mask_normalize   = []

for i in range(len(x)):
    # Static: zero-fill missing, no z-score (matches original notebook)
    static_normalize.append(
        np.where(static_mask[i] > 0, static[i], 0).tolist()
    )
    x_one    = []
    mask_one = []
    for j in range(len(x[i])):
        x_one.append(np.where(mask[i][j] > 0, x[i][j], 0).tolist())
        mask_one.append(np.where(mask[i][j] > 0, 1, 0).tolist())
    x_normalize.append(x_one)
    mask_normalize.append(mask_one)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))

pickle.dump(
    (x_normalize, y, static_normalize, mask_normalize, name),
    open(os.path.join(out_dir, 'data_normalized.pkl'), 'wb')
)
stat_dict = {
    'x_mean': x_mean, 'x_std': x_std,
    'static_mean': static_mean, 'static_std': static_std
}
pickle.dump(stat_dict, open(os.path.join(out_dir, 'stat.pkl'), 'wb'))

cnt  = sum(sum(row) for patient in mask_normalize for row in patient)
cnt1 = sum(len(row) for patient in mask_normalize for row in patient)
pos  = sum(y)
print(f'Observed Rate:  {cnt / cnt1:.4f}')
print(f'Mortality Rate: {pos / len(y):.4f}  ({pos}/{len(y)})')
print('Done. Saved: data_normalized.pkl, stat.pkl')
