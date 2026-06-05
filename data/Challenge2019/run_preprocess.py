"""
Preprocess Challenge 2019 (Sepsis) data from NeuralCDE training_A/training_B PSV files.
Output: data_normalized.pkl, stat.pkl
"""
import numpy as np
import os
import pickle
import csv

# NeuralCDE sepsis data: training_A + training_B (.psv, pipe-separated)
DATA_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'NeuralCDE-master', 'experiments', 'datasets', 'data', 'sepsis'
))
TRAINING_DIRS = [os.path.join(DATA_ROOT, 'training_A'), os.path.join(DATA_ROOT, 'training_B')]

# Collect (dir_path, filename) pairs from both directories
training_file_pairs = []
for d in TRAINING_DIRS:
    for f in os.listdir(d):
        if f.endswith('.psv'):
            training_file_pairs.append((d, f))

# Remove known bad file (no dynamic data)
training_file_pairs = [(d, f) for d, f in training_file_pairs if f != 'p108796.psv']
training_file_pairs.sort(key=lambda pair: pair[1])
training_size = len(training_file_pairs)
print(f'Total files: {training_size}')
print(f'First file: {training_file_pairs[0]}')

# Column layout: [0-33] dynamic (HR..Platelets), [34-38] static (Age,Gender,Unit1,Unit2,HospAdmTime),
#                [-2] ICULOS, [-1] SepsisLabel
static_num = 5
dynamic_num = 34
max_length = 60

x = []
static = []
y = []
name = []
mask = []
static_mask = []

for idx, (dir_path, file) in enumerate(training_file_pairs):
    if idx % 5000 == 0:
        print(f'  Processing {idx}/{training_size}...')
    x_one = []
    mask_one = []
    static_one = np.zeros((static_num))
    static_mask_one = np.ones((static_num))
    with open(os.path.join(dir_path, file), mode='r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        los = None
        for row in reader:
            if row[0] == 'HR':
                continue
            if los is None:
                los = int(float(row[-2]))
            x_row = np.zeros((dynamic_num))
            mask_row = np.zeros((dynamic_num))
            for i in range(dynamic_num):
                if row[i] not in ('', 'NaN', 'nan'):
                    x_row[i] = float(row[i])
                    mask_row[i] = 1
            x_one.append(x_row)
            mask_one.append(mask_row)
        for i in range(static_num):
            if row[i + dynamic_num] not in ('', 'NaN', 'nan'):
                static_one[i] = float(row[i + dynamic_num])
            else:
                static_mask_one[i] = 0
        assert int(float(row[-2])) - los + 1 == len(x_one), \
            f'{file}: ICULOS span mismatch, got {len(x_one)} rows'
    if len(x_one) >= 8:
        x.append(x_one[:max_length])
        mask.append(mask_one[:max_length])
        static.append(static_one)
        y.append(int(row[-1]))
        name.append(file.split('.')[0])
        static_mask.append(static_mask_one)

print(f'Kept samples (>= 8 timesteps): {len(x)}')
print(f'Sepsis rate: {sum(y) / len(y):.4f}')
print(f'Observed Rate (raw): computing...')

cnt = 0
cnt1 = 0
for i in range(len(mask)):
    for j in range(len(mask[i])):
        cnt += sum(mask[i][j])
        cnt1 += len(mask[i][j])
print(f'Observed Rate: {cnt / cnt1:.4f}')

# Compute normalization stats
print('Computing normalization statistics...')
static_arr = np.array(static)
static_mask_arr = np.array(static_mask)
static_masked = np.ma.masked_array(static_arr, static_mask_arr == 0)
static_mean = np.mean(static_masked, 0)
static_std = np.std(static_masked, 0)

x_flat = []
mask_flat = []
for i in range(len(x)):
    x_flat += x[i]
    mask_flat += mask[i]
x_masked = np.ma.masked_array(x_flat, np.array(mask_flat) == 0)
x_mean = np.mean(x_masked, 0)
x_std = np.std(x_masked, 0)

# Normalize
x_normalize = []
static_normalize = []
mask_normalize = []
for i in range(len(x)):
    static_normalize.append(np.where(static_mask[i] > 0,
                                     (static[i] - static_mean) / static_std, 0).tolist())
    x_one = []
    mask_one = []
    for j in range(len(x[i])):
        x_one.append(np.where(mask[i][j] > 0,
                               (x[i][j] - x_mean) / x_std, 0).tolist())
        mask_one.append(np.where(mask[i][j] > 0, 1, 0).tolist())
    x_normalize.append(x_one)
    mask_normalize.append(mask_one)

index_array = list(range(len(x_normalize)))

x_random = [x_normalize[i] for i in index_array]
y_random = [y[i] for i in index_array]
static_random = [static_normalize[i] for i in index_array]
mask_random = [mask_normalize[i] for i in index_array]
name_random = [name[i] for i in index_array]

# Save outputs in same directory as this script
out_dir = os.path.dirname(__file__)
pickle.dump((x_random, y_random, static_random, mask_random, name_random),
            open(os.path.join(out_dir, 'data_normalized.pkl'), 'wb'))
stat_dict = {'x_mean': x_mean, 'x_std': x_std,
             'static_mean': static_mean, 'static_std': static_std}
pickle.dump(stat_dict, open(os.path.join(out_dir, 'stat.pkl'), 'wb'))

print('Done. Saved: data_normalized.pkl, stat.pkl')
