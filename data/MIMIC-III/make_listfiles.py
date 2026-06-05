import os
import sys
import random

random.seed(42)

base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', '..', '..', 'mimic3-benchmarks', 'data')
base = os.path.normpath(base)

# decompensation and length-of-stay have ~3M rows each -> OOM
# cap them to a manageable size (still large enough for meaningful experiments)
# NOTE: decompensation uses SMART's custom one-per-stay format (see below),
#       so no capping is needed there.
MAX_SAMPLES = {
    'in-hospital-mortality': None,
    'phenotyping':           None,
    'decompensation':        None,   # handled separately via decompensation_smart/
    'length-of-stay':        200000,
}

tasks = list(MAX_SAMPLES.keys())

for task in tasks:
    # ------------------------------------------------------------------
    # Decompensation: use SMART's one-per-stay format (period_length<=24)
    # rather than the standard benchmark's sliding-window format
    # (period_length up to 2803h).  The standard benchmark creates a label
    # at hour t but our preprocessing only keeps the first 24 h of data,
    # so samples with t >> 24 have completely mismatched features/labels.
    #
    # Prerequisites: run create_decompensation.py first:
    #   cd mimic3-benchmarks
    #   python ../SMART/data/MIMIC-III/create_decompensation.py \
    #       data/root  data/decompensation_smart
    # ------------------------------------------------------------------
    if task == 'decompensation':
        task_dir   = os.path.join(base, 'decompensation_smart')
        train_src  = os.path.join(task_dir, 'train', 'listfile.csv')
        test_src   = os.path.join(task_dir, 'test',  'listfile.csv')
        if not os.path.exists(train_src):
            print(f"SKIP decompensation: {train_src} not found.\n"
                  f"  Run create_decompensation.py first (see comment above).")
            continue
        with open(train_src, 'r') as f:
            lines = f.readlines()
        header = lines[0]
        data   = lines[1:]
        random.shuffle(data)
        # no capping: SMART format has only ~29K train stays
        split = int(len(data) * 0.9)
        train_data = data[:split]
        val_data   = data[split:]
        with open(os.path.join(task_dir, 'train_listfile.csv'), 'w') as f:
            f.write(header); f.writelines(train_data)
        with open(os.path.join(task_dir, 'val_listfile.csv'), 'w') as f:
            f.write(header); f.writelines(val_data)
        with open(test_src, 'r') as f:
            test_lines = f.readlines()
        with open(os.path.join(task_dir, 'test_listfile.csv'), 'w') as f:
            f.writelines(test_lines)
        print(f"decompensation (SMART fmt): train={len(train_data)}, "
              f"val={len(val_data)}, test={len(test_lines)-1}")
        continue

    task_dir = os.path.join(base, task)
    train_src = os.path.join(task_dir, 'train', 'listfile.csv')
    test_src  = os.path.join(task_dir, 'test',  'listfile.csv')

    if not os.path.exists(train_src):
        print(f"SKIP {task}: {train_src} not found")
        continue

    with open(train_src, 'r') as f:
        lines = f.readlines()
    header = lines[0]
    data   = lines[1:]
    random.shuffle(data)

    # subsample if needed
    cap = MAX_SAMPLES[task]
    if cap is not None and len(data) > cap:
        data = data[:cap]

    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data   = data[split:]

    with open(os.path.join(task_dir, 'train_listfile.csv'), 'w') as f:
        f.write(header)
        f.writelines(train_data)
    with open(os.path.join(task_dir, 'val_listfile.csv'), 'w') as f:
        f.write(header)
        f.writelines(val_data)

    with open(test_src, 'r') as f:
        test_lines = f.readlines()
    # also cap test set proportionally
    if cap is not None:
        test_ratio = cap / (len(lines) - 1)
        test_cap = max(1000, int((len(test_lines) - 1) * test_ratio))
        test_data = test_lines[1:test_cap + 1]
    else:
        test_data = test_lines[1:]
    with open(os.path.join(task_dir, 'test_listfile.csv'), 'w') as f:
        f.write(test_lines[0])
        f.writelines(test_data)

    print(f"{task}: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

print("Done")
