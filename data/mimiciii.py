import pickle
import random
import numpy as np
from data.dataloader import CustomDataset
from data.feature_registry import get_feature_names, validate_registry

# MIMIC-III YerevaNN benchmark 17 variables (order matches preprocess_mimiciii.py output)
FEATURE_NAMES_MIMIC = get_feature_names('mimic')


class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None


def _flatten_scalar_label(y):
    if isinstance(y, (list, tuple, np.ndarray)):
        return float(y[0])
    return float(y)


def _detect_los_label_unit(y):
    """Infer whether length-of-stay labels are stored in hours or days."""
    flat = np.array([_flatten_scalar_label(v) for v in y], dtype=np.float32)
    if len(flat) == 0:
        return 'hours'
    return 'hours' if np.percentile(flat, 95) > 30.0 else 'days'


def _build_sample(x, y, mask, idx):
    x_len = len(x[idx])
    return {
        "x": x[idx],
        "labels": y[idx],
        "lens": x_len,
        "mask": mask[idx],
        "time": np.arange(1, x_len + 1, dtype=np.float32),
    }


def _split_by_sizes(x, y, mask, split_sizes):
    n_train, n_val = split_sizes[0], split_sizes[1]
    splits = [
        (0, n_train),
        (n_train, n_train + n_val),
        (n_train + n_val, len(x)),
    ]
    result = []
    for start, end in splits:
        data = []
        for idx in range(start, end):
            data.append(_build_sample(x, y, mask, idx))
        result.append(data)
    return result, [list(range(start, end)) for start, end in splits]


def _split_random(x, y, mask, training_ratio=0.8, split_seed=42):
    patient_index = list(range(len(x)))
    random.Random(split_seed).shuffle(patient_index)
    train_num = int(len(x) * training_ratio)
    val_num = int(len(x) * ((1 - training_ratio) / 2))
    splits_idx = [
        patient_index[:train_num],
        patient_index[train_num:train_num + val_num],
        patient_index[train_num + val_num:],
    ]
    result = []
    for indices in splits_idx:
        data = []
        for idx in indices:
            data.append(_build_sample(x, y, mask, idx))
        result.append(data)
    return result, splits_idx


def load_mimic_iii_mortality(training_ratio=0.8, split_seed=42):
    loaded = pickle.load(open('./data/MIMIC-III/mortality_normalized.pkl', 'rb'))
    if len(loaded) == 5:
        x, y, mask, name, split_sizes = loaded
        (train_data, val_data, test_data), indices_list = _split_by_sizes(x, y, mask, split_sizes)
    else:
        x, y, mask, name = loaded
        (train_data, val_data, test_data), indices_list = _split_random(
            x, y, mask, training_ratio, split_seed
        )
    validate_registry('mimic_mortality', len(x[0][0]))

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)
    for ds, indices in zip((train_dataset, val_dataset, test_dataset), indices_list):
        ds.feature_names = FEATURE_NAMES_MIMIC
        ds.patient_ids = tuple(name[idx] for idx in indices)
    return train_dataset, val_dataset, test_dataset


def load_mimic_iii_phenotyping(training_ratio=0.8, split_seed=42):
    loaded = pickle.load(open('./data/MIMIC-III/phenotyping_normalized.pkl', 'rb'))
    if len(loaded) == 5:
        x, y, mask, name, split_sizes = loaded
        indices_list = [
            range(0, split_sizes[0]),
            range(split_sizes[0], split_sizes[0] + split_sizes[1]),
            range(split_sizes[0] + split_sizes[1], len(x)),
        ]
    else:
        x, y, mask, name = loaded
        patient_index = list(range(len(x)))
        random.Random(split_seed).shuffle(patient_index)
        train_num = int(len(x) * training_ratio)
        val_num = int(len(x) * ((1 - training_ratio) / 2))
        indices_list = [
            patient_index[:train_num],
            patient_index[train_num:train_num + val_num],
            patient_index[train_num + val_num:],
        ]

    datasets = []
    for indices in indices_list:
        data = []
        for idx in indices:
            x_len = len(x[idx])
            data.append({
                "x": x[idx],
                "labels": [float(_) for _ in y[idx]],
                "lens": x_len,
                "mask": mask[idx],
                "time": np.arange(1, x_len + 1, dtype=np.float32),
            })
        datasets.append(CustomDataset(data))

    for ds, indices in zip(datasets, indices_list):
        ds.feature_names = FEATURE_NAMES_MIMIC
        ds.patient_ids = tuple(name[idx] for idx in indices)
    validate_registry('mimic_phenotyping', len(x[0][0]))
    return datasets[0], datasets[1], datasets[2]


def load_mimic_iii_decompensation(training_ratio=0.8, split_seed=42):
    loaded = pickle.load(open('./data/MIMIC-III/decompensation_normalized.pkl', 'rb'))
    if len(loaded) == 5:
        x, y, mask, name, split_sizes = loaded
        (train_data, val_data, test_data), indices_list = _split_by_sizes(x, y, mask, split_sizes)
    else:
        x, y, mask, name = loaded
        (train_data, val_data, test_data), indices_list = _split_random(
            x, y, mask, training_ratio, split_seed
        )
    validate_registry('mimic_decompensation', len(x[0][0]))

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)
    for ds, indices in zip((train_dataset, val_dataset, test_dataset), indices_list):
        ds.feature_names = FEATURE_NAMES_MIMIC
        ds.patient_ids = tuple(name[idx] for idx in indices)
    return train_dataset, val_dataset, test_dataset


def load_mimic_iii_lengthofstay(training_ratio=0.8, task='classification', label_unit='auto',
                                 split_seed=42):
    loaded = pickle.load(open('./data/MIMIC-III/lengthofstay_normalized.pkl', 'rb'))
    if len(loaded) == 5:
        x, y, mask, name, split_sizes = loaded
        indices_list = [
            range(0, split_sizes[0]),
            range(split_sizes[0], split_sizes[0] + split_sizes[1]),
            range(split_sizes[0] + split_sizes[1], len(x)),
        ]
    else:
        x, y, mask, name = loaded
        patient_index = list(range(len(x)))
        random.Random(split_seed).shuffle(patient_index)
        train_num = int(len(x) * training_ratio)
        val_num = int(len(x) * ((1 - training_ratio) / 2))
        indices_list = [
            patient_index[:train_num],
            patient_index[train_num:train_num + val_num],
            patient_index[train_num + val_num:],
        ]

    # pkl 中 y 以小时存储（preprocess_mimiciii.py 存入的是 hours）
    # get_bin_custom 内部 bins 已乘以 24（day->hour），直接传入 hours 即可，不需再 *24
    if label_unit == 'auto':
        label_unit = _detect_los_label_unit(y)

    y_flat = [_flatten_scalar_label(yi) for yi in y]
    if task == 'classification':
        y_hours = y_flat if label_unit == 'hours' else [yi * 24.0 for yi in y_flat]
        y = [get_bin_custom(yi, 10) for yi in y_hours]
    elif task == 'regression':
        y_days = y_flat if label_unit == 'days' else [yi / 24.0 for yi in y_flat]
        y = [[yi] for yi in y_days]
    else:
        raise ValueError(f"Unsupported lengthofstay task: {task}")

    datasets = []
    for indices in indices_list:
        data = []
        for idx in indices:
            x_len = len(x[idx])
            data.append({
                "x": x[idx],
                "labels": y[idx],
                "lens": x_len,
                "mask": mask[idx],
                "time": np.arange(1, x_len + 1, dtype=np.float32),
            })
        datasets.append(CustomDataset(data))

    for ds, indices in zip(datasets, indices_list):
        ds.feature_names = FEATURE_NAMES_MIMIC
        ds.patient_ids = tuple(name[idx] for idx in indices)
    validate_registry('mimic_lengthofstay', len(x[0][0]))
    return datasets[0], datasets[1], datasets[2]
