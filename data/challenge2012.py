import pickle
import random
import numpy as np
from data.dataloader import CustomDataset
from data.feature_registry import get_feature_names, validate_registry


FEATURE_NAMES_C12 = get_feature_names('c12')

_DEFAULT_SPLIT_SEED = 42
_DEFAULT_DATA_PATH = './data/Challenge2012/data_normalized.pkl'


def load_challenge_2012(
        training_ratio=0.8,
        split_seed=_DEFAULT_SPLIT_SEED,
        data_path=_DEFAULT_DATA_PATH):
    with open(data_path, 'rb') as handle:
        x, y, static, mask, name = pickle.load(handle)
    validate_registry('c12', len(x[0][0]))
    patient_index = list(range(len(x)))
    random.Random(split_seed).shuffle(patient_index)

    train_num = int(len(x) * training_ratio)
    val_num = int(len(x) * ((1 - training_ratio) / 2))

    splits_idx = [
        patient_index[:train_num],
        patient_index[train_num:train_num + val_num],
        patient_index[train_num + val_num:],
    ]
    datasets = []
    for indices in splits_idx:
        data = []
        for idx in indices:
            x_len = len(x[idx])
            data.append({
                "x": x[idx],
                "labels": y[idx],
                "lens": x_len,
                "mask": mask[idx],
                "static": static[idx],
                "time": np.arange(1, x_len + 1, dtype=np.float32),
            })
        ds = CustomDataset(data)
        ds.feature_names = FEATURE_NAMES_C12
        ds.patient_ids = tuple(name[idx] for idx in indices)
        ds.split_seed = split_seed
        datasets.append(ds)
    return datasets[0], datasets[1], datasets[2]
