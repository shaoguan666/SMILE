import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import hashlib
import numpy as np
from typing import Any, Dict, List


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # A split-stable sample identifier is required for DDP de-duplication
        # and deterministic shuffled-history controls.  It is deliberately the
        # dataset index, never the current batch index.
        sample = dict(self.data[idx])
        sample.setdefault("sample_id", idx)
        return sample

    def dropout_data(self, drop_rate=0.1):
        for i in range(len(self.data)):
            for j in range(len(self.data[i]['x'])):
                for k in range(len(self.data[i]['x'][j])):
                    if self.data[i]['mask'][j][k] == 1 and random.random() < drop_rate:
                        self.data[i]['x'][j][k] = 0
                        self.data[i]['mask'][j][k] = 0


def collate_fn(features: List[Dict[str, Any]]):
    batch = {}
    for key in features[0].keys():
        if key in ["x", "mask", "time", "policy_history_mask"]:
            batch[key] = pad_sequence([torch.tensor(patient[key]) for patient in features], True)
        else:
            batch[key] = torch.tensor([patient[key] for patient in features])
    return batch


class DeterministicShuffledHistoryDataset(Dataset):
    """Attach patient-shuffled cross-variable mask histories to a split.

    Each source variable uses a deterministic cyclic derangement derived from
    ``split``, ``seed``, ``epoch`` and the source variable id.  Consequently the
    mapping is independent of batch size/order, DDP rank and sampler padding.
    The target variable's own history is always copied from the target patient.
    Validation/test mappings are fixed; training may change once per epoch.
    """

    def __init__(self, dataset, split, seed, train=False):
        self.dataset = dataset
        self.data = dataset.data
        self.split = str(split)
        self.seed = int(seed)
        self.train = bool(train)
        self.epoch = 0
        self.feature_names = getattr(dataset, "feature_names", None)
        self.patient_ids = getattr(dataset, "patient_ids", None)

    def __len__(self):
        return len(self.dataset)

    def dropout_data(self, drop_rate=0.1):
        return self.dataset.dropout_data(drop_rate)

    def set_epoch(self, epoch):
        self.epoch = int(epoch) if self.train else 0

    def _offset(self, variable_id):
        n = len(self)
        if n <= 1:
            return 0
        epoch = self.epoch if self.train else 0
        payload = f"{self.split}|{self.seed}|{epoch}|{int(variable_id)}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return 1 + value % (n - 1)

    def source_index(self, sample_index, variable_id):
        if len(self) <= 1:
            return int(sample_index)
        return (int(sample_index) + self._offset(variable_id)) % len(self)

    def __getitem__(self, idx):
        target = dict(self.dataset[idx])
        target.setdefault("sample_id", idx)
        mask = np.asarray(target["mask"], dtype=np.float32)
        if mask.ndim != 2:
            raise ValueError("mask must have shape (T, V)")
        time_steps, num_vars = mask.shape
        history = np.zeros((time_steps, num_vars, num_vars), dtype=np.float32)

        source_masks = {}
        for source_var in range(num_vars):
            source_idx = self.source_index(idx, source_var)
            source_masks[source_var] = np.asarray(
                self.dataset[source_idx]["mask"], dtype=np.float32
            )

        for target_var in range(num_vars):
            for source_var in range(num_vars):
                if source_var == target_var:
                    history[:, target_var, source_var] = mask[:, source_var]
                else:
                    source = source_masks[source_var]
                    usable = min(time_steps, source.shape[0])
                    history[:usable, target_var, source_var] = source[:usable, source_var]

        target["policy_history_mask"] = history
        return target
