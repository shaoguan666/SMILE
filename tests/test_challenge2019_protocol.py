import pickle
import random
import tempfile
import unittest
from pathlib import Path

from data.challenge2019 import C19_PROTOCOL, load_challenge_2019


def _write_c19_fixture(path, n=20):
    x = [[[float(i)] * 34] for i in range(n)]
    y = [i % 2 for i in range(n)]
    static = [[0.0] * 5 for _ in range(n)]
    mask = [[[1.0] * 34] for _ in range(n)]
    names = [f"p{i:06d}" for i in range(n)]
    with path.open("wb") as handle:
        pickle.dump((x, y, static, mask, names), handle)
    return names


class Challenge2019ProtocolTest(unittest.TestCase):
    def test_c19_split_matches_smart_run_seed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data_normalized.pkl"
            names = _write_c19_fixture(data_path)

            train, val, test = load_challenge_2019(
                split_seed=1, data_path=str(data_path))

        expected = list(range(len(names)))
        random.Random(1).shuffle(expected)
        train_num = int(len(names) * 0.8)
        val_num = int(len(names) * ((1 - 0.8) / 2))
        self.assertEqual(
            train.patient_ids, tuple(names[i] for i in expected[:train_num]))
        self.assertEqual(
            val.patient_ids,
            tuple(names[i] for i in expected[train_num:train_num + val_num]))
        self.assertEqual(
            test.patient_ids,
            tuple(names[i] for i in expected[train_num + val_num:]))
        self.assertEqual(train.split_seed, 1)
        self.assertEqual(train.protocol, C19_PROTOCOL)

    def test_c19_run_seeds_change_split_without_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data_normalized.pkl"
            _write_c19_fixture(data_path)

            splits_seed1 = load_challenge_2019(
                split_seed=1, data_path=str(data_path))
            splits_seed42 = load_challenge_2019(
                split_seed=42, data_path=str(data_path))

        self.assertNotEqual(
            splits_seed1[0].patient_ids, splits_seed42[0].patient_ids)
        for splits in (splits_seed1, splits_seed42):
            id_sets = [set(dataset.patient_ids) for dataset in splits]
            self.assertTrue(id_sets[0].isdisjoint(id_sets[1]))
            self.assertTrue(id_sets[0].isdisjoint(id_sets[2]))
            self.assertTrue(id_sets[1].isdisjoint(id_sets[2]))


if __name__ == "__main__":
    unittest.main()
