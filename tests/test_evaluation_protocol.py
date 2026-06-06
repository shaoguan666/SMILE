import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

import main_finetune
from experiments.bibm_smile.aggregate_results import parse_log


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.mnar_bias_scale = nn.Parameter(torch.zeros(4, 3, 3))
        self.proj = nn.Linear(2, 2)


class EvaluationProtocolTest(unittest.TestCase):
    def test_validation_thresholds_are_selected_per_metric(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.05, 0.15, 0.4, 0.35, 0.7, 0.9])
        preds = np.column_stack([1 - probs, probs])

        f1_threshold, best_f1 = main_finetune._best_f1_threshold(labels, probs)
        minpse_threshold, best_minpse = main_finetune._best_minpse_threshold(labels, probs)

        self.assertAlmostEqual(
            main_finetune._binary_metrics_at_threshold(labels, preds, f1_threshold)["f1"],
            best_f1,
        )
        self.assertAlmostEqual(
            main_finetune._binary_metrics_at_threshold(labels, preds, minpse_threshold)["minpse"],
            best_minpse,
        )

    def test_validation_aggregation_never_falls_back_to_benchmark_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "training.log"
            log_path.write_text(
                "AUC of ROC = 0.8000\nAUC of PRC = 0.7000\n"
                "min(+P, Se) = 0.6000\nf1_score = 0.6500\n",
                encoding="utf-8",
            )
            payload = {
                "test_metrics": {
                    "auroc": 0.81,
                    "auprc": 0.71,
                    "minpse": 0.61,
                    "f1_score": 0.66,
                },
                "validation_threshold_metrics": {"threshold": 0.3, "f1": 0.55},
            }
            (root / "eval_results.json").write_text(json.dumps(payload), encoding="utf-8")

            metrics = parse_log(log_path, "c12", "validation")

            self.assertEqual(metrics["auroc"], 81.0)
            self.assertEqual(metrics["auprc"], 71.0)
            self.assertNotIn("f1", metrics)
            self.assertNotIn("minpse", metrics)

    def test_validation_aggregation_without_eval_json_omits_threshold_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "training.log"
            log_path.write_text(
                "AUC of ROC = 0.8000\nAUC of PRC = 0.7000\n"
                "min(+P, Se) = 0.6000\nf1_score = 0.6500\n",
                encoding="utf-8",
            )

            metrics = parse_log(log_path, "c12", "validation")

            self.assertEqual(metrics, {"auroc": 80.0, "auprc": 70.0})

    def test_validation_aggregation_accepts_only_v2_threshold_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "training.log"
            log_path.write_text("", encoding="utf-8")
            payload = {
                "test_metrics": {"auroc": 0.81, "auprc": 0.71, "minpse": 0.61, "f1_score": 0.66},
                "validation_threshold_metrics": {
                    "protocol": "validation_selected_per_metric_v2",
                    "f1": 0.55,
                    "minpse": 0.44,
                },
            }
            (root / "eval_results.json").write_text(json.dumps(payload), encoding="utf-8")

            metrics = parse_log(log_path, "c12", "validation")

            self.assertAlmostEqual(metrics["f1"], 55.0)
            self.assertAlmostEqual(metrics["minpse"], 44.0)

    def test_legacy_comiss_shape_is_ignored_only_for_disabled_path(self):
        source = ToyEncoder().state_dict()
        source["attn.mnar_bias_scale"] = torch.ones(4)
        source = {"module." + key: value for key, value in source.items()}

        enabled_model = ToyEncoder()
        with self.assertRaises(RuntimeError):
            main_finetune._load_model_state_dict(enabled_model, source)

        disabled_model = ToyEncoder()
        ignored = main_finetune._load_model_state_dict(
            disabled_model,
            source,
            allow_unused_mnar_bias_mismatch=True,
        )
        self.assertEqual(ignored, ["attn.mnar_bias_scale"])
        self.assertEqual(tuple(disabled_model.attn.mnar_bias_scale.shape), (4, 3, 3))


if __name__ == "__main__":
    unittest.main()
