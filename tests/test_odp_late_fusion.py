import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import main_finetune
from data.dataloader import CustomDataset, DeterministicShuffledHistoryDataset, collate_fn
from models.odp import (
    CausalObservationPolicyEncoder,
    NoCoMissClinicalEncoder,
    ODPLateFusionEncoder,
    ParameterMatchedAdapterEncoder,
    causal_policy_bce,
    masked_policy_mean,
)
from utils.odp_metrics import forecasting_metrics, make_forecast_records


def model_args(num_vars=4):
    return SimpleNamespace(
        input_dim=num_vars,
        d_model=8,
        num_class=2,
        dropout=0.0,
        max_len=6,
        e_layers=1,
        n_heads=2,
        time_dim=16,
        obs_density_window=5,
        policy_hidden_dim=64,
        policy_kernel_size=7,
        seed=42,
    )


def toy_batch(batch=2, time_steps=5, num_vars=4):
    torch.manual_seed(3)
    mask = torch.randint(0, 2, (batch, time_steps, num_vars)).float()
    return {
        "x": torch.randn(batch, time_steps, num_vars),
        "mask": mask.clone(),
        "original_mask": mask,
        "time": torch.arange(1, time_steps + 1).float().repeat(batch, 1),
        "lens": torch.tensor([time_steps, max(1, time_steps - 2)]),
    }


class PolicyCausalityTest(unittest.TestCase):
    def test_current_and_future_mask_cannot_change_present_or_past(self):
        encoder = CausalObservationPolicyEncoder(4, mode="full").eval()
        batch = toy_batch()
        reference = encoder(batch["original_mask"], batch["time"], batch["lens"])

        changed_current = batch["original_mask"].clone()
        changed_current[:, 2] = 1 - changed_current[:, 2]
        current = encoder(changed_current, batch["time"], batch["lens"])
        torch.testing.assert_close(reference["hidden"][:, 2], current["hidden"][:, 2])
        torch.testing.assert_close(reference["logits"][:, 2], current["logits"][:, 2])

        changed_future = batch["original_mask"].clone()
        changed_future[:, 3:] = 1 - changed_future[:, 3:]
        future = encoder(changed_future, batch["time"], batch["lens"])
        torch.testing.assert_close(reference["hidden"][:, :3], future["hidden"][:, :3])
        torch.testing.assert_close(reference["logits"][:, :3], future["logits"][:, :3])

    def test_policy_api_rejects_clinical_values(self):
        encoder = CausalObservationPolicyEncoder(4)
        batch = toy_batch()
        with self.assertRaises(TypeError):
            encoder(batch["original_mask"], batch["time"], batch["lens"], x=batch["x"])

    def test_padding_changes_do_not_affect_valid_outputs(self):
        encoder = CausalObservationPolicyEncoder(4).eval()
        batch = toy_batch()
        reference = encoder(batch["original_mask"], batch["time"], batch["lens"])
        changed_mask = batch["original_mask"].clone()
        changed_time = batch["time"].clone()
        changed_mask[1, 3:] = 99
        changed_time[1, 3:] = 999
        changed = encoder(changed_mask, changed_time, batch["lens"])
        torch.testing.assert_close(reference["hidden"][1, :3], changed["hidden"][1, :3])
        torch.testing.assert_close(reference["logits"][1, :3], changed["logits"][1, :3])


class LossPoolingMetricsTest(unittest.TestCase):
    def test_primary_loss_excludes_t0_padding_and_short_sequences(self):
        logits = torch.zeros(2, 4, 3, requires_grad=True)
        target = torch.zeros_like(logits)
        lens = torch.tensor([3, 1])
        base = causal_policy_bce(logits, target, lens)
        target[:, 0] = 1
        target[0, 3] = 1
        changed = causal_policy_bce(logits, target, lens)
        torch.testing.assert_close(base, changed)
        zero = causal_policy_bce(logits[:1, :1], target[:1, :1], torch.tensor([1]))
        self.assertTrue(torch.isfinite(zero))
        self.assertEqual(float(zero), 0.0)

    def test_pooling_excludes_t0_and_padding_without_nan(self):
        hidden = torch.zeros(2, 4, 2, 3)
        hidden[:, 0] = 100
        hidden[0, 1:3] = 2
        hidden[0, 3] = 999
        pooled = masked_policy_mean(hidden, torch.tensor([3, 1]))
        torch.testing.assert_close(pooled[0], torch.full((2, 3), 2.0))
        torch.testing.assert_close(pooled[1], torch.zeros(2, 3))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_metrics_primary_excludes_t0_and_undefined_variables(self):
        prob = torch.tensor([[[0.99, 0.2], [0.2, 0.1], [0.8, 0.1]]])
        target = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]]])
        records = make_forecast_records(prob, target, torch.tensor([3]), torch.tensor([7]))
        primary = forecasting_metrics(records)
        secondary = forecasting_metrics(records, include_t0=True)
        self.assertEqual(primary["scope"], "t>=1_primary")
        self.assertEqual(secondary["scope"], "t>=0_secondary")
        self.assertEqual(primary["eligible_variables"], 1)
        self.assertEqual(primary["total_variables"], 2)
        self.assertIsNone(primary["per_variable"][1]["auprc"])


class FusionAndGradientTest(unittest.TestCase):
    def test_zero_alpha_matches_density_and_only_changes_cls(self):
        args = model_args()
        density = NoCoMissClinicalEncoder(args, use_density=True).eval()
        odp = ODPLateFusionEncoder(args).eval()
        odp.clinical.load_state_dict(density.state_dict())
        batch = toy_batch()
        clinical = density(**batch)
        fused = odp(**batch)
        torch.testing.assert_close(clinical, fused)
        self.assertFalse(torch.count_nonzero(odp.policy_projection.weight) == 0)
        with torch.no_grad():
            odp.alpha.fill_(0.5)
        changed = odp(**batch)
        torch.testing.assert_close(changed[:, :, 1:], clinical[:, :, 1:])

    def test_expected_gradient_topology(self):
        args = model_args()
        model = ODPLateFusionEncoder(args)
        batch = toy_batch()
        fused, policy = model(**batch, return_policy=True)
        task_loss = (fused[:, :, 0] * torch.randn_like(fused[:, :, 0])).sum()
        task_loss.backward(retain_graph=True)
        self.assertTrue(torch.isfinite(model.alpha.grad))
        self.assertNotEqual(float(model.alpha.grad), 0.0)
        self.assertTrue(model.policy_projection.weight.grad is None or
                        torch.count_nonzero(model.policy_projection.weight.grad) == 0)
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0
                            for p in model.policy_encoder.parameters()))

        model.zero_grad(set_to_none=True)
        policy_loss = causal_policy_bce(policy["logits"], batch["original_mask"], batch["lens"])
        policy_loss.backward(retain_graph=True)
        policy_grads = [p.grad for p in model.policy_encoder.parameters() if p.grad is not None]
        self.assertTrue(any(torch.isfinite(g).all() and torch.count_nonzero(g) for g in policy_grads))

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.alpha.fill_(0.5)
        fused = model(**batch)
        fused[:, :, 0].square().sum().backward()
        self.assertTrue(torch.count_nonzero(model.policy_projection.weight.grad))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)
                            for p in model.policy_encoder.parameters()))

    def test_alpha_optimizer_step_and_parameter_matched_gradients(self):
        args = model_args()
        model = ODPLateFusionEncoder(args)
        batch = toy_batch()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        fused = model(**batch)
        fused[:, :, 0].sum().backward()
        optimizer.step()
        self.assertNotEqual(float(model.alpha), 0.0)

        adapter = ParameterMatchedAdapterEncoder(args)
        with torch.no_grad():
            adapter.beta.fill_(0.5)
        adapter(**batch)[:, :, 0].square().sum().backward()
        self.assertTrue(torch.count_nonzero(adapter.beta.grad))
        self.assertTrue(torch.count_nonzero(adapter.adapter_in.weight.grad))
        self.assertTrue(torch.count_nonzero(adapter.adapter_out.weight.grad))

    def test_parameter_match_within_two_percent_for_all_dataset_widths(self):
        for variables in (17, 34, 37):
            adapter = ParameterMatchedAdapterEncoder(model_args(variables))
            self.assertLessEqual(adapter.match_percent, 2.0)


class ShuffleCheckpointTest(unittest.TestCase):
    def _dataset(self, size=5, time_steps=4, variables=3):
        data = []
        for sample in range(size):
            mask = np.zeros((time_steps, variables), dtype=np.float32)
            mask[:] = np.arange(variables) + 10 * sample
            data.append({"x": mask.copy(), "mask": mask, "time": np.arange(time_steps),
                         "lens": time_steps, "labels": sample % 2})
        return CustomDataset(data)

    def test_shuffle_is_deranged_stable_and_preserves_own_history(self):
        base = self._dataset()
        a = DeterministicShuffledHistoryDataset(base, "validation", 42)
        b = DeterministicShuffledHistoryDataset(base, "validation", 42)
        for index in range(len(base)):
            item_a = a[index]
            item_b = b[index]
            np.testing.assert_array_equal(item_a["policy_history_mask"], item_b["policy_history_mask"])
            for variable in range(3):
                self.assertNotEqual(a.source_index(index, variable), index)
                np.testing.assert_array_equal(
                    item_a["policy_history_mask"][:, variable, variable],
                    np.asarray(base[index]["mask"])[:, variable],
                )
        batches = [collate_fn([a[i] for i in order]) for order in ([0, 1], [1, 0])]
        by_id = {}
        for batch in batches:
            for row, sample_id in enumerate(batch["sample_id"].tolist()):
                current = batch["policy_history_mask"][row]
                if sample_id in by_id:
                    torch.testing.assert_close(by_id[sample_id], current)
                by_id[sample_id] = current

    def test_variable_length_shuffled_histories_are_padded(self):
        base = self._dataset(size=3)
        base.data[1]['x'] = base.data[1]['x'][:2]
        base.data[1]['mask'] = base.data[1]['mask'][:2]
        base.data[1]['time'] = base.data[1]['time'][:2]
        base.data[1]['lens'] = 2
        wrapped = DeterministicShuffledHistoryDataset(base, 'validation', 42)
        batch = collate_fn([wrapped[0], wrapped[1]])
        self.assertEqual(tuple(batch['policy_history_mask'].shape), (2, 4, 3, 3))
        self.assertTrue(torch.count_nonzero(batch['policy_history_mask'][1, 2:]) == 0)

    def test_ddp_prefix_checkpoint_round_trip_restores_odp_metadata(self):
        args = model_args()
        source = ODPLateFusionEncoder(args, shuffled=True)
        with torch.no_grad():
            source.alpha.fill_(0.25)
            source.shuffle_seed.fill_(123)
        prefixed = {"module." + key: value for key, value in source.state_dict().items()}
        target = ODPLateFusionEncoder(args, shuffled=True)
        main_finetune._load_model_state_dict(target, prefixed)
        self.assertEqual(float(target.alpha), 0.25)
        self.assertEqual(int(target.shuffle_seed), 123)
        torch.testing.assert_close(target.policy_projection.weight, source.policy_projection.weight)


if __name__ == "__main__":
    unittest.main()
