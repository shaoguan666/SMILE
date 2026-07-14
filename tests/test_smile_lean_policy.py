import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import run_all_experiments
from models.smart import (
    CausalD2PolicyEncoder,
    EmbeddingDecoder,
    SMILELeanEncoder,
    SMILELeanPolicyEncoder,
    causal_policy_bce,
)


ROOT = Path(__file__).resolve().parents[1]


def make_args(num_vars=5, max_len=6, conditioning='hidden'):
    return SimpleNamespace(
        input_dim=num_vars,
        d_model=8,
        n_heads=2,
        e_layers=1,
        dropout=0.0,
        max_len=max_len,
        time_dim=4,
        obs_density_window=5,
        policy_conditioning=conditioning,
        policy_hidden_dim=8,
        policy_kernel_size=3,
        abl_no_density=False,
        abl_no_mnar_bias=False,
        abl_no_film=False,
        abl_no_time_mnar=False,
        abl_no_time_pe=False,
        abl_random_bias=False,
        abl_global_comiss=False,
    )


def synthetic_batch(batch_size=2, time_steps=6, num_vars=5):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(batch_size, time_steps, num_vars, generator=generator)
    mask = torch.randint(
        0, 2, (batch_size, time_steps, num_vars), generator=generator
    )
    clean = torch.randint(
        0, 2, (batch_size, time_steps, num_vars), generator=generator
    )
    lens = torch.tensor([time_steps, time_steps - 2][:batch_size])
    if batch_size > 2:
        lens = torch.full((batch_size,), time_steps, dtype=torch.long)
    time = torch.arange(time_steps).float().repeat(batch_size, 1)
    return x, mask, clean, lens, time


class CausalD2PolicyTest(unittest.TestCase):
    def test_cutoff_and_future_changes_do_not_affect_past_or_cutoff(self):
        model = CausalD2PolicyEncoder(5, hidden_dim=8, time_dim=4, kernel_size=3)
        model.eval()
        _, _, clean, lens, time = synthetic_batch()
        cutoff = 3
        changed = clean.clone()
        changed[:, cutoff:, :] = 1 - changed[:, cutoff:, :]

        out_a = model(clean, time=time, lens=lens)
        out_b = model(changed, time=time, lens=lens)

        torch.testing.assert_close(
            out_a['hidden'][:, :cutoff + 1],
            out_b['hidden'][:, :cutoff + 1],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            out_a['prob'][:, :cutoff + 1],
            out_b['prob'][:, :cutoff + 1],
            rtol=0,
            atol=0,
        )

    def test_current_mask_flip_does_not_affect_same_position(self):
        model = CausalD2PolicyEncoder(5, hidden_dim=8, time_dim=4, kernel_size=3)
        model.eval()
        _, _, clean, lens, time = synthetic_batch()
        position = 2
        changed = clean.clone()
        changed[:, position, :] = 1 - changed[:, position, :]

        out_a = model(clean, time=time, lens=lens)
        out_b = model(changed, time=time, lens=lens)
        torch.testing.assert_close(
            out_a['hidden'][:, position], out_b['hidden'][:, position],
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            out_a['prob'][:, position], out_b['prob'][:, position],
            rtol=0, atol=0,
        )

    def test_padding_is_ignored_by_outputs_and_policy_loss(self):
        model = CausalD2PolicyEncoder(5, hidden_dim=8, time_dim=4, kernel_size=3)
        model.eval()
        _, _, clean, lens, time = synthetic_batch()
        changed = clean.clone()
        changed[1, lens[1]:] = 1 - changed[1, lens[1]:]

        out_a = model(clean, time=time, lens=lens)
        out_b = model(changed, time=time, lens=lens)
        torch.testing.assert_close(out_a['hidden'], out_b['hidden'])
        torch.testing.assert_close(out_a['prob'], out_b['prob'])

        loss_a = causal_policy_bce(out_a['logits'], clean, lens)
        loss_b = causal_policy_bce(out_a['logits'], changed, lens)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        self.assertTrue(torch.equal(out_a['hidden'][1, lens[1]:],
                                    torch.zeros_like(out_a['hidden'][1, lens[1]:])))


class SMILELeanPolicyTest(unittest.TestCase):
    def test_zero_initialized_film_matches_lean_baseline(self):
        args = make_args()
        baseline = SMILELeanEncoder(args).eval()
        policy = SMILELeanPolicyEncoder(args).eval()
        incompatible = policy.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(
            key.startswith(('policy_encoder.', 'policy_film.'))
            for key in incompatible.missing_keys
        ))

        x, mask, clean, lens, time = synthetic_batch()
        with torch.no_grad():
            expected = baseline(
                x, lens, mask, time=time, original_mask=clean
            )
            actual = policy(
                x, lens, mask, time=time, original_mask=clean
            )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_supported_dataset_variable_shapes(self):
        for num_vars in (37, 34, 17):
            with self.subTest(num_vars=num_vars):
                args = make_args(num_vars=num_vars, max_len=4)
                model = SMILELeanPolicyEncoder(args).eval()
                x, mask, clean, lens, time = synthetic_batch(
                    time_steps=4, num_vars=num_vars
                )
                with torch.no_grad():
                    representation, policy = model(
                        x, lens, mask, time=time,
                        original_mask=clean, return_policy=True,
                    )
                self.assertEqual(representation.shape, (2, num_vars, 5, 8))
                self.assertEqual(policy['hidden'].shape, (2, 4, num_vars, 8))
                self.assertEqual(policy['prob'].shape, (2, 4, num_vars))

    def test_original_mask_none_falls_back_to_lean(self):
        args = make_args()
        baseline = SMILELeanEncoder(args).eval()
        policy = SMILELeanPolicyEncoder(args).eval()
        policy.load_state_dict(baseline.state_dict(), strict=False)
        x, mask, _, lens, time = synthetic_batch()

        with torch.no_grad():
            expected = baseline(x, lens, mask, time=time, original_mask=None)
            actual, output = policy(
                x, lens, mask, time=time,
                original_mask=None, return_policy=True,
            )
        self.assertIsNone(output)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_no_policy_bypasses_even_nonzero_policy_film(self):
        args = make_args(conditioning='no-policy')
        baseline = SMILELeanEncoder(args).eval()
        policy = SMILELeanPolicyEncoder(args).eval()
        policy.load_state_dict(baseline.state_dict(), strict=False)
        with torch.no_grad():
            policy.policy_film.generator[-1].weight.fill_(0.25)
            policy.policy_film.generator[-1].bias.fill_(0.5)
        x, mask, clean, lens, time = synthetic_batch()

        with torch.no_grad():
            expected = baseline(
                x, lens, mask, time=time, original_mask=clean
            )
            actual = policy(
                x, lens, mask, time=time, original_mask=clean
            )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_all_conditioning_modes_forward(self):
        x, mask, clean, lens, time = synthetic_batch()
        for mode in ('hidden', 'pi', 'residual', 'shuffled',
                     'density', 'time', 'no-policy'):
            with self.subTest(mode=mode):
                model = SMILELeanPolicyEncoder(
                    make_args(conditioning=mode)
                ).eval()
                with torch.no_grad():
                    representation, output = model(
                        x, lens, mask, time=time,
                        original_mask=clean, return_policy=True,
                    )
                self.assertEqual(representation.shape, (2, 5, 7, 8))
                self.assertEqual(output['prob'].shape, (2, 6, 5))

    def test_checkpoint_round_trip(self):
        args = make_args()
        model = SMILELeanPolicyEncoder(args).eval()
        x, mask, clean, lens, time = synthetic_batch()
        with torch.no_grad():
            expected, expected_policy = model(
                x, lens, mask, time=time,
                original_mask=clean, return_policy=True,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'policy.pth'
            torch.save({'encoder': model.state_dict()}, checkpoint_path)
            restored = SMILELeanPolicyEncoder(args).eval()
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            restored.load_state_dict(checkpoint['encoder'])

        with torch.no_grad():
            actual, actual_policy = restored(
                x, lens, mask, time=time,
                original_mask=clean, return_policy=True,
            )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_policy['hidden'], expected_policy['hidden'])
        torch.testing.assert_close(actual_policy['prob'], expected_policy['prob'])

    def test_policy_uses_clean_mask_not_corrupted_input_mask(self):
        model = SMILELeanPolicyEncoder(make_args()).eval()
        x, mask, clean, lens, time = synthetic_batch()
        corrupted = 1 - mask
        with torch.no_grad():
            _, output_a = model(
                x, lens, mask, time=time,
                original_mask=clean, return_policy=True,
            )
            _, output_b = model(
                x, lens, corrupted, time=time,
                original_mask=clean, return_policy=True,
            )
        torch.testing.assert_close(output_a['hidden'], output_b['hidden'])
        torch.testing.assert_close(output_a['prob'], output_b['prob'])

    def test_synthetic_combined_loss_backward_and_ema(self):
        args = make_args(max_len=4)
        online = SMILELeanPolicyEncoder(args).train()
        target = copy.deepcopy(online).eval()
        target.requires_grad_(False)
        decoder = EmbeddingDecoder(args).train()
        optimizer = torch.optim.Adam(
            list(online.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        x, mask, clean, lens, time = synthetic_batch(
            time_steps=4, num_vars=args.input_dim
        )
        corrupted = mask.clone()
        corrupted[:, 1] = 0

        with torch.no_grad():
            target_representation = target(
                x, lens, mask, time=time, original_mask=clean
            )
        representation, output = online(
            x, lens, corrupted, time=time,
            original_mask=clean, return_policy=True,
        )
        decoded = decoder(representation)
        reconstruction = F.mse_loss(decoded, target_representation)
        policy_loss = causal_policy_bce(output['logits'], clean, lens)
        loss = reconstruction + 0.1 * policy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertIsNotNone(online.policy_encoder.causal_conv.weight.grad)
        self.assertIsNotNone(online.policy_film.generator[-1].weight.grad)
        with torch.no_grad():
            for online_param, target_param in zip(
                    online.parameters(), target.parameters()):
                target_param.mul_(0.996).add_(online_param, alpha=0.004)


class PolicyDetachTest(unittest.TestCase):
    def _recon_only_grads(self, args):
        model = SMILELeanPolicyEncoder(args).train()
        x, mask, clean, lens, time = synthetic_batch(
            time_steps=args.max_len, num_vars=args.input_dim)
        representation = model(x, lens, mask, time=time, original_mask=clean)
        # Pure main-objective loss, NO policy BCE.
        loss = representation.pow(2).mean()
        loss.backward()
        enc_grad = model.policy_encoder.causal_conv.weight.grad
        film_grad = model.policy_film.generator[-1].weight.grad
        return enc_grad, film_grad

    def test_detached_condition_blocks_task_gradient_into_policy_encoder(self):
        # Default (no finetune_policy) -> detach: reconstruction/classification
        # must NOT train the policy encoder; it is BCE-only.
        args = make_args(max_len=4)
        self.assertTrue(SMILELeanPolicyEncoder(args).policy_detach)
        enc_grad, film_grad = self._recon_only_grads(args)
        self.assertIsNone(enc_grad)          # policy encoder untouched by main loss
        self.assertIsNotNone(film_grad)      # FiLM still adapts to the task

    def test_finetune_policy_allows_joint_task_gradient(self):
        # --finetune-policy -> no detach: the main objective may train the policy
        # encoder through the FiLM path (finetune adds no policy BCE).
        args = make_args(max_len=4)
        args.finetune_policy = True
        self.assertFalse(SMILELeanPolicyEncoder(args).policy_detach)
        enc_grad, film_grad = self._recon_only_grads(args)
        self.assertIsNotNone(enc_grad)
        self.assertIsNotNone(film_grad)


class PolicyRunnerTest(unittest.TestCase):
    def test_model_name_mapping(self):
        expected = {
            'smart-smile-lean-policy': 'hidden',
            'smart-smile-lean-policy-pi': 'pi',
            'smart-smile-lean-policy-residual': 'residual',
            'smart-smile-lean-policy-shuffled': 'shuffled',
            'smart-smile-lean-policy-density': 'density',
            'smart-smile-lean-policy-time': 'time',
            'smart-smile-lean-policy-no-policy': 'no-policy',
        }
        self.assertEqual(run_all_experiments.POLICY_MODELS, expected)
        for model_name, conditioning in expected.items():
            self.assertEqual(
                run_all_experiments.policy_model_flags(model_name),
                ['--use-smile-lean-policy',
                 '--policy-conditioning', conditioning],
            )

    def test_dry_run_forwards_policy_and_skip_test_flags(self):
        result = subprocess.run(
            [
                sys.executable,
                'run_all_experiments.py',
                '--models', 'smart-smile-lean-policy-shuffled',
                '--datasets', 'c12',
                '--seeds', '3407',
                '--pretrain-epochs', '1',
                '--finetune-epochs', '1',
                '--batch-size', '8',
                '--policy-loss-weight', '0.25',
                '--skip-test',
                '--dry-run',
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout
        self.assertIn('--use-smile-lean-policy', output)
        self.assertIn('--policy-conditioning shuffled', output)
        self.assertIn('--policy-loss-weight 0.25', output)
        self.assertIn('--skip-test', output)


if __name__ == '__main__':
    unittest.main()
