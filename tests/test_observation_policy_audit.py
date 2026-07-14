import numpy as np
import torch
import unittest

from audit_observation_policy import (
    ShallowLogReg,
    ShallowPlus,
    PerVarConv,
    CrossVarConv,
    CrossVarConvShuffled,
    flip_prob,
    shift_right,
    observed_run_length,
    build_loader,
    compute_static_priors,
    valid_mask_from_lens,
    count_params,
    match_pervar_width,
)
from data.dataloader import CustomDataset


def _make(model, V=3, T=7, B=2):
    mask = torch.randint(0, 2, (B, T, V), dtype=torch.float32)
    x = torch.randn(B, T, V)
    time = torch.arange(1, T + 1, dtype=torch.float32).repeat(B, 1)
    return model, mask, x, time


class CausalityTests(unittest.TestCase):
    """Every predictor at t may use only strictly-past mask/value information."""

    def _assert_causal(self, model, V=4, T=8, B=3, cut=5):
        torch.manual_seed(7)
        model = model.eval()
        mask = torch.randint(0, 2, (B, T, V), dtype=torch.float32)
        x = torch.randn(B, T, V)
        time = torch.arange(1, T + 1, dtype=torch.float32).repeat(B, 1)
        ref = model(mask, x, time)
        # Perturb current + future mask AND values at positions >= cut.
        cm, cx = mask.clone(), x.clone()
        cm[:, cut:, :] = 1.0 - cm[:, cut:, :]
        cx[:, cut:, :] = cx[:, cut:, :] + 3.0
        pert = model(cm, cx, time)
        # Outputs through position `cut` (inclusive: predicts m_cut from <cut) must not move.
        torch.testing.assert_close(ref[:, :cut + 1], pert[:, :cut + 1],
                                   rtol=1e-4, atol=1e-5)

    def test_shallow_lr_causal(self):
        self._assert_causal(ShallowLogReg(4, 8))

    def test_shallow_plus_causal(self):
        self._assert_causal(ShallowPlus(4, 8, time_dim=4, d_hidden=8))

    def test_d1_pervar_causal(self):
        self._assert_causal(PerVarConv(4, 8, time_dim=4, d_hidden=8, kernel=3))

    def test_d2_crossvar_causal(self):
        self._assert_causal(CrossVarConv(4, 8, time_dim=4, d_hidden=8, kernel=3))

    def test_d3_crossval_causal(self):
        self._assert_causal(CrossVarConv(4, 8, time_dim=4, d_hidden=8, kernel=3,
                                         use_values=True))

    def test_d2_shuffled_causal(self):
        # D2-shuffled permutes non-target channels across the BATCH (patient) axis,
        # but each channel is still shift_right (strictly past) and the conv is
        # causal -> no future-in-time leakage. Pin the permutation across both
        # forward calls so the only change is the perturbed future; strictly-past
        # inputs (for every patient/channel) are then identical and early outputs
        # must not move.
        from unittest import mock
        fixed = lambda n, **kw: torch.arange(n - 1, -1, -1, device=kw.get('device'))
        with mock.patch('torch.randperm', side_effect=fixed):
            self._assert_causal(CrossVarConvShuffled(4, 8, time_dim=4, d_hidden=8, kernel=3))


class TransitionFormulaTests(unittest.TestCase):
    def test_flip_prob_matches_definition(self):
        pi = torch.tensor([0.2, 0.7, 0.9, 0.4])
        m_prev = torch.tensor([0.0, 1.0, 0.0, 1.0])
        # m_prev=0 -> flip prob = pi ; m_prev=1 -> flip prob = 1-pi
        expected = torch.tensor([0.2, 0.3, 0.9, 0.6])
        torch.testing.assert_close(flip_prob(pi, m_prev), expected)

    def test_persistence_predicts_no_flip(self):
        # A persistence model has pi = m_prev, so implied flip prob is exactly 0.
        m_prev = torch.tensor([0.0, 1.0, 1.0, 0.0])
        torch.testing.assert_close(flip_prob(m_prev, m_prev), torch.zeros(4))


class HelperTests(unittest.TestCase):
    def test_shift_right_drops_current(self):
        m = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1)
        s = shift_right(m)
        torch.testing.assert_close(s.view(-1), torch.tensor([0.0, 1.0, 2.0, 3.0]))

    def test_run_length_vectorized_matches_loop(self):
        torch.manual_seed(1)
        m = (torch.rand(3, 10, 4) > 0.4).float()
        ms = shift_right(m)
        vec = observed_run_length(ms, 10)
        # reference loop
        out = torch.zeros_like(ms)
        prev = torch.zeros(3, 4)
        for t in range(10):
            prev = (prev + 1.0) * ms[:, t, :]
            out[:, t, :] = prev
        torch.testing.assert_close(vec, out / 10.0)

    def test_d1wide_matches_d2_param_budget(self):
        V, L, td, k = 17, 24, 16, 7
        d2 = count_params(CrossVarConv(V, L, td, 64, k, use_values=False))
        h = match_pervar_width(V, L, td, k, d2)
        d1w = count_params(PerVarConv(V, L, td, h, k))
        # Capacity match should be within 5% of the D2 budget.
        self.assertLess(abs(d1w - d2) / d2, 0.05)

    def test_valid_mask_from_lens_excludes_padding(self):
        actual = valid_mask_from_lens(torch.tensor([2, 1]), 3)
        expected = torch.tensor([[True, True, False], [True, False, False]])
        self.assertTrue(torch.equal(actual, expected))

    def test_static_priors_valid_and_production_conventions(self):
        samples = [
            {'x': np.zeros((2, 2), np.float32), 'mask': np.array([[1, 0], [0, 1]], np.float32),
             'time': np.array([1, 2], np.float32), 'lens': 2, 'labels': 0},
            {'x': np.zeros((1, 2), np.float32), 'mask': np.array([[1, 1]], np.float32),
             'time': np.array([1], np.float32), 'lens': 1, 'labels': 0},
        ]
        loader = build_loader(CustomDataset(samples), batch_size=2, shuffle=False, workers=0)
        p_valid, p_production, p_time = compute_static_priors(
            loader, input_dim=2, max_len=2, device=torch.device('cpu'))
        torch.testing.assert_close(p_valid, torch.tensor([2 / 3, 2 / 3]))
        torch.testing.assert_close(p_production, torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(p_time, torch.tensor([[1.0, 0.5], [0.0, 1.0]]))


if __name__ == '__main__':
    unittest.main()
