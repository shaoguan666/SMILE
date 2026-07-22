import unittest
from types import SimpleNamespace

import torch

from models.smart import SMILELeanEncoder


def encoder_args(**overrides):
    values = {
        "input_dim": 5,
        "d_model": 32,
        "n_heads": 4,
        "e_layers": 2,
        "dropout": 0.0,
        "max_len": 8,
        "time_dim": 16,
        "obs_density_window": 5,
        "abl_no_density": False,
        "abl_no_mnar_bias": False,
        "abl_no_film": False,
        "abl_no_time_mnar": False,
        "abl_no_time_pe": False,
        "abl_random_bias": False,
        "abl_global_comiss": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PhysicalTimePEAblationTest(unittest.TestCase):
    def test_ablation_removes_only_the_additive_time_projection(self):
        full = SMILELeanEncoder(encoder_args())
        ablated = SMILELeanEncoder(encoder_args(abl_no_time_pe=True))

        full_keys = set(full.state_dict())
        ablated_keys = set(ablated.state_dict())
        self.assertEqual(
            full_keys - ablated_keys,
            {"time_pe_proj.weight", "time_pe_proj.bias"},
        )
        self.assertEqual(ablated_keys - full_keys, set())

        removed = sum(parameter.numel() for parameter in full.time_pe_proj.parameters())
        self.assertEqual(removed, 16 * 32 + 32)
        self.assertEqual(
            sum(parameter.numel() for parameter in full.parameters())
            - sum(parameter.numel() for parameter in ablated.parameters()),
            removed,
        )

    def test_ablation_keeps_time_conditioned_interaction_paths(self):
        ablated = SMILELeanEncoder(encoder_args(abl_no_time_pe=True))

        self.assertTrue(hasattr(ablated, "time_encoder"))
        for block in ablated.blocks:
            self.assertTrue(block.use_film)
            self.assertTrue(block.use_time_mnar)
            self.assertTrue(block.needs_time_enc)

    def test_full_projection_is_zero_initialized(self):
        full = SMILELeanEncoder(encoder_args())

        self.assertTrue(torch.count_nonzero(full.time_pe_proj.weight).item() == 0)
        self.assertTrue(torch.count_nonzero(full.time_pe_proj.bias).item() == 0)


if __name__ == "__main__":
    unittest.main()
