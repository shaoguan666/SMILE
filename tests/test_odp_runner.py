import unittest
from types import SimpleNamespace

import run_all_experiments as runner
import main_pretrain


class ODPRunnerTest(unittest.TestCase):
    def args(self):
        return SimpleNamespace(
            python_executable="python",
            use_torchrun=True,
            nproc_per_node=2,
            master_port_base=31800,
            devices="0,1",
            batch_size=64,
        )

    def test_all_locked_model_ids_have_explicit_flags(self):
        expected = {
            "smart-smile-lean-nocomiss-base",
            "smart-smile-lean-nocomiss-density",
            "smart-smile-lean-odp-late",
            "smart-smile-lean-odp-late-shuffled",
            "smart-smile-lean-nocomiss-pmatch",
        }
        self.assertEqual(set(runner.ODP_MODELS), expected)
        for model in expected:
            flags = runner.odp_model_flags(model)
            self.assertIn("--odp-model", flags)
            self.assertEqual(flags[-2:], ["--model-id", model])

    def test_two_process_launch_and_matched_batch_metadata(self):
        args = self.args()
        command = runner.launch_prefix(args, 3)
        self.assertIn("torch.distributed.run", command)
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "2")
        self.assertEqual(command[command.index("--master_port") + 1], "31802")
        env = runner.build_launch_env(args)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["SMART_PER_RANK_BATCH"], "32")
        self.assertEqual(env["SMART_WORLD_SIZE"], "2")
        self.assertEqual(env["SMART_EFFECTIVE_GLOBAL_BATCH"], "64")
        self.assertEqual(env["NCCL_P2P_DISABLE"], "1")
        self.assertEqual(env["NCCL_IB_DISABLE"], "1")

    def test_forecasting_modes_are_complete(self):
        self.assertEqual(set(runner.FORECAST_MODELS.values()),
                         {"prior", "time", "own", "full", "shuffled"})

    def test_odp_checkpoint_selection_cannot_fall_back_to_save_last(self):
        odp = SimpleNamespace(save_last=True, odp_model="late")
        main_pretrain.apply_pretrain_checkpoint_rule(odp, uses_curriculum=True)
        self.assertFalse(odp.save_last)
        legacy = SimpleNamespace(save_last=False, odp_model="none")
        main_pretrain.apply_pretrain_checkpoint_rule(legacy, uses_curriculum=True)
        self.assertTrue(legacy.save_last)


if __name__ == "__main__":
    unittest.main()
