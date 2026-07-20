import unittest

import run_all_experiments as runner
from models import smart


class SmartSmileLeanCleanupTest(unittest.TestCase):
    def test_public_model_line_excludes_removed_experiments(self):
        self.assertEqual(runner.ALL_MODELS, ["smart", "smart-smile-lean"])
        self.assertFalse(hasattr(runner, "ODP_MODELS"))
        self.assertFalse(hasattr(runner, "FORECAST_MODELS"))

    def test_lean_v2_architecture_is_removed(self):
        self.assertFalse(hasattr(smart, "SMILELeanV2Encoder"))
        self.assertFalse(hasattr(smart, "SMILELeanV2BasicBlock"))
        self.assertFalse(hasattr(smart, "DualHeadClassifier"))

    def test_public_lean_ablations_still_use_the_lean_runner_path(self):
        for model in runner.ABLATION_MODELS:
            self.assertIn(model, runner._LEAN_MODELS)
            self.assertIn(model, runner._LEAN_V1_ABLATION_MODELS)


if __name__ == "__main__":
    unittest.main()
