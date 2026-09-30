"""Focused checks for the independent P3b1 runtime probe."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class RuntimeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import benchmark_learning_runtime as probe
        cls.probe = probe
        probe.configure()

    def test_complete_cpu_scores_match_existing_dmc_scorer(self):
        from guandan.env import HandEnv
        from guandan.learning.model import DMCNetwork, score_actions
        import torch

        model = DMCNetwork().eval()
        env = HandEnv()
        obs = env.reset(102000, initial_level=2)
        actions = env.legal_actions(0)
        actual = self.probe.score_complete(model, obs, actions, "cpu")
        expected = score_actions(model, obs, actions, 256)
        self.assertEqual(len(actual), len(actions))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-4))

    def test_development_seed_guard_and_percentile(self):
        from guandan.learning.model import DMCNetwork

        with self.assertRaisesRegex(ValueError, "development seed"):
            self.probe.sample_hand(DMCNetwork(), "cpu", 9000000, 2, 0, 3)
        self.assertEqual(self.probe.percentile([1, 2, 3, 4], .5), 2.5)
        self.assertAlmostEqual(self.probe.percentile([1, 2, 3, 4], .95), 3.85)


if __name__ == "__main__":
    unittest.main()
