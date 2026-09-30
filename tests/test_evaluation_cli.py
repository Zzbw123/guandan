"""Controller tests for artifact serialization and acceptance boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("evaluation_cli", ROOT / "scripts/evaluate.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class EvaluationCliTests(unittest.TestCase):
    def test_json_roundtrip_is_canonical_with_numeric_keys(self):
        original = {"strata": {2: 10, 10: 10, 14: 10}, "distribution": {0.0: 2, 0.125: 4, 1.0: 1}}
        self.assertEqual(cli.canonical(original), cli.canonical(json.loads(json.dumps(original))))

    def test_primary_lower_bound_and_invalid_run(self):
        config = {"primary_matchup": "team|team|greedy", "split": "validation"}
        def summary(interval):
            return {config["primary_matchup"]: {"metrics": {"win_rate": {"ci95": interval}}}}
        self.assertEqual(cli.primary_result(True, summary([.5, .7]), config)["conclusion"], "NOT_ESTABLISHED")
        self.assertEqual(cli.primary_result(True, summary([.501, .7]), config)["conclusion"], "ABOVE_50_PERCENT")
        self.assertEqual(cli.primary_result(False, summary([.6, .7]), config)["conclusion"], "INVALID")
        self.assertEqual(cli.primary_result(True, summary(None), config)["conclusion"], "NOT_ESTABLISHED")

    def test_outcome_hash_ignores_completion_order(self):
        rows = [{"trial_id": "b", "win": 1}, {"trial_id": "a", "win": 0}]
        self.assertEqual(cli.outcome_digest(rows), cli.outcome_digest(list(reversed(rows))))
        altered = [{"trial_id": "b", "win": 0}, {"trial_id": "a", "win": 0}]
        self.assertNotEqual(cli.outcome_digest(rows), cli.outcome_digest(altered))

    def test_timing_quantiles_reject_invalid_samples(self):
        self.assertEqual(cli.sample_summary([1, 2, 3, 4]), {"count": 4, "p50": 2, "p95": 4, "max": 4})
        for value in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                cli.sample_summary([value])


if __name__ == "__main__":
    unittest.main()
