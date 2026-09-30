from copy import deepcopy
import json
from pathlib import Path
import unittest

from guandan.evaluation.config import validate_config

ROOT = Path(__file__).resolve().parents[1]


class EvaluationConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/evaluation/p2-validation.json").read_text())
        self.splits = json.loads((ROOT / "configs/evaluation/seed-splits.json").read_text())

    def test_frozen_budget_and_full_matrix(self):
        seeds, matchups = validate_config(self.config, self.splits)
        self.assertEqual(seeds, list(range(200000, 200130)))
        self.assertEqual(len(set(x.key for x in matchups)), 27)
        self.assertEqual(sum(8 if x.focal == x.teammate else 16 for x in matchups) * len(seeds), 46800)

    def test_reject_reserved_overlap_and_changed_primary(self):
        for field, value in (("split", "reserved_test"), ("seed_start", 109999), ("primary_matchup", "team|team|random"),
                             ("primary_metric", "level_gain"), ("deal_count", 13), ("workers", True),
                             ("policies", ["team"]), ("decision_timeout_ms", float("nan"))):
            with self.subTest(field=field):
                config = deepcopy(self.config)
                config[field] = value
                with self.assertRaises(ValueError):
                    validate_config(config, self.splits)
        for field in ("reserved_test", "development"):
            splits = deepcopy(self.splits)
            splits[field] = {"start": 200000, "stop": 210000}
            with self.assertRaises(ValueError):
                validate_config(self.config, splits)

    def test_smoke_uses_development_only(self):
        config = json.loads((ROOT / "configs/evaluation/p2-smoke.json").read_text())
        seeds, matchups = validate_config(config, self.splits)
        self.assertTrue(set(seeds).isdisjoint(range(200000, 210000)))
        self.assertEqual(len(matchups), 27)


if __name__ == "__main__":
    unittest.main()
