"""Pure edge cases for P3g; no artifact or CUDA reads."""
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.p3g_scores import (Aggregate, BASE, ROOT, canonical_digest,
    normalized_targets, summarize, verify_phase1_evaluation_binding)


class ScoreTests(unittest.TestCase):
    def test_first_model_tie_and_exact_teacher_membership(self):
        row = summarize([.5, .5, .2], [4, 4, 1], [1])
        self.assertEqual(row["argmax_index"], 0)
        self.assertTrue(row["teacher_top1"])
        self.assertEqual(row["normalized_regret"], 0)
        self.assertTrue(row["finish_miss"])

    def test_regret_and_mse(self):
        row = summarize([1, 0, -.5], [1, 2, 3], [0, 2])
        self.assertEqual(normalized_targets([1, 2, 3]), [-.8, 0., .8])
        self.assertEqual(row["normalized_regret"], 1.)
        self.assertAlmostEqual(row["candidate_mse"], (1.8 ** 2 + 1.3 ** 2) / 3)
        self.assertFalse(row["finish_miss"])
        self.assertEqual(row["saturated_count"], 1)

    def test_equal_teacher_and_singleton(self):
        for scores, teacher in (([.2], [9]), ([.1, -.99], [4, 4])):
            row = summarize(scores, teacher, [])
            self.assertTrue(row["teacher_top1"])
            self.assertEqual(row["normalized_regret"], 0)
            self.assertIsNone(row["finish_miss"])
            self.assertEqual(normalized_targets(teacher), [0.] * len(scores))

    def test_invalid_arrays(self):
        for scores, teacher in (([], []), ([0], [0, 1]), ([math.nan], [0]),
                                ([0], [math.inf]), ([math.inf], [0])):
            with self.assertRaises(ValueError):
                summarize(scores, teacher, [])
        for teacher in ([], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                normalized_targets(teacher)

    def test_finish_index_validation(self):
        for indexes in ((-1,), (1,), (True,), (0, 0), (.0,)):
            with self.assertRaises(ValueError):
                summarize([0], [0], indexes)

    def test_aggregation_denominators(self):
        acc = Aggregate()
        a = summarize([.99], [1], [0])
        b = summarize([1., 0., 0.], [1, 2, 3], [2])
        acc.add(a)
        acc.add(b)
        row = acc.result()
        self.assertEqual(row["observations"], 2)
        self.assertEqual(row["saturation_rate"], .5)
        self.assertEqual(row["finish_miss_rate"], .5)
        self.assertEqual(row["mean_teacher_top1"], .5)
        self.assertAlmostEqual(row["mean_candidate_mse"], (a["candidate_mse"] + b["candidate_mse"]) / 2)
        self.assertIsNone(Aggregate().result()["mean_candidate_mse"])

    def test_canonical_hash_order_unicode(self):
        self.assertEqual(canonical_digest({"甲": [1, 2], "b": 3}),
                         canonical_digest({"b": 3, "甲": [1, 2]}))
        self.assertNotEqual(canonical_digest([1, 2]), canonical_digest([2, 1]))

    def test_phase1_evaluation_binding_rejects_wrong_identity_and_pins(self):
        seed = 314380
        checkpoint = BASE / f"training/teacher-{seed}/phase1"
        folder = ROOT / "synthetic-evaluation"
        prereg = {"input_sha256": {
            (checkpoint / "raw.pt").relative_to(ROOT).as_posix(): "raw",
            (checkpoint / "manifest.json").relative_to(ROOT).as_posix(): "manifest"}}
        binding = dict(checkpoint=str(checkpoint), manifest_sha256="manifest",
                       candidate_sha256="raw", preregistration_sha256="pre",
                       phase1_model_sha256="model", job="teacher-phase1-314380",
                       seed=seed, arm="teacher", games=208)
        report = {k: v for k, v in binding.items() if k not in ("checkpoint", "manifest_sha256")}
        def fake_read(path):
            return binding if Path(path).name == "binding.json" else {
                "sha256": "raw", "model_sha256": "model"}
        def fake_digest(path):
            return "raw" if Path(path).name == "raw.pt" else "manifest"
        with patch("experiments.p3g_scores.read", side_effect=fake_read), \
             patch("experiments.p3g_scores.digest", side_effect=fake_digest):
            verify_phase1_evaluation_binding(folder, report, prereg, "pre", seed)
            for target, key, wrong in ((binding, "checkpoint", str(BASE / "wrong")),
                    (binding, "manifest_sha256", "wrong"),
                    (binding, "candidate_sha256", "wrong"),
                    (binding, "preregistration_sha256", "wrong"),
                    (report, "candidate_sha256", "wrong"),
                    (report, "phase1_model_sha256", "wrong"),
                    (report, "preregistration_sha256", "wrong"),
                    (report, "seed", 314381), (report, "games", 207)):
                previous = target[key]
                target[key] = wrong
                with self.subTest(key=key, wrong=wrong), self.assertRaises(ValueError):
                    verify_phase1_evaluation_binding(folder, report, prereg, "pre", seed)
                target[key] = previous


if __name__ == "__main__":
    unittest.main()
