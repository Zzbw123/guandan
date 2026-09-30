"""Validation data boundaries and protection against false promotion."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments/p3d"))
from protocol import schedule, summarize, specification, TRAIN_SEEDS, VALIDATION_SEEDS


def successful_rows():
    rows=[]
    for trial in schedule():
        rows.append(dict(trial_id=trial.trial_id,**asdict(trial),focal_team=trial.focal_team,
            status="ok",win=1,team_reward=1,focal_level_gain=3,opponent_level_gain=0,
            finish_order=[trial.focal_seat,(trial.focal_seat+2)%4],terminal_digest="a"*64,
            steps=20,illegal_actions=0,timeouts=0,error=None))
        rows[-1]["policies"]=list(rows[-1]["policies"])
    return rows


class ProtocolTests(unittest.TestCase):
    def test_specification_survives_preregistration_json_roundtrip(self):
        self.assertEqual(specification(),json.loads(json.dumps(specification())))

    def test_schedule_keeps_clusters_and_unseen_split_distinct(self):
        trials=schedule()
        self.assertEqual(len({t.trial_id for t in trials}),1560)
        self.assertTrue(all(100000<=s<110000 for s in TRAIN_SEEDS))
        self.assertTrue(all(200000<=s<210000 for s in VALIDATION_SEEDS))
        self.assertFalse(set(VALIDATION_SEEDS)&set(range(200000,200130)))
        self.assertFalse(set(TRAIN_SEEDS)&set(VALIDATION_SEEDS))
        for opponent in ("greedy","random","team"):
            subset=[t for t in trials if t.matchup_id.endswith(opponent)]
            self.assertEqual(set(Counter(t.deal_seed for t in subset).values()),{8})
            self.assertEqual(Counter(t.level for t in subset),{k:40 for k in range(2,15)})
            for seed in VALIDATION_SEEDS:
                group=[t for t in subset if t.deal_seed==seed]
                self.assertEqual({(t.rotation,t.swap) for t in group},{(r,s) for r in range(4) for s in range(2)})

    def test_degenerate_perfect_results_do_not_promote(self):
        result=summarize(successful_rows())
        self.assertEqual(result["validation_gate"],"NOT_ESTABLISHED")
        self.assertTrue(result["results"]["greedy"]["metrics"]["win_rate"]["degenerate"])
        self.assertFalse(result["model_promoted"])

    def test_missing_duplicate_and_failed_trials_block_analysis(self):
        rows=successful_rows()
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError): summarize(bad)
        rows[0]["status"]="error"
        with self.assertRaises(ValueError): summarize(rows)
