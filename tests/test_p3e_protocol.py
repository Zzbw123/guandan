"""Budgets and independent data boundaries for paired training-size research."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from p3e_protocol import specification,schedule,TRAIN,DEV,VALIDATION,INITS

class P3EProtocolTests(unittest.TestCase):
    def test_frozen_json_and_selection(self):
        spec=specification()
        self.assertEqual(spec,json.loads(json.dumps(spec)))
        self.assertEqual(spec['candidate'],'314370-1600')
        self.assertEqual(set(spec['training_order']),set(INITS))
        self.assertEqual(set(spec['development_order']),{f'{s}-{n}' for s in INITS for n in (400,1600)})
        self.assertEqual(len(TRAIN),1600)

    def test_seed_partitions_and_reserved_set(self):
        self.assertTrue(all(100000<=s<110000 for s in TRAIN+DEV))
        self.assertFalse(set(TRAIN)&set(DEV))
        self.assertTrue(all(200000<=s<210000 for s in VALIDATION))
        self.assertFalse(set(VALIDATION)&set(range(201000,201065)))
        self.assertFalse(specification()['reserved_test_executed'])

    def test_complete_paired_deal_strata(self):
        for validation,groups in ((False,26),(True,65)):
            trials=schedule(validation)
            self.assertEqual(len(trials),groups*8*(3 if validation else 1))
            self.assertEqual(len({t.trial_id for t in trials}),len(trials))
            for matchup in {t.matchup_id for t in trials}:
                rows=[t for t in trials if t.matchup_id==matchup]
                self.assertEqual(set(Counter(t.deal_seed for t in rows).values()),{8})
                self.assertEqual(Counter(t.level for t in rows),{k:groups//13*8 for k in range(2,15)})
                for seed in {t.deal_seed for t in rows}:
                    paired=[t for t in rows if t.deal_seed==seed]
                    self.assertEqual({(t.rotation,t.swap) for t in paired},{(r,s) for r in range(4) for s in range(2)})

    def test_greedy_reference_uses_same_deals(self):
        a,b=schedule(),schedule(baseline=True)
        self.assertEqual([(t.deal_seed,t.level,t.rotation,t.swap,t.focal_team) for t in a],
                         [(t.deal_seed,t.level,t.rotation,t.swap,t.focal_team) for t in b])
        self.assertTrue(all(t.policies==('greedy',)*4 for t in b))
