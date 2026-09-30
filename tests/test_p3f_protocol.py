"""P3f sample independence, partition and fixed candidate contract."""
import unittest
from experiments.p3f_protocol import specification,schedule,TRAIN_SEEDS,DEV,VALIDATION,INITS

class ProtocolTests(unittest.TestCase):
    def test_schedule_is_grouped_complete_and_disjoint(self):
        for val,n in ((False,26),(True,65)):
            ts=schedule(val);expected=n*8*(3 if val else 1)
            self.assertEqual(len(ts),expected);self.assertEqual(len({t.trial_id for t in ts}),expected)
            for opponent in {t.matchup_id for t in ts}:
                for seed in {t.deal_seed for t in ts}:
                    group=[t for t in ts if t.matchup_id==opponent and t.deal_seed==seed]
                    self.assertEqual({(t.rotation,t.swap) for t in group},{(r,s) for r in range(4) for s in range(2)})
                    self.assertEqual(len({t.level for t in group}),1)
        self.assertFalse(set(TRAIN_SEEDS)&set(DEV));self.assertFalse(set(VALIDATION)&set(TRAIN_SEEDS+DEV))
        self.assertTrue(all(100000<=s<=109999 for s in TRAIN_SEEDS+DEV))
        self.assertEqual(len(TRAIN_SEEDS),800)

    def test_fixed_multi_initialization_and_no_adaptive_candidate(self):
        spec=specification();jobs={f'{a}-{s}' for a in ('control','teacher') for s in INITS}
        self.assertEqual(set(spec['training_order']),jobs);self.assertEqual(set(spec['development_order']),jobs)
        self.assertEqual(spec['validation_initializer'],314380)
        self.assertEqual(spec['phase1_hands']+spec['phase2_hands'],800)
        self.assertEqual(spec['teacher_observation_batch'],64)
        self.assertEqual(spec['teacher_epochs'],1)
        self.assertFalse(spec['reserved_test_executed']);self.assertFalse(spec['model_promoted'])

if __name__=='__main__':unittest.main()
