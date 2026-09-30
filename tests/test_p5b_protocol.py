import json,sys,tempfile,unittest
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'experiments')]
from experiments.p5b_protocol import specification,schedule,config,INITS,ARMS

class ProtocolTests(unittest.TestCase):
    def test_json_budget_and_primary_candidate(self):
        s=specification();self.assertEqual(json.loads(json.dumps(s)),s)
        self.assertEqual(s['total_training_hands'],9600)
        self.assertEqual(s['hands_per_job'],1600)
        self.assertEqual(s['primary_candidate'],'mixed-314510')
        self.assertEqual(s['checkpoint_waves'],[0,1,200,400])
    def test_balanced_complete_schedules(self):
        for validation,n in [(False,208),(True,1560)]:
            trials=schedule(validation);self.assertEqual(len(trials),n)
            self.assertEqual(len({t.trial_id for t in trials}),n)
            self.assertTrue(all(v==8 for v in Counter((t.matchup_id,t.deal_seed) for t in trials).values()))
            self.assertTrue(all(sum(p=='dmc' for p in t.policies)==2 for t in trials))
    def test_blocked_orders_fixed_and_complete(self):
        s=specification();jobs={f'{a}-{i}' for i in INITS for a in ARMS}
        for key in ('training_order','development_order'):
            self.assertEqual(set(s[key]),jobs)
            for i in (0,2,4):self.assertEqual(s[key][i].split('-')[1],s[key][i+1].split('-')[1])
    def test_invalid_configs_and_sealed_split(self):
        for seed,arm in [(True,'mixed'),(314500,'mixed'),(314510,'best')]:
            with self.assertRaises(ValueError):config(seed,arm)
        s=specification();self.assertFalse(set(s['training_seeds'])&set(s['validation_seeds']))
        self.assertTrue(all(seed<9000000 for seed in s['training_seeds']+s['development_seeds']+s['validation_seeds']))

class GuardTests(unittest.TestCase):
    def test_real_spawn_load_complete_candidates(self):
        import torch
        from experiments.p5a_training import GPUTrainer,score_many
        from experiments.p5a_checkpoint import save_checkpoint
        from experiments.p5b_guard import GPUInferenceGuard
        from benchmark_learning_runtime import high_branch_fixture
        t=GPUTrainer(config(314510,'mixed'));obs,legal=high_branch_fixture()
        expected=int(score_many(t.model,[(obs,legal)],1024)[0].argmax())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model';m=save_checkpoint(t,path)
            with GPUInferenceGuard(path,m['sha256']) as g:
                action,meta=g.act(obs,legal)
                self.assertEqual(action,legal[expected]);self.assertEqual(meta['scored_candidates'],8769)
                self.assertEqual(meta['device']['type'],'cuda')
            self.assertFalse(g.is_alive())
    def test_fail_closed_crash_and_illegal(self):
        from experiments.p5b_guard import GPUInferenceGuard,GuardError
        from guandan.env import HandEnv
        env=HandEnv();obs=env.reset(109850);legal=env.legal_actions(obs.player_id)
        for mode in ('crash_act','bad_index'):
            with GPUInferenceGuard(Path('unused'),'0'*64,_test_mode=mode) as g:
                with self.assertRaises(GuardError):g.act(obs,legal)
                self.assertTrue(g.closed);self.assertFalse(g.is_alive())
    def test_deadline_and_pin_rejection(self):
        from experiments.p5b_guard import GPUInferenceGuard,GuardTimeout
        from guandan.env import HandEnv
        env=HandEnv();obs=env.reset(109851);legal=env.legal_actions(obs.player_id)
        with GPUInferenceGuard(Path('unused'),'0'*64,_test_mode='hang_act',decision_timeout=.15) as g:
            with self.assertRaises(GuardTimeout):g.act(obs,legal)
            self.assertFalse(g.is_alive())
        with self.assertRaises(ValueError):GPUInferenceGuard(Path('unused'),'invalid')

if __name__=='__main__':unittest.main()
