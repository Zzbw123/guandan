import json,sys,tempfile,unittest
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'experiments')]
from experiments.p5e_protocol import specification,schedule,config,blocked_order,INITS,ARMS

class ProtocolTests(unittest.TestCase):
    def test_json_budget_and_primary_candidate(self):
        s=specification();self.assertEqual(json.loads(json.dumps(s)),s)
        self.assertEqual(s['total_training_hands'],9600)
        self.assertEqual(s['hands_per_job'],1600)
        self.assertEqual(s['primary_candidate'],'label_balanced-314510')
        self.assertEqual(s['paired_control'],'ordinary-314510')
        self.assertEqual(s['checkpoint_waves'],[0,1,2,3,4,200,400])
        self.assertEqual(s['version'],'gd-p5e-paired-label-balanced-v1')
        self.assertEqual(s['validation_seeds'],list(range(208000,208065)))
        self.assertEqual(s['bootstrap_seed'],314553)
    def test_only_objective_differs(self):
        for initializer in INITS:
            ordinary=config(initializer,'ordinary')
            balanced=config(initializer,'label_balanced')
            self.assertEqual(ordinary['mode'],balanced['mode'])
            self.assertEqual(ordinary['mode'],'mixed')
            self.assertEqual(ordinary['objective'],'ordinary')
            self.assertEqual(balanced['objective'],'label_balanced')
            self.assertEqual({k:v for k,v in ordinary.items() if k!='objective'},
                             {k:v for k,v in balanced.items() if k!='objective'})
    def test_balanced_complete_schedules(self):
        for validation,n in [(False,208),(True,1560)]:
            trials=schedule(validation);self.assertEqual(len(trials),n)
            self.assertEqual(len({t.trial_id for t in trials}),n)
            self.assertTrue(all(v==8 for v in Counter((t.matchup_id,t.deal_seed) for t in trials).values()))
            self.assertTrue(all(sum(p=='dmc' for p in t.policies)==2 for t in trials))
    def test_blocked_orders_fixed_and_complete(self):
        s=specification();jobs={f'{a}-{i}' for i in INITS for a in ARMS}
        self.assertEqual(s['training_order'],blocked_order(314550))
        self.assertEqual(s['development_order'],blocked_order(314551))
        for key in ('training_order','development_order'):
            self.assertEqual(set(s[key]),jobs)
            for i in (0,2,4):self.assertEqual(s[key][i].split('-')[1],s[key][i+1].split('-')[1])
    def test_invalid_configs_and_sealed_split(self):
        for seed,arm in [(True,'ordinary'),(314500,'ordinary'),(314510,'best')]:
            with self.assertRaises(ValueError):config(seed,arm)
        s=specification();self.assertFalse(set(s['training_seeds'])&set(s['validation_seeds']))
        self.assertTrue(all(seed<9000000 for seed in s['training_seeds']+s['development_seeds']+s['validation_seeds']))
        self.assertFalse(set(s['development_seeds']) & set(s['validation_seeds']))
        self.assertEqual(s['training_seeds'][0],100000)
        self.assertEqual(s['training_seeds'][-1],101599)
    def test_seed_scan_boundary_and_parse_error(self):
        from experiments.p5e_scan import scan
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'artifacts'/'evaluations';folder.mkdir(parents=True)
            (folder/'historical.json').write_text('{"deal_seed": 207000}',encoding='utf-8')
            result=scan(root);self.assertEqual(result['status'],'PASS')
            (folder/'historical.json').write_text('{"deal_seed": 208000}',encoding='utf-8')
            result=scan(root);self.assertEqual(result['status'],'FAIL')
            self.assertEqual(result['hits'][0]['value'],208000)
            (folder/'historical.json').write_text('{invalid',encoding='utf-8')
            result=scan(root);self.assertEqual(result['status'],'FAIL')
            self.assertTrue(result['errors'])

class GuardTests(unittest.TestCase):
    def test_real_spawn_load_complete_candidates(self):
        import torch
        from experiments.p5d_training import GPUTrainer,score_many
        from experiments.p5d_checkpoint import save_checkpoint
        from experiments.p5e_guard import GPUInferenceGuard,GuardError
        from benchmark_learning_runtime import high_branch_fixture
        obs,legal=high_branch_fixture()
        self.assertEqual(len(legal),8769)
        with tempfile.TemporaryDirectory() as d:
            for arm in ARMS:
                with self.subTest(arm=arm):
                    t=GPUTrainer(config(314510,arm))
                    expected=int(score_many(t.model,[(obs,legal)],1024)[0].argmax())
                    path=Path(d)/arm;m=save_checkpoint(t,path)
                    with GPUInferenceGuard(path,m['sha256']) as g:
                        action,meta=g.act(obs,legal)
                        self.assertEqual(action,legal[expected]);self.assertEqual(meta['scored_candidates'],8769)
                        self.assertEqual(meta['device']['type'],'cuda')
                    self.assertFalse(g.is_alive())
                    with self.assertRaisesRegex(GuardError,'pinned checkpoint hash'):
                        GPUInferenceGuard(path,'0'*64)
    def test_fail_closed_crash_and_illegal(self):
        from experiments.p5e_guard import GPUInferenceGuard,GuardError
        from guandan.env import HandEnv
        env=HandEnv();obs=env.reset(109850);legal=env.legal_actions(obs.player_id)
        for mode in ('crash_act','bad_index'):
            with GPUInferenceGuard(Path('unused'),'0'*64,_test_mode=mode) as g:
                with self.assertRaises(GuardError):g.act(obs,legal)
                self.assertTrue(g.closed);self.assertFalse(g.is_alive())
    def test_deadline_and_pin_rejection(self):
        from experiments.p5e_guard import GPUInferenceGuard,GuardTimeout
        from guandan.env import HandEnv
        env=HandEnv();obs=env.reset(109851);legal=env.legal_actions(obs.player_id)
        with GPUInferenceGuard(Path('unused'),'0'*64,_test_mode='hang_act',decision_timeout=.15) as g:
            with self.assertRaises(GuardTimeout):g.act(obs,legal)
            self.assertFalse(g.is_alive())
        with self.assertRaises(ValueError):GPUInferenceGuard(Path('unused'),'invalid')
    def test_startup_timeout_reaps_worker(self):
        from experiments.p5e_guard import GPUInferenceGuard,GuardTimeout
        with self.assertRaises(GuardTimeout):
            GPUInferenceGuard(Path('unused'),'0'*64,startup_timeout=.15,_test_mode='hang_start')

if __name__=='__main__':unittest.main()
