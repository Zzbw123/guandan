"""P3j objective identity, paired baseline, real CUDA recovery and guard contracts."""
from pathlib import Path
import tempfile,unittest,copy
from unittest.mock import patch
try:import torch
except ImportError:torch=None
from experiments.p3j_protocol import config,schedule,specification,INITS,ARMS,TRAIN_SEEDS,DEV,VALIDATION

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

class ProtocolTests(unittest.TestCase):
    def test_fixed_budget_and_blocked_schedule(self):
        s=specification();self.assertEqual(s['primary_candidate'],'centered-314380')
        self.assertEqual(len(schedule()),208);self.assertEqual(len(schedule(True)),1560)
        self.assertEqual(len(set(t.trial_id for t in schedule(True))),1560)
        for a,b in ((TRAIN_SEEDS,DEV),(TRAIN_SEEDS,VALIDATION),(DEV,VALIDATION)):self.assertFalse(set(a)&set(b))
        for key in ('training_order','development_order'):
            self.assertEqual(set(s[key]),{f'{a}-{i}' for i in INITS for a in ARMS})
            for j in range(0,6,2):self.assertEqual(s[key][j].split('-')[1],s[key][j+1].split('-')[1])

@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA required')
class CUDATests(unittest.TestCase):
    def compare(self,a,b):
        for key in ('model','optimizer','policy_rng','torch_rng','cuda_rng','episodes','waves','updates','used_deal_seeds'):
            self.assertTrue(equal(a[key],b[key]),key)

    def test_absolute_reproduces_p3h_aux_bitwise(self):
        from experiments.p3j_training import GPUTrainer
        from experiments.p3h_training import GPUTrainer as Old
        from experiments.p3j_checkpoint import capture
        from experiments.p3h_checkpoint import capture as oldcapture
        cfg=config(314380);old=Old({k:v for k,v in cfg.items() if k!='objective'})
        deals=[(109100+i,2+i,i) for i in range(4)]
        a=old.train_wave(deals);oldstate=oldcapture(old)
        new=GPUTrainer(cfg);b=new.train_wave(deals);newstate=capture(new)
        self.compare(oldstate,newstate)
        for k,v in a.items():self.assertEqual(v,b[k],k)

    def test_centered_resume_separate_process_and_production_guard(self):
        from experiments.p3j_training import GPUTrainer
        from experiments.p3j_checkpoint import save_checkpoint,load_checkpoint,capture
        from experiments.p3j_guard import GPUInferenceGuard
        from guandan.env import HandEnv
        from benchmark_learning_runtime import high_branch_fixture
        import subprocess,sys,json
        t=GPUTrainer(config(314381,'centered'))
        first=t.train_wave([(109110+i,2+i,i) for i in range(4)])
        self.assertEqual(sum(b['auxiliary_requests'] for b in first['batches']),first['samples'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'wave';m=save_checkpoint(t,path)
            self.assertEqual(m['config']['objective'],'centered')
            deals=[(109120+i,6+i,i) for i in range(4)]
            expected=t.train_wave(deals);payload=capture(t)
            out=Path(d)/'child'
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/p3j_resume_test.py'),str(path),m['sha256'],str(out)],capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            data=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
            self.compare(payload,data)
            actual=json.loads((out.parent/'wave.json').read_text('utf-8'));self.assertEqual(expected,actual)
            restored=load_checkpoint(path,m['sha256']);self.assertEqual(restored.config['objective'],'centered')
            with GPUInferenceGuard(path,m['sha256']) as guard:
                obs,legal=high_branch_fixture();action,info=guard.act(obs,legal)
                self.assertIn(action,legal);self.assertEqual(info['scored_candidates'],8769)
            self.assertTrue(guard.closed);self.assertFalse(guard.is_alive())
            from experiments.p3h_checkpoint import load_checkpoint as oldload
            with self.assertRaises(ValueError):oldload(path,m['sha256'])
            with self.assertRaises(ValueError):load_checkpoint(path,'0'*64)

    def test_fault_boundary_reload_and_invalid_objective(self):
        from experiments.p3j_training import GPUTrainer
        from experiments.p3j_checkpoint import save_checkpoint,load_checkpoint,capture
        cfg=config(314382,'centered');t=GPUTrainer(cfg);deals=[(109130+i,2+i,i) for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'before';m=save_checkpoint(t,p)
            with patch('experiments.p3j_training.centered_backward',side_effect=ValueError('fault')):
                with self.assertRaises(ValueError):t.train_wave(deals)
            self.assertFalse(t.ready_for_checkpoint)
            with self.assertRaises(ValueError):save_checkpoint(t,Path(d)/'bad')
            with self.assertRaises(RuntimeError):t.train_wave(deals)
            clean=GPUTrainer(cfg);a=clean.train_wave(deals);snapshot=capture(clean)
            recovered=load_checkpoint(p,m['sha256']);b=recovered.train_wave(deals)
            self.assertEqual(a,b);self.compare(snapshot,capture(recovered))
        for objective in (None,'aux','dmc',True):
            with self.assertRaises(ValueError):GPUTrainer({**cfg,'objective':objective})
        for weight in (0.,.2,True,float('nan')):
            with self.assertRaises(ValueError):GPUTrainer({**cfg,'auxiliary_weight':weight})

class GuardFaultTests(unittest.TestCase):
    def test_failures_are_reaped(self):
        from experiments.p3j_guard import GPUInferenceGuard,GuardError
        from guandan.env import HandEnv
        with self.assertRaises((GuardError,TimeoutError)):
            GPUInferenceGuard(Path('unused'),'0'*64,startup_timeout=.2,_test_mode='hang_start')
        env=HandEnv();obs=env.reset(109140);legal=env.legal_actions(obs.player_id)
        for mode in ('hang_act','crash_act','bad_index'):
            guard=GPUInferenceGuard(Path('unused'),'0'*64,decision_timeout=.2,_test_mode=mode)
            with self.assertRaises((GuardError,TimeoutError)):guard.act(obs,legal)
            self.assertTrue(guard.closed);self.assertFalse(guard.is_alive())
