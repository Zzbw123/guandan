"""P3h numerical and operational contracts on actual CUDA when available."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch=None
from experiments.p3h_protocol import config,schedule,specification,TRAIN_SEEDS,DEV,VALIDATION
from experiments.p3h_objective import auxiliary_backward

class ProtocolTests(unittest.TestCase):
    def test_splits_and_fixed_candidate(self):
        self.assertEqual(len(schedule()),208);self.assertEqual(len(schedule(True)),1560)
        for a,b in ((TRAIN_SEEDS,DEV),(TRAIN_SEEDS,VALIDATION),(DEV,VALIDATION)):
            self.assertFalse(set(a)&set(b))
        self.assertEqual(specification()['validation_initializer'],314380)
        self.assertEqual(len(set(t.trial_id for t in schedule(True))),1560)
        self.assertEqual(config(314380,'aux')['auxiliary_weight'],.1)
        with self.assertRaises(ValueError):config(314383,'aux')

    def test_invalid_objective_arguments(self):
        for w in (-1,True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):auxiliary_backward(None,[],w)
        for chunk in (0,False,1.5):
            with self.assertRaises(ValueError):auxiliary_backward(None,[],.1,chunk)

@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA required')
class CUDATests(unittest.TestCase):
    def same_state(self,a,b):
        for k,v in a.model.state_dict().items():self.assertTrue(torch.equal(v,b.model.state_dict()[k]),k)
        for k,state in a.optimizer.state_dict()['state'].items():
            for field,v in state.items():self.assertTrue(torch.equal(v,b.optimizer.state_dict()['state'][k][field]))
        self.assertEqual(a.rng.getstate(),b.rng.getstate())
        self.assertEqual((a.episodes,a.waves,a.updates),(b.episodes,b.waves,b.updates))

    def test_zero_weight_matches_old_wave_bitwise(self):
        from guandan_gpu.training import GPUTrainer as Old
        from experiments.p3h_training import GPUTrainer
        cfg=config(314380);oldcfg={k:v for k,v in cfg.items() if k!='auxiliary_weight'}
        old=Old(oldcfg);a=old.train_wave([(109000+i,2+i,i) for i in range(4)])
        new=GPUTrainer(cfg);b=new.train_wave([(109000+i,2+i,i) for i in range(4)])
        self.same_state(old,new)
        for k,v in a.items():self.assertEqual(v,b[k])
        self.assertEqual(b['auxiliary_candidates'],0)

    def test_dense_combined_loss_gradients_one_step(self):
        from guandan.env import HandEnv
        from guandan.agents import GreedyAgent
        from guandan.learning.encoding import encode_observation,encode_action
        from experiments.p3h_training import GPUTrainer
        from guandan.types import PlayerObservation,Action
        a=GPUTrainer(config(314380,'aux'));b=GPUTrainer(config(314380,'aux'))
        obs=PlayerObservation(0,(0,54),2,0,(2,3,3,3),(),None,None,(),(),0,False,None)
        req=[(obs,[Action('single',(0,),2),Action('single',(54,),2),Action('pair',(0,54),2)]),
             (obs,[Action('pair',(0,54),2)])]
        def dmc(model):
            st=torch.tensor([encode_observation(o) for o,l in req],device='cuda')
            ac=torch.tensor([encode_action(l[0]) for o,l in req],device='cuda')
            return (model(st,ac)-torch.tensor([1.,-1.],device='cuda')).square().mean()
        dmc(a.model).backward();res=auxiliary_backward(a.model,req,.1,2)
        terms=[]
        for o,legal in req:
            raw=[GreedyAgent().score(o,act) for act in legal];lo,hi=min(raw),max(raw)
            tar=[0.]*len(raw) if lo==hi else [1.6*(x-lo)/(hi-lo)-.8 for x in raw]
            st=torch.tensor([encode_observation(o)]*len(legal),device='cuda')
            ac=torch.tensor([encode_action(act) for act in legal],device='cuda')
            terms.append((b.model(st,ac)-torch.tensor(tar,device='cuda')).square().mean())
        aux=torch.stack(terms).mean();(dmc(b.model)+.1*aux).backward()
        self.assertAlmostEqual(res['loss'],aux.item(),places=6)
        self.assertEqual(res['candidates'],4);self.assertEqual(res['requests'],2)
        for x,y in zip(a.model.parameters(),b.model.parameters()):torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-5)
        a.optimizer.step();b.optimizer.step()
        for x,y in zip(a.model.parameters(),b.model.parameters()):torch.testing.assert_close(x,y,atol=2e-6,rtol=1e-5)

    def test_high_candidates_no_truncation(self):
        from benchmark_learning_runtime import high_branch_fixture
        from experiments.p3h_training import GPUTrainer
        trainer=GPUTrainer(config(314380,'aux'));obs,legal=high_branch_fixture()
        r=auxiliary_backward(trainer.model,[(obs,legal)],.1,1024)
        self.assertEqual(r['candidates'],8769)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters()))

    def test_aux_wave_coverage_checkpoint_resume_and_guard(self):
        from experiments.p3h_training import GPUTrainer
        from experiments.p3h_checkpoint import save_checkpoint,load_checkpoint
        from experiments.p3h_guard import GPUInferenceGuard
        from guandan.env import HandEnv
        trainer=GPUTrainer(config(314381,'aux'))
        r=trainer.train_wave([(109010+i,2+i,i) for i in range(4)])
        self.assertEqual(sum(b['samples'] for b in r['batches']),r['samples'])
        self.assertEqual(sum(b['auxiliary_requests'] for b in r['batches']),r['samples'])
        self.assertEqual(sum(b['auxiliary_candidates'] for b in r['batches']),r['auxiliary_candidates'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint';m=save_checkpoint(trainer,p)
            self.assertEqual(m['versions']['training'],'gd-p3h-wave-v1')
            restored=load_checkpoint(p,m['sha256']);self.same_state(trainer,restored)
            deals=[(109020+i,2+i,i) for i in range(4)]
            expected=trainer.train_wave(deals);actual=restored.train_wave(deals)
            self.same_state(trainer,restored);self.assertEqual(expected,actual)
            env=HandEnv();obs=env.reset(109030);legal=env.legal_actions(obs.player_id)
            with GPUInferenceGuard(p,m['sha256']) as guard:
                action,info=guard.act(obs,legal)
                self.assertIn(action,legal);self.assertEqual(info['scored_candidates'],len(legal))
            self.assertTrue(guard.closed);self.assertFalse(guard.is_alive())
            with self.assertRaises(ValueError):load_checkpoint(p,'0'*64)

    def test_fail_closed_and_invalid_config(self):
        from experiments.p3h_training import GPUTrainer
        trainer=GPUTrainer(config(314380,'aux'))
        with patch('experiments.p3h_training.auxiliary_backward',side_effect=ValueError('injected')):
            with self.assertRaises(ValueError):trainer.train_wave([(109040+i,2+i,i) for i in range(4)])
        self.assertFalse(trainer.ready_for_checkpoint)
        with self.assertRaises(RuntimeError):trainer.train_wave([(109050+i,2+i,i) for i in range(4)])
        for weight in (None,True,.2,float('nan')):
            with self.assertRaises(ValueError):GPUTrainer({**config(314380), 'auxiliary_weight':weight})

class GuardFaultTests(unittest.TestCase):
    def test_startup_hang_and_action_fault_reaped(self):
        from experiments.p3h_guard import GPUInferenceGuard,GuardError
        from guandan.env import HandEnv
        with self.assertRaises((GuardError,TimeoutError)):
            GPUInferenceGuard(Path('unused'),'0'*64,startup_timeout=.2,_test_mode='hang_start')
        env=HandEnv();obs=env.reset(109060);legal=env.legal_actions(obs.player_id)
        for mode in ('hang_act','crash_act','bad_index'):
            g=GPUInferenceGuard(Path('unused'),'0'*64,decision_timeout=.2,_test_mode=mode)
            with self.assertRaises((GuardError,TimeoutError)):g.act(obs,legal)
            self.assertTrue(g.closed);self.assertFalse(g.is_alive())
