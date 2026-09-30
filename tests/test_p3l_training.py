"""P3l wave constant identity, isolated checkpoint and process recovery contracts."""
from pathlib import Path
import tempfile,unittest,json,subprocess,sys
from unittest.mock import patch
try:import torch
except ImportError:torch=None
from test_p3j_training import equal

def config(seed=314381,arm='normcap'):
    return dict(seed=seed,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4,
                auxiliary_weight=.1,objective=arm)

@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA required')
class CUDATests(unittest.TestCase):
    def compare(self,a,b):
        for key in ('model','optimizer','policy_rng','torch_rng','cuda_rng','episodes','waves','updates','used_deal_seeds'):
            self.assertTrue(equal(a[key],b[key]),key)

    def test_constant_two_updates_waves_match_historical_centered_bitwise(self):
        from experiments.p3l_training import GPUTrainer
        from experiments.p3j_training import GPUTrainer as Old
        from experiments.p3l_checkpoint import capture
        from experiments.p3j_checkpoint import capture as oldcapture
        deals=[[(109100+i,2+i,i) for i in range(4)],[(109104+i,6+i,i) for i in range(4)]]
        old=Old(config(314380,'centered'))
        expected=[old.train_wave(d) for d in deals];before=oldcapture(old)
        new=GPUTrainer(config(314380,'constant'))
        actual=[new.train_wave(d) for d in deals];self.compare(before,capture(new))
        for a,b in zip(expected,actual):
            for key in ('hands','samples','mean_loss','auxiliary_loss','auxiliary_candidates','updates'):
                self.assertEqual(a[key],b[key],key)
            for x,y in zip(a['batches'],b['batches']):
                for key,value in x.items():self.assertEqual(value,y[key],key)
                self.assertEqual(y['alpha'],.1)

    def test_normcap_resume_separate_process_full_wave(self):
        from experiments.p3l_training import GPUTrainer
        from experiments.p3l_checkpoint import save_checkpoint,load_checkpoint,capture
        t=GPUTrainer(config());row=t.train_wave([(109110+i,2+i,i) for i in range(4)])
        self.assertEqual(sum(b['state_denominator'] for b in row['batches']),row['samples'])
        self.assertEqual(sum(sum(b['candidate_sizes']) for b in row['batches']),row['auxiliary_candidates'])
        for b in row['batches']:
            self.assertEqual(b['state_denominator'],b['samples'])
            self.assertEqual(b['auxiliary_requests'],b['samples'])
            self.assertLessEqual(b['alpha'],1.)
            if b['auxiliary_to_dmc_ratio'] is not None:self.assertLessEqual(b['auxiliary_to_dmc_ratio'],.1+1e-12)
            if b['joint_descent_factor'] is not None:self.assertGreaterEqual(b['joint_descent_factor'],.9-1e-12)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wave';m=save_checkpoint(t,p)
            expected=t.train_wave([(109120+i,6+i,i) for i in range(4)]);payload=capture(t)
            out=Path(d)/'child'
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/p3l_resume_test.py'),str(p),m['sha256'],str(out)],capture_output=True,text=True,timeout=240)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            data=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
            self.compare(payload,data)
            self.assertEqual(expected,json.loads((out.parent/'wave.json').read_text('utf-8')))
            restored=load_checkpoint(p,m['sha256']);self.assertEqual(restored.config['objective'],'normcap')
            from experiments.p3j_checkpoint import load_checkpoint as oldload
            with self.assertRaises(ValueError):oldload(p,m['sha256'])
            with self.assertRaises(ValueError):load_checkpoint(p,'0'*64)
            with self.assertRaises(FileExistsError):save_checkpoint(restored,p)

    def test_fault_boundary_reload_and_invalid_config(self):
        from experiments.p3l_training import GPUTrainer
        from experiments.p3l_checkpoint import save_checkpoint,load_checkpoint,capture
        cfg=config(314382);t=GPUTrainer(cfg);deals=[(109130+i,2+i,i) for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'before';m=save_checkpoint(t,p)
            with patch('experiments.p3l_training.joint_backward',side_effect=ValueError('fault')):
                with self.assertRaises(ValueError):t.train_wave(deals)
            self.assertFalse(t.ready_for_checkpoint)
            with self.assertRaises(ValueError):save_checkpoint(t,Path(d)/'bad')
            with self.assertRaises(RuntimeError):t.train_wave(deals)
            clean=GPUTrainer(cfg);a=clean.train_wave(deals);snapshot=capture(clean)
            recovered=load_checkpoint(p,m['sha256']);b=recovered.train_wave(deals)
            self.assertEqual(a,b);self.compare(snapshot,capture(recovered))
        for objective in (None,'centered','dmc',True):
            with self.assertRaises(ValueError):GPUTrainer({**cfg,'objective':objective})
        for weight in (0.,.2,True,float('nan')):
            with self.assertRaises(ValueError):GPUTrainer({**cfg,'auxiliary_weight':weight})

    def test_strict_payload_refuses_drift_and_preserves_rng(self):
        from experiments.p3l_training import GPUTrainer
        from experiments.p3l_checkpoint import capture,load_checkpoint,META
        import copy
        from io import BytesIO
        from hashlib import sha256
        t=GPUTrainer(config());payload=capture(t)
        faults=[('versions',{'checkpoint':'wrong'}),('sources',{}),('boundary','mid_wave'),
                ('episodes',1),('used_deal_seeds',[109100]),('config',{**config(),'objective':'bad'}),
                ('model',{**payload['model'],next(iter(payload['model'])):torch.zeros(1)}),
                ('optimizer',{'state':{},'param_groups':[]}),('torch_rng',torch.zeros(1,dtype=torch.uint8)),
                ('cuda_rng',[])]
        with tempfile.TemporaryDirectory() as directory:
            for index,(key,value) in enumerate(faults):
                with self.subTest(key=key):
                    bad=copy.deepcopy(payload);bad[key]=value
                    buffer=BytesIO();torch.save(bad,buffer);data=buffer.getvalue()
                    manifest={k:bad[k] for k in META}
                    manifest.update(sha256=sha256(data).hexdigest(),bytes=len(data))
                    target=Path(directory)/str(index);target.mkdir()
                    (target/'checkpoint.pt').write_bytes(data)
                    (target/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
                    torch.manual_seed(876543);torch.cuda.manual_seed_all(765432)
                    before=torch.get_rng_state().clone()
                    cuda=[g.clone() for g in torch.cuda.get_rng_state_all()]
                    with self.assertRaises(ValueError):load_checkpoint(target)
                    self.assertTrue(torch.equal(before,torch.get_rng_state()))
                    self.assertTrue(all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all())))
