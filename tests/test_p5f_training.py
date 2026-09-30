import copy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from experiments.p5f_protocol import config, specification, schedule
from experiments.p5f_training import TeacherTrainer
from experiments.p5f_checkpoint import save_checkpoint, restore, capture, load_model
from guandan.env import HandEnv


def requests():
    env=HandEnv();obs=env.reset(100001)
    return [(obs,env.legal_actions(obs.player_id))]


class ProtocolTests(unittest.TestCase):
    def test_fixed_protocol(self):
        s=specification()
        self.assertEqual(json.loads(json.dumps(s)),s)
        self.assertEqual(len(schedule()),416)
        self.assertEqual(len({t.trial_id for t in schedule()}),416)
        self.assertEqual(s['validation_games'],0)
        self.assertEqual(len(s['training_order']),6)
        self.assertFalse(set(d[0] for d in s['training_deals']) & set(d[0] for d in s['development_deals']))


class TrainingTests(unittest.TestCase):
    def test_resume_and_model_load_both_objectives(self):
        for objective in ('ranking','regression'):
            t=TeacherTrainer(config(314560,objective),'0'*64)
            t.update(requests())
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'checkpoint';m=save_checkpoint(t,p)
                t.update(requests());expected=capture(t)
                resumed,_=restore(p,m['sha256'],'0'*64)
                resumed.update(requests());actual=capture(resumed)
                self.assertEqual(actual['updates'],2)
                for k,v in expected['model'].items():self.assertTrue(torch.equal(v,actual['model'][k]))
                for i,row in expected['optimizer']['state'].items():
                    for k,v in row.items():self.assertTrue(torch.equal(v,actual['optimizer']['state'][i][k]))
                self.assertTrue(torch.equal(expected['torch_rng'],actual['torch_rng']))
                for a,b in zip(expected['cuda_rng'],actual['cuda_rng']):self.assertTrue(torch.equal(a,b))
                model,_=load_model(p,m['sha256'],sha256((p/'manifest.json').read_bytes()).hexdigest())
                self.assertFalse(next(model.parameters()).requires_grad)

    def test_failure_requires_reload(self):
        t=TeacherTrainer(config(314560,'ranking'),'0'*64)
        with patch.object(t.optimizer,'step',side_effect=RuntimeError('injected Adam failure')):
            with self.assertRaises(RuntimeError):t.update(requests())
        self.assertFalse(t.ready)
        with self.assertRaises(RuntimeError):t.update(requests())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):save_checkpoint(t,Path(tmp)/'bad')

    def test_rehashed_metadata_and_tensor_tampering(self):
        t=TeacherTrainer(config(314560,'ranking'),'0'*64);t.update(requests())
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);m=save_checkpoint(t,base/'good')
            with self.assertRaises(ValueError):restore(base/'good','f'*64,'0'*64)
            with self.assertRaises(ValueError):restore(base/'good',m['sha256'],'1'*64)
            original=torch.load(base/'good/checkpoint.pt',weights_only=True)
            changes=[('versions',lambda p:p['versions'].update(checkpoint='wrong')),
                     ('sources',lambda p:p['sources'].update(fake='a'*64)),
                     ('boundary',lambda p:p.update(boundary='mid-update')),
                     ('objective',lambda p:p['config'].update(objective='unknown')),
                     ('samples',lambda p:p.update(samples=0)),
                     ('adam_step',lambda p:p['optimizer']['state'][0]['step'].fill_(22)),
                     ('adam_shape',lambda p:p['optimizer']['state'][0].update(exp_avg=torch.zeros(1))),
                     ('nonfinite',lambda p:next(iter(p['model'].values())).fill_(float('nan'))),
                     ('rng',lambda p:p.update(torch_rng=torch.zeros(2,dtype=torch.uint8)))]
            for name,change in changes:
                p=copy.deepcopy(original);change(p);folder=base/name;folder.mkdir()
                torch.save(p,folder/'checkpoint.pt');data=(folder/'checkpoint.pt').read_bytes()
                manifest={k:p[k] for k in m if k not in ('sha256','bytes')}
                manifest.update(sha256=sha256(data).hexdigest(),bytes=len(data))
                (folder/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises((ValueError,RuntimeError),msg=name):
                    restore(folder,manifest['sha256'],'0'*64)


if __name__=='__main__':unittest.main()
