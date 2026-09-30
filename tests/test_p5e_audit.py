"""Exercise the new audit against real historical trajectories and tampered logs."""
from pathlib import Path
import sys,copy,random,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.controller_p5b_training import audit_wave as old_audit,rows
from experiments.p5b_protocol import config as old_config
from experiments.controller_p5e_training import audit_wave
from experiments.p5e_protocol import config


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wave=next(rows(ROOT/'artifacts/evaluations/p5b-pool-v1/training/mixed-314510/waves.jsonl.gz'))
        _,samples=old_audit(cls.wave,0,old_config(314510,'mixed'),random.Random(314510))
        assert len(samples)<=256
        n=len(samples);p=sum(s[2]==1 for s in samples);m=n-p
        cls.row=copy.deepcopy(cls.wave)
        cls.row['objective']='ordinary'
        cls.row['batches']=[dict(start=0,objective='ordinary',n=n,positive=p,negative=m,
            positive_weight=float(p>0),negative_weight=float(m>0),single_class=not(p and m),loss=cls.wave['mean_loss'])]

    def test_full_real_wave_with_ordinary_metadata(self):
        counts,_=audit_wave(self.row,0,config(314510,'ordinary'),random.Random(314510))
        self.assertEqual(counts['samples'],self.wave['samples'])
        self.assertEqual(counts['positive']+counts['negative'],counts['samples'])
        self.assertEqual(counts['single_class_batches']+counts['dual_class_batches'],counts['updates'])

    def test_objective_and_batch_log_tamper_rejected(self):
        for key,value in [('objective','label_balanced'),('negative_weight',.125),('negative',-1),('n',0),('start',1)]:
            bad=copy.deepcopy(self.row)
            if key=='objective':bad[key]=value
            else:bad['batches'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                audit_wave(bad,0,config(314510,'ordinary'),random.Random(314510))


if __name__=='__main__':unittest.main()
