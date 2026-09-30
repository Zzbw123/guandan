"""Independent metadata audit must detect denominator and objective mutations."""
import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.controller_p5d_audit import audit_batch_rows


class BatchAuditTests(unittest.TestCase):
    def fixture(self,objective='label_balanced'):
        samples=[(None,None,y) for y in (1,1,1,-1,-1)]
        row=dict(mean_loss=.9,batches=[
            dict(objective=objective,start=0,n=4,positive=3,negative=1,
                 positive_weight=2/3 if objective=='label_balanced' else 1.,
                 negative_weight=2. if objective=='label_balanced' else 1.,single_class=False,loss=1.),
            dict(objective=objective,start=4,n=1,positive=0,negative=1,
                 positive_weight=0.,negative_weight=1.,single_class=True,loss=.5)])
        return samples,row,dict(batch_size=4,objective=objective)

    def test_tail_and_objectives(self):
        for name in ('ordinary','label_balanced'):
            audit_batch_rows(*self.fixture(name))

    def test_metadata_tampering(self):
        samples,row,cfg=self.fixture()
        for key,value in dict(n=5,positive=4,negative=2,start=1,objective='ordinary',
            positive_weight=.5,negative_weight=1.,single_class=True).items():
            changed=copy.deepcopy(row);changed['batches'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):audit_batch_rows(samples,changed,cfg)

    def test_missing_extra_and_boolean_denominators(self):
        samples,row,cfg=self.fixture()
        variants=[]
        bad=copy.deepcopy(row);bad['batches'].pop();variants.append(bad)
        bad=copy.deepcopy(row);bad['batches'][0]['extra']=1;variants.append(bad)
        bad=copy.deepcopy(row);bad['batches'][1]['n']=True;variants.append(bad)
        for changed in variants:
            with self.assertRaises(ValueError):audit_batch_rows(samples,changed,cfg)

    def test_loss_finite_and_aggregate(self):
        samples,row,cfg=self.fixture()
        for value in (float('nan'),float('inf'),-.1,1.1):
            changed=copy.deepcopy(row);changed['batches'][0]['loss']=value
            with self.assertRaises(ValueError):audit_batch_rows(samples,changed,cfg)


if __name__=='__main__':unittest.main()
