"""CPU-only contracts for frozen P3l scheduling and historical seed metadata."""
import gzip,json,tempfile,unittest,sys
from unittest.mock import patch
from collections import Counter
from pathlib import Path
from experiments.p3l_protocol import specification,schedule,config,blocked_order,VALIDATION,ARMS,INITS
from experiments.p3l_scan import scan

class ProtocolTests(unittest.TestCase):
    def test_fixed_budget_and_roundtrip(self):
        spec=specification()
        self.assertEqual(json.loads(json.dumps(spec,allow_nan=False)),spec)
        self.assertEqual(ARMS,['constant','normcap'])
        self.assertEqual(spec['primary_candidate'],'normcap-314380')
        self.assertEqual(spec['training_seeds'],list(range(100200,100800)))
        self.assertEqual(VALIDATION,list(range(206000,206065)))
        self.assertEqual(spec['bootstrap_seed'],314425)
        self.assertEqual(spec['order_seeds'],dict(training=314423,development=314424,validation=314426))
        for key,seed in [('training_order',314423),('development_order',314424)]:
            jobs=spec[key];self.assertEqual(jobs,blocked_order(seed))
            self.assertEqual(len(set(jobs)),6)
            for i in range(0,6,2):
                self.assertEqual(jobs[i].split('-')[1],jobs[i+1].split('-')[1])
                self.assertEqual({jobs[i].split('-')[0],jobs[i+1].split('-')[0]},set(ARMS))
        for initializer in INITS:
            for arm in ARMS:self.assertEqual(config(initializer,arm)['auxiliary_weight'],.1)
        for initializer,arm in [(True,'constant'),(314383,'constant'),(314380,'centered')]:
            with self.assertRaises(ValueError):config(initializer,arm)

    def test_complete_stratified_schedule(self):
        for validation,games,deals in [(False,208,26),(True,1560,65)]:
            trials=schedule(validation)
            self.assertEqual(len(trials),games);self.assertEqual(len({t.trial_id for t in trials}),games)
            for opponent in (['greedy','random','team'] if validation else ['greedy']):
                group=[t for t in trials if t.matchup_id==f'dmc|dmc|{opponent}']
                self.assertEqual(Counter(t.deal_seed for t in group),{s:8 for s in (VALIDATION if validation else range(108100,108126))})
                levels=Counter(t.level for t in group)
                self.assertEqual(levels,{level:8*(5 if validation else 2) for level in range(2,15)})
                self.assertTrue(all(sum(p=='dmc' for p in t.policies)==2 for t in group))

    def test_scan_scope_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'artifacts/evaluations';base.mkdir(parents=True)
            (base/'p3j-historical.json').write_text(json.dumps({'deal_seed':206000}))
            (base/'p3k-seeds.jsonl').write_text(json.dumps({'nested':[{'validation_seeds':[206001]}]})+'\n')
            with gzip.open(base/'history.json.gz','wt') as f:json.dump({'seed':206002},f)
            (base/'p3l-prospective.json').write_text(json.dumps({'seed':206003}))
            (base/'sealed-test.json').write_text('not parsed')
            receipt=scan(root)
            self.assertEqual(receipt['status'],'FAIL')
            self.assertEqual({x['value'] for x in receipt['hits']},{206000,206001,206002})
            self.assertEqual(len(receipt['files']),4)
            self.assertEqual(receipt['hash_only_sealed'],['artifacts/evaluations/sealed-test.json'])
            self.assertFalse(receipt['errors'])
            (base/'malformed.jsonl').write_text('{broken')
            self.assertEqual(len(scan(root)['errors']),1)

    def test_missing_acceptance_fails_before_output_creation(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import p3l_run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'artifacts/evaluations';base.mkdir(parents=True)
            for name in ('p3l-seed-availability.json','p3l-test-receipt.json','p3l-resume-test.json'):
                (base/name).write_text(json.dumps({'status':'PASS'}))
            out=base/'p3l-never-created'
            with patch.object(p3l_run,'ROOT',root),patch.object(p3l_run,'historical_sources',return_value=[]),patch.object(sys,'argv',['p3l_run.py',str(out)]):
                with self.assertRaises(FileNotFoundError):p3l_run.main()
            self.assertFalse(out.exists())

    def test_secondary_pairing_uses_original_deal_groups(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import p3l_run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for arm,value in [('constant',.25),('normcap',.75)]:
                out=root/f'evaluations/validation-{arm}';out.mkdir(parents=True)
                summary={f'dmc|dmc|{opponent}':{'clusters':[dict(deal_seed=206000+i,level=2+i%13,games=8,win_rate=value) for i in range(65)]} for opponent in ('greedy','random','team')}
                (out/'report.json').write_text(json.dumps({'summary':summary}))
            p3l_run.paired_validation(root)
            report=json.loads((root/'paired-validation.json').read_text())
            self.assertEqual(report['role'],'SECONDARY_POINTWISE_DESCRIPTION')
            for comparison in report['comparisons'].values():
                self.assertEqual(comparison['normcap_minus_constant']['estimate'],.5)
                self.assertEqual(len(comparison['clusters']),65)
                self.assertTrue(comparison['normcap_minus_constant']['degenerate'])

if __name__=='__main__':unittest.main()
