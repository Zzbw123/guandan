import copy, unittest
from experiments.p5c_diagnostic import metrics,aggregate,rebuild_hand,rows,BASE
from experiments.controller_p5c_audit import verify_scores
from guandan.types import Action

class DiagnosticTests(unittest.TestCase):
    def test_forced_pass(self):
        m=metrics([.2],[0],[],[]);self.assertFalse(m['optional']);self.assertIsNone(m['pass_margin'])
    def test_single_finish(self):
        m=metrics([.2],[],[0],[0]);self.assertFalse(m['finish_miss']);self.assertIsNone(m['finish_margin'])
    def test_first_tie(self):
        m=metrics([.1,.1],[0],[1],[1]);self.assertEqual(m['argmax'],0);self.assertTrue(m['finish_miss'])
    def test_full_denominator(self):
        self.assertEqual(metrics([1,2,3],[0],[1,2],[2])['candidates'],3)
    def test_partition_rejection(self):
        for pa,pl in [([0],[0,1]),([0],[]),([],[1])]:
            with self.assertRaises(ValueError):metrics([1,2],pa,pl,[])
    def test_nonfinite_rejection(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):metrics([value],[0],[],[])
    def test_invalid_finish(self):
        with self.assertRaises(ValueError):metrics([1,2],[0],[1],[0])
    def test_empty_rejection(self):
        with self.assertRaises(ValueError):metrics([],[],[],[])
    def test_independent_tamper_rejection(self):
        legal=[Action('pass'),Action('single',(0,),2)]
        row=dict(scores=[0.,1.],metrics=metrics([0.,1.],[0],[1],[1]))
        verify_scores(row,[0.,1.],legal,1)
        for change in ('value','length','nan','argmax','denominator','margin'):
            bad=copy.deepcopy(row)
            if change=='value':bad['scores'][0]=2.
            elif change=='length':bad['scores'].pop()
            elif change=='nan':bad['scores'][0]=float('nan')
            elif change=='argmax':bad['metrics']['argmax']=0
            elif change=='denominator':bad['metrics']['candidates']=3
            else:bad['metrics']['pass_margin']=123.
            with self.assertRaises((ValueError,AssertionError)):verify_scores(bad,[0.,1.],legal,1)
    def test_aggregate_missing_opportunity(self):
        m=metrics([.2],[0],[],[])
        result=aggregate([], [dict(source='s',model='m',metrics=m)])['scoring']['s|m']
        self.assertEqual(result['states'],1);self.assertEqual(result['optional'],0)
        self.assertNotIn('pass_margin_count',result)
    def test_real_source_mutations(self):
        h=next(rows(BASE/'training/selfplay-314510/waves.jsonl.gz'))['hands'][0]
        good,states=rebuild_hand('selfplay-314510',0,h)
        self.assertGreater(good['samples'],0);self.assertGreater(len(states),0)
        for change in ('seed','role','selected','reward','sample','candidate'):
            bad=copy.deepcopy(h)
            if change=='seed':bad['seed']+=1
            elif change=='role':bad['roles'][0]='frozen'
            elif change=='selected':bad['decisions'][0]['selected']=-1
            elif change=='reward':bad['team_rewards'].reverse()
            elif change=='sample':bad['decisions'][0]['sample']=False
            else:bad['decisions'][0]['candidates']+=1
            with self.assertRaises(ValueError):rebuild_hand('selfplay-314510',0,bad)

if __name__=='__main__':unittest.main()
