"""Controller recomputation of raw score arrays and diagnostic provenance."""
from collections import Counter
from pathlib import Path
import json
import math
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3e_common import check,digest,read,write

def review():
    score=ROOT/'artifacts/evaluations/p3f-score-diagnostic-v1'
    target=ROOT/'artifacts/evaluations/p3f-target-trace-v1'
    pre=read(score/'specification.json');report=read(score/'report.json')
    check(report['status']=='PASS' and not report['failures'],'diagnostic result')
    for key in ('source_hashes','input_hashes'):
        for name,value in pre[key].items():check(digest(ROOT/name)==value,f'diagnostic provenance {name}')
    for name,key in [('specification.json','specification_sha256'),('scores.jsonl','scores_sha256'),('runtime.json','runtime_sha256')]:
        check(digest(score/name)==report[key],'score artifact hash')
    jobs={};maxerror=0.;seen=set()
    with (score/'scores.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);identity=(r['job'],r['trial_id'],r['step']);check(identity not in seen,'unique diagnostic decision');seen.add(identity)
            a,b,c=r['scores'],r['scores_chunk37'],r['scores_direct_forward']
            check(len(a)==len(b)==len(c)==r['candidate_count'] and all(math.isfinite(v) for v in a+b+c),'finite complete arrays')
            first=max(range(len(a)),key=a.__getitem__);check(first==r['argmax_index']==r['selected_index'],'actual exact argmax')
            for other,key in ((b,'max_chunk_error'),(c,'max_forward_error')):
                error=max(abs(x-y) for x,y in zip(a,other));maxerror=max(maxerror,error)
                check(error==r[key] and error<=1e-6,'independent numeric fidelity')
                check(max(range(len(other)),key=other.__getitem__)==first,'exact ranking fidelity')
            check(r['status']=='PASS' and not r['chosen_finish_encoding_collision'],'decision pass')
            finish=r['finish_indexes'];is_finish=bool(finish);miss=is_finish and first not in finish
            check(r['chosen_finishes']==(first in finish),'chosen finish')
            check(r['saturated_count']==sum(abs(v)>=.99 for v in a),'saturated values')
            check(r['spread']==max(a)-min(a),'score spread')
            check(r['finish_score_gap']==(max(a)-max(a[i] for i in finish) if finish else None),'finish preference gap')
            j=jobs.setdefault(r['job'],dict(counts=Counter(),spreads=[],missed_finish_gaps=[]))
            j['counts'].update(dict(states=1,candidates=len(a),finish_states=int(is_finish),missed_finish=int(miss),
                saturated=r['saturated_count'],other_states=int(not is_finish)))
            j['spreads'].append(r['spread'])
            if miss:j['missed_finish_gaps'].append(r['finish_score_gap'])
    for job,j in jobs.items():
        counts=j['counts'];src=report['jobs'][job]
        check(counts['states']==src['states_scored'] and counts['candidates']==src['candidates_scored'],'score coverage aggregate')
        check(counts['finish_states']==src['finish_opportunities'] and counts['other_states']==128,'selection coverage')
        j['median_spread']=statistics.median(j.pop('spreads'))
        gaps=j.pop('missed_finish_gaps');j['median_missed_finish_gap']=statistics.median(gaps) if gaps else None
        j['counts']=dict(counts)
    tp=read(target/'preregistration.json');tr=read(target/'report.json');multiset=read(target/'target-multiset.json')
    for name,value in tp['source_sha256'].items():check(digest(ROOT/name)==value,'target provenance')
    check(tr['status']=='PASS' and tr['actual_input_target_multiset_matched'],'target tracing')
    total=sum(r['count'] for r in multiset);positive=sum(r['count'] for r in multiset if r['reward']==1)
    check(all(r['reward'] in (-1,1) and r['count']>0 for r in multiset),'target values')
    check(total==tr['actual_samples']==tr['wave']['samples']==sum(h['samples'] for h in tr['wave']['hands']),'target cardinality')
    check(positive==sum(b['positive'] for b in tr['batches']),'positive label count')
    check(len(tr['gradient_batches'])==tr['wave']['updates'] and sum(b['samples'] for b in tr['gradient_batches'])==total,'gradient coverage')
    check(all(b['device']=='cuda:0' and b['max_abs_error']<=1e-6 for b in tr['gradient_batches']),'gradient formula')
    old=read(ROOT/'artifacts/evaluations/p3e-scale-v2/delivery-receipt.json')
    for name,value in old['artifact_sha256'].items():check(digest(ROOT/name)==value,f'P3e delivery {name}')
    return dict(status='PASS',score_states=len(seen),scored_candidates=sum(j['counts']['candidates'] for j in jobs.values()),
        jobs=jobs,max_score_error=maxerror,target_samples=total,target_positive=positive,target_updates=tr['wave']['updates'],
        p3e_receipt_files=len(old['artifact_sha256']),review_sha256=digest(Path(__file__)),
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for folder in (score,target)
                      for p in sorted(folder.iterdir()) if p.is_file()},
        scope='raw score and diagnostic evidence review; no global correctness or strength claim')

if __name__=='__main__':
    result=review();out=ROOT/'artifacts/evaluations/p3f-diagnostic-controller.json';write(out,result)
    print(json.dumps(result,ensure_ascii=False))
