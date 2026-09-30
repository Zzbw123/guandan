"""Controller reconstruction and arithmetic audit, independent of probe selection loop."""
from pathlib import Path
import sys,json,gzip,math,csv,statistics,copy
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.p3e_common import read,write,digest,check
from experiments.p3k_run import OUT,paths,hashvalue,bindings,setup,load_models
from guandan.env import HandEnv
from guandan.types import Action
from guandan.agents import GreedyAgent

def reconstruct():
    result=[]
    for source,path in paths()[1].items():
        teacher=source.startswith('teacher')
        identifiers=[str(s) if teacher else f'dmc|dmc|greedy:{s}:0:0:0'
            for s in (range(100000,100013) if teacher else range(108100,108113))]
        found={}
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);key=str(r['seed']) if teacher else r['trial_id']
                if key in identifiers:
                    check(key not in found,'unique source game');found[key]=r['replay']
        check(set(found)==set(identifiers),'all 13 fixed source games')
        for key in identifiers:
            replay=found[key];env=HandEnv.from_hands(replay['initial_hands'],replay['initial_level'],replay['starting_player'])
            check(env.state_digest()==replay['initial_digest'],'initial state')
            records=[]
            for i,s in enumerate(replay['steps']):
                obs=env.observe(s['player']);a=Action.from_dict(s['action'])
                if teacher or s['player']%2==0:
                    legal=env.legal_actions(s['player']);check(a in legal,'executed action legal')
                    records.append((i,obs,legal,a,s['action']))
                env.step(s['player'],a,state_version=s['state_version'])
                check(env.state_digest()==s['digest'],'each replay state digest')
            check(env.state.terminal and env.state_digest()==replay['final_digest'],'final state')
            n=len(records);check(n>=4,'enough source states')
            for index in ((n-1)//3,(2*n-2)//3):
                i,obs,legal,a,ad=records[index]
                identity=dict(source=source,game=key,seed=int(key) if teacher else int(key.split(':')[1]),
                    step=i,player=obs.player_id,level=obs.level,reward=env.state.settlement.team_rewards[obs.player_id%2],
                    eligible_count=n,candidate_count=len(legal),candidate_sha256=hashvalue([v.to_dict() for v in legal]),
                    observation_sha256=hashvalue(asdict(obs)),action=ad,final_digest=replay['final_digest'])
                result.append((identity,obs,legal,a))
    return result

def scalar_close(a,b,msg,rtol=1e-9,atol=1e-10):
    if a is None or b is None:check(a is b,msg);return
    check(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=rtol,abs_tol=atol),msg)

def validate_rows(rows,selection,models):
    expected={(r['source'],r['game'],r['step']):r for r in selection};seen=set()
    check(len(selection)==234 and len(expected)==234 and len(rows)==2106,'coverage counts')
    for row in rows:
        identity=row['identity'];key=(identity['source'],identity['game'],identity['step'])
        check(key in expected and identity==expected[key],'reconstructed identity, reward, full candidates')
        job=row['model'];probe=(key,job);check(job in models and probe not in seen,'unique fixed model/state');seen.add(probe)
        m=row['metrics'];q=row['scores'];t=row['targets'];n=identity['candidate_count']
        check(n==m['candidate_count']==len(q)==len(t),'complete output denominator')
        check(all(math.isfinite(x) and -1.000001<=x<=1.000001 for x in q+t),'finite bounded outputs')
        g=m['gram'];check(len(g)==3 and all(len(v)==3 for v in g),'Gram shape')
        for i in range(3):
            check(g[i][i]>=0 and math.isfinite(g[i][i]),'nonnegative squared norm')
            for j in range(3):
                scalar_close(g[i][j],g[j][i],'Gram symmetry')
                scalar_close(g[i][j],sum(v[i][j] for v in m['parameter_grams'].values()),'parameter sum',atol=1e-7)
                check(g[i][j]**2<=g[i][i]*g[j][j]+1e-8*max(1.,g[i][i]*g[j][j]),'Gram Cauchy bound')
        d=math.sqrt(g[0][0]);scalar_close(m['geometry']['dmc_norm'],d,'DMC norm')
        for i,name in ((1,'absolute'),(2,'centered')):
            a=math.sqrt(g[i][i]);v=m['geometry'][name]
            cosine=None if min(a,d)<=1e-12 else max(-1.,min(1.,g[0][i]/(a*d)))
            scalar_close(v['norm'],a,'aux norm');scalar_close(v['cosine'],cosine,'cosine')
            scalar_close(v['weighted_norm_ratio'],None if d<=1e-12 else .1*a/d,'weighted ratio')
            scalar_close(v['dmc_descent_factor'],None if d<=1e-12 else 1+.1*g[0][i]/g[0][0],'descent factor')
            check(v['conflict']==(None if cosine is None else cosine<0),'conflict flag')
        residual=[a-b for a,b in zip(q,t)];mu=statistics.mean(residual)
        scalar_close(m['losses']['absolute'],statistics.mean(x*x for x in residual),'complete absolute loss',rtol=1e-5,atol=2e-6)
        scalar_close(m['losses']['centered'],statistics.mean((x-mu)**2 for x in residual),'complete centered loss',rtol=1e-5,atol=2e-6)
        chosen=q.index(max(q));qt=sorted(q,reverse=True);tt=sorted(t,reverse=True)
        exact=dict(singleton=n==1,teacher_top_gap=None if n==1 else tt[0]-tt[1],teacher_top_ties=sum(x==tt[0] for x in t),
            model_top_gap=None if n==1 else qt[0]-qt[1],model_span=max(q)-min(q),top1=int(t[chosen]==max(t)),
            teacher_regret=max(t)-t[chosen],saturation_fraction=sum(abs(x)>=.95 for x in q)/n,
            tanh_slope_mean=statistics.mean(1-x*x for x in q))
        for k,v in exact.items():scalar_close(m[k],v,'score-derived '+k)
        scalar_close(m['losses']['dmc'],(m['executed_q']-identity['reward'])**2,'DMC loss/terminal label',rtol=1e-5,atol=2e-6)
        if n==1:check(m['losses']['centered']==0 and g[2][2]==0,'singleton centered zero')
    check(len(seen)==len(expected)*len(models),'complete model Cartesian product')

def summarize(rows):
    result=[]
    for source in paths()[1]:
        for model in paths()[0]:
            cell=[r['metrics'] for r in rows if r['identity']['source']==source and r['model']==model]
            check(len(cell)==26,'26 observations per source/model')
            for subset in ('all','multi','singleton'):
                part=[m for m in cell if subset=='all' or (subset=='singleton')==m['singleton']]
                base=dict(source=source,model=model,subset=subset,states=len(part))
                quantities={k:[r[k] for r in part] for k in ('model_span','model_top_gap','teacher_top_gap','top1','teacher_regret','saturation_fraction','tanh_slope_mean')}
                quantities.update({'loss_'+k:[r['losses'][k] for r in part] for k in ('dmc','absolute','centered')})
                for arm in ('absolute','centered'):
                    for k in ('cosine','weighted_norm_ratio','dmc_descent_factor'):
                        quantities[arm+'_'+k]=[r['geometry'][arm][k] for r in part]
                    valid=[r['geometry'][arm]['cosine'] for r in part if r['geometry'][arm]['cosine'] is not None]
                    desc=[r['geometry'][arm]['dmc_descent_factor'] for r in part if r['geometry'][arm]['dmc_descent_factor'] is not None]
                    for label,vals in (('conflict',[x<0 for x in valid]),('strong_conflict',[x<-.1 for x in valid]),('reverse_dmc',[x<0 for x in desc])):
                        result.append(dict(**base,metric=arm+'_'+label,valid=len(vals),nulls=len(part)-len(vals),
                            mean=statistics.mean(vals) if vals else None,median=statistics.median(vals) if vals else None,
                            minimum=min(vals) if vals else None,maximum=max(vals) if vals else None))
                for k,values in quantities.items():
                    vals=[x for x in values if x is not None]
                    result.append(dict(**base,metric=k,valid=len(vals),nulls=len(part)-len(vals),mean=statistics.mean(vals) if vals else None,
                        median=statistics.median(vals) if vals else None,minimum=min(vals) if vals else None,maximum=max(vals) if vals else None))
    return result

def negative(rows,selection,models,receipt):
    cases={}
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,IndexError):cases[name]='REJECTED';return
        raise ValueError('negative case accepted: '+name)
    for name,mut in (
        ('dropped_probe',lambda r:r.pop()),('duplicated_probe',lambda r:r.__setitem__(1,copy.deepcopy(r[0]))),
        ('wrong_reward',lambda r:r[0]['identity'].__setitem__('reward',-r[0]['identity']['reward'])),
        ('dropped_candidate',lambda r:r[0]['scores'].pop()),
        ('changed_cosine',lambda r:r[0]['metrics']['geometry']['absolute'].__setitem__('cosine',.123)),
        ('changed_gram',lambda r:r[0]['metrics']['gram'][0].__setitem__(0,0.)),
        ('changed_state',lambda r:r[0]['identity'].__setitem__('step',9999)),
        ('wrong_model',lambda r:r[0].__setitem__('model','unregistered'))):
        edited=copy.deepcopy(rows);mut(edited);reject(name,lambda:validate_rows(edited,selection,models))
    altered=dict(receipt,model_promoted=True)
    reject('fake_promotion',lambda:check(altered['model_promoted'] is False,'no promotion'))
    return dict(status='PASS',cases=cases)

def main():
    import torch
    from experiments.p3k_core import oracle
    pre=bindings(OUT);receipt=read(OUT/'run-receipt.json')
    check(receipt['model_promoted'] is receipt['reserved_test_executed'] is False,'scope')
    check(receipt['new_training_hands']==receipt['new_evaluation_games']==0,'read-only scope')
    for file,key in (('selection.json','selection_sha256'),('probes.jsonl','probes_sha256'),('preregistration.json','preregistration_sha256')):
        check(digest(OUT/file)==receipt[key],'receipt binding')
    setup();nets=load_models();selected=reconstruct();selection=[r[0] for r in selected]
    check(selection==read(OUT/'selection.json'),'independent complete selection')
    with (OUT/'probes.jsonl').open(encoding='utf-8') as f:rows=[json.loads(s) for s in f]
    validate_rows(rows,selection,list(nets));bykey={(r['identity']['source'],r['identity']['game'],r['identity']['step'],r['model']):r for r in rows}
    count=0;maxq=maxgram=0.
    for identity,obs,legal,action in selected:
        raw=[float(GreedyAgent().score(obs,a)) for a in legal];lo=min(raw);span=max(raw)-lo
        target=[0. if span==0 else 1.6*(x-lo)/span-.8 for x in raw]
        for name,net in nets.items():
            row=bykey[(identity['source'],identity['game'],identity['step'],name)];g,loss,q,t=oracle(net,obs,legal,action,identity['reward'])
            for a,b in zip(row['targets'],target):scalar_close(a,b,'independent teacher target',atol=1e-7)
            qerr=max(abs(a-b) for a,b in zip(q.cpu().tolist(),row['scores']));maxq=max(maxq,qerr);check(qerr<=2e-6,'controller full q')
            for i,a in enumerate(g):
                for j,b in enumerate(g):
                    value=torch.dot(a.double(),b.double()).item();old=row['metrics']['gram'][i][j]
                    maxgram=max(maxgram,abs(value-old));scalar_close(value,old,'controller Gram',rtol=1e-3,atol=1e-5)
            for k,v in zip(('dmc','absolute','centered'),loss):scalar_close(v,row['metrics']['losses'][k],'controller loss',rtol=1e-5,atol=2e-6)
            ix=legal.index(action);scalar_close(row['metrics']['executed_q'],row['scores'][ix],'executed score',atol=2e-6)
            scalar_close(row['metrics']['executed_teacher_target'],row['targets'][ix],'executed teacher score')
            count+=1
        if count%234==0:print(json.dumps(dict(audited_probes=count)),flush=True)
    check(count==2106 and receipt['dense_gradient_comparisons']==6318,'all gradient checks')
    check(receipt['model_before']==receipt['model_after'],'weights preserved')
    summaries=summarize(rows);folder=OUT/'outputs/tables';folder.mkdir(parents=True,exist_ok=False)
    for p in (folder/'source-model-summary.csv',OUT/'outputs/metrics.csv'):
        with p.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    write(OUT/'summary.json',summaries)
    write(OUT/'controller-negative.json',negative(rows,selection,list(nets),receipt))
    bindings(OUT)
    result=dict(status='PASS',engineering='ACCEPTED_FIXED_WEIGHT_GRADIENT_DIAGNOSTIC_V1',games=117,states=len(selected),
        candidates=sum(len(r[2]) for r in selected),singleton_states=sum(len(r[2])==1 for r in selected),
        probes=count,independent_dense_probes=count,max_score_abs_error=maxq,max_gram_abs_error=maxgram,
        model_promoted=False,reserved_test_executed=False,overall_P3='PARTIALLY_ACCEPTED',
        source_sha256=pre['source_sha256'],run_receipt_sha256=digest(OUT/'run-receipt.json'),summary_sha256=digest(OUT/'summary.json'))
    write(OUT/'controller-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)

if __name__=='__main__':main()
