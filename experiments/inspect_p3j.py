"""Fixed 128-state teacher-fidelity diagnostic with independent direct-forward audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
import gzip
import json
from dataclasses import asdict
from hashlib import sha256
import math
from experiments.p3e_common import read,write,digest,check
from experiments.p3j_protocol import INITS,ARMS
from guandan.env import HandEnv
from guandan.types import Action
from guandan.agents import GreedyAgent
from guandan.learning.encoding import encode_observation,encode_action

def states():
    path=ROOT/'artifacts/evaluations/p3f-teacher-v1/training/teacher-314380/phase1/replays.jsonl.gz'
    n=0
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);env=HandEnv();env.reset(r['seed'],initial_level=r['level'],starting_player=r['starting_player'])
            for stepno,s in enumerate(r['replay']['steps']):
                obs=env.observe(s['player']);legal=env.legal_actions(obs.player_id)
                if len(legal)>1:
                    yield dict(seed=r['seed'],step=stepno,player=obs.player_id),obs,legal
                    n+=1
                    if n==128:return
                env.step(obs.player_id,Action.from_dict(s['action']),state_version=s['state_version'])
    raise ValueError('insufficient diagnostic observations')

def candidate_hash(legal):
    return sha256(json.dumps([a.to_dict() for a in legal],sort_keys=True,separators=(',',':')).encode()).hexdigest()

def metrics(scores,raw):
    check(len(scores)==len(raw) and len(raw)>1 and all(math.isfinite(x) for x in scores+raw),'finite complete score arrays')
    lo,hi=min(raw),max(raw);i=max(range(len(scores)),key=scores.__getitem__)
    target=[0.]*len(raw) if lo==hi else [1.6*(x-lo)/(hi-lo)-.8 for x in raw]
    return dict(top1=int(raw[i]==hi),regret=0. if lo==hi else (hi-raw[i])/(hi-lo),
        mse=sum((a-b)**2 for a,b in zip(scores,target))/len(raw))

def models(root):
    from experiments.p3j_training import GPUTrainer
    from experiments.p3j_checkpoint import load_checkpoint
    result={};pins={}
    for seed in INITS:
        for arm in ARMS:
            job=f'{arm}-{seed}';path=root/'training'/job/'final';m=read(path/'manifest.json')
            GPUTrainer(m['config'])
            t=load_checkpoint(path,m['sha256']);t.model.eval()
            result[job]=t.model;pins[job]=m['sha256']
    return result,pins

def run(root):
    from guandan_gpu.training import score_many
    out=root/'fidelity';out.mkdir(exist_ok=False)
    nets,pins=models(root);rows=[]
    with gzip.open(out/'scores.jsonl.gz','xt',encoding='utf-8') as f:
        for identity,obs,legal in states():
            raw=[float(GreedyAgent().score(obs,a)) for a in legal]
            scores={job:score_many(net,[(obs,legal)],1024)[0].tolist() for job,net in nets.items()}
            row=dict(identity=identity,observation=asdict(obs),candidate_sha256=candidate_hash(legal),
                candidate_count=len(legal),teacher_scores=raw,scores=scores,
                metrics={job:metrics(v,raw) for job,v in scores.items()})
            f.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
    summary={job:{k:sum(r['metrics'][job][k] for r in rows)/len(rows) for k in ('top1','regret','mse')} for job in nets}
    write(out/'report.json',dict(status='PASS',states=len(rows),candidates=sum(r['candidate_count'] for r in rows),
        model_sha256=pins,summary=summary,scores_sha256=digest(out/'scores.jsonl.gz'),
        preregistration_sha256=digest(root/'preregistration.json'),script_sha256=digest(Path(__file__))))

def verify(root):
    import torch
    folder=root/'fidelity';report=read(folder/'report.json');nets,pins=models(root)
    check(pins==report['model_sha256'] and digest(folder/'scores.jsonl.gz')==report['scores_sha256'],'fidelity bindings')
    check(report['preregistration_sha256']==digest(root/'preregistration.json'),'fidelity preregistration')
    with gzip.open(folder/'scores.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(s) for s in f]
    selected=list(states());check(len(rows)==len(selected)==128,'fixed full state selection')
    totals={job:dict(top1=0.,regret=0.,mse=0.) for job in nets};maxerr=0.
    for row,(identity,obs,legal) in zip(rows,selected):
        check(row['identity']==identity and row['candidate_sha256']==candidate_hash(legal) and row['candidate_count']==len(legal),'reconstructed state and full candidate identity')
        check(row['observation']==json.loads(json.dumps(asdict(obs))),'visible observation identity')
        raw=[float(GreedyAgent().score(obs,a)) for a in legal]
        check(raw==row['teacher_scores'],'independent teacher scores')
        minimum=min(raw);maximum=max(raw);span=maximum-minimum
        expected=[0. if span==0 else (x-minimum)/span*1.6-.8 for x in raw]
        for job,net in nets.items():
            values=row['scores'][job];check(len(values)==len(legal) and all(math.isfinite(v) for v in values),'finite full model output')
            direct=[]
            with torch.no_grad():
                for start in range(0,len(legal),237):
                    part=legal[start:start+237]
                    st=torch.tensor([encode_observation(obs)]*len(part),device='cuda')
                    ac=torch.tensor([encode_action(a) for a in part],device='cuda')
                    direct.extend(net(st,ac).cpu().tolist())
            err=max(abs(x-y) for x,y in zip(values,direct));maxerr=max(maxerr,err);check(err<=2e-6,'independent direct forward')
            chosen=values.index(max(values));calc=dict(top1=int(raw[chosen]==maximum),
                regret=0. if span==0 else (maximum-raw[chosen])/span,
                mse=sum((v-t)*(v-t) for v,t in zip(values,expected))/len(legal))
            for k,v in calc.items():
                check(math.isclose(v,row['metrics'][job][k],abs_tol=1e-12),'independent fidelity metric')
                totals[job][k]+=v/128
    for job in nets:
        for k,v in totals[job].items():check(math.isclose(v,report['summary'][job][k],abs_tol=1e-12),'fidelity aggregate')
    result=dict(status='PASS',states=128,candidates=sum(len(v[2]) for v in selected),
        direct_forward_comparisons=128*6,max_score_error=maxerr,summary=totals,
        report_sha256=digest(folder/'report.json'),script_sha256=digest(Path(__file__)))
    if not (folder/'controller-audit.json').exists():write(folder/'controller-audit.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    root=Path(sys.argv[1]).resolve()
    if len(sys.argv)<3 or sys.argv[2]!='--verify':run(root)
    verify(root)
