"""Read-only P5b training corpus diagnostic; no optimization or new games."""
from pathlib import Path
import sys, os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse, gzip, json, math, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from experiments.p3e_common import check, read, write, digest
from guandan.env import HandEnv
from guandan.types import Action

BASE=ROOT/'artifacts/evaluations/p5b-pool-v1'
JOBS=[f'{arm}-{seed}' for seed in (314510,314511,314512) for arm in ('selfplay','mixed')]
SOURCES=['docs/P5C_PROTOCOL.md','experiments/p5c_diagnostic.py','experiments/controller_p5c_audit.py',
         'tests/test_p5c_diagnostic.py','scripts/p5c_test.py']

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def identity(obs,legal):
    return sha256(canonical([asdict(obs),[a.to_dict() for a in legal]]).encode()).hexdigest()

def categories(obs, legal):
    passes=[i for i,a in enumerate(legal) if a.kind=='pass']
    plays=[i for i,a in enumerate(legal) if a.kind!='pass']
    finish=[i for i,a in enumerate(legal) if a.kind!='pass' and len(a.cards)==len(obs.hand)]
    return passes,plays,finish

def metrics(scores,passes,plays,finish):
    values=[float(v) for v in scores]; n=len(values)
    check(n>0 and all(math.isfinite(v) for v in values),'finite nonempty scores')
    check(sorted(passes+plays)==list(range(n)) and len(set(passes+plays))==n,'complete partition')
    check(len(set(finish))==len(finish) and set(finish)<=set(plays),'finish subset')
    index=max(range(n),key=values.__getitem__)
    nonfinish=[i for i in range(n) if i not in finish]
    return dict(candidates=n,argmax=index,optional=bool(passes and plays),
        optional_pass=bool(passes and plays and index in passes),
        finish_opportunity=bool(finish),finish_miss=bool(finish and index not in finish),
        pass_margin=max(values[i] for i in passes)-max(values[i] for i in plays) if passes and plays else None,
        finish_margin=max(values[i] for i in finish)-max(values[i] for i in nonfinish) if finish and nonfinish else None)

def rebuild_hand(job,n,h):
    arm=job.split('-')[0]; patterns=(('current','greedy','current','greedy'),('current','team','greedy','team'),
        ('current','frozen','team','frozen'),('current','greedy','frozen','team'))
    expected=['current']*4 if arm=='selfplay' else [patterns[(n//4)%4][(s-n%4)%4] for s in range(4)]
    check(h['roles']==expected,'role schedule')
    check((h['seed'],h['level'],h['starting_player'])==(100000+n,2+n%13,n%4),'deal schedule')
    env=HandEnv();env.reset(h['seed'],initial_level=h['level'],starting_player=h['starting_player'])
    check(h['replay']['initial_digest']==env.state_digest(),'initial digest')
    check(h['replay']['initial_hands']==[list(x) for x in env.state.initial_hands],'initial hands')
    counts=Counter(hands=1,current_seat_hands=expected.count('current'))
    selected=[];seen=set();current=0
    check(len(h['decisions'])==len(h['replay']['steps'])==h['steps'],'complete hand')
    for step,(d,s) in enumerate(zip(h['decisions'],h['replay']['steps'],strict=True)):
        obs=env.observe(s['player']);legal=env.legal_actions(obs.player_id);a=Action.from_dict(s['action'])
        check(type(d['selected']) is int and 0<=d['selected']<len(legal),'selection range')
        check(d['player']==obs.player_id and d['role']==expected[obs.player_id],'actor')
        check(d['candidates']==len(legal) and legal[d['selected']]==a,'full candidate selection')
        check(type(d['sample']) is bool and d['sample']==(d['role']=='current'),'sample ownership')
        counts['decisions']+=1
        if d['sample']:
            current+=1; counts['samples']+=1
            reward=h['team_rewards'][obs.player_id%2];check(reward in (-1,1),'reward domain')
            label='positive' if reward==1 else 'negative';kind='pass' if a.kind=='pass' else 'play'
            counts[label]+=1;counts[kind]+=1;counts[kind+'_'+label]+=1
            passes,plays,finish=categories(obs,legal)
            optional=bool(passes and plays)
            counts['optional']+=optional;counts['optional_pass']+=optional and a.kind=='pass'
            counts['forced_pass']+=bool(passes and not plays)
            counts['finish_opportunity']+=bool(finish);counts['finish_miss']+=bool(finish and d['selected'] not in finish)
            flags=[]
            for key,condition in [('first_current',True),('first_optional',optional),('first_finish',bool(finish))]:
                if condition and key not in seen:seen.add(key);flags.append(key)
            if n%400<16 and flags:
                selected.append((dict(id=f'{job}/{n}/{step}',source=job,n=n,step=step,flags=flags,
                    identity=identity(obs,legal),passes=passes,plays=plays,finish=finish),obs,legal))
        env.step(obs.player_id,a,state_version=s['state_version'])
        check(env.state_digest()==s['digest'],'step digest')
    check(env.state.terminal and env.state_digest()==h['terminal_digest'],'terminal digest')
    check(list(env.state.settlement.team_rewards)==h['team_rewards'],'replayed rewards')
    check(current==h['samples'],'hand sample count')
    counts['current_seat_wins']=sum(h['team_rewards'][s%2]==1 for s in range(4) if expected[s]=='current')
    return dict(job=job,n=n,quarter=n//400,block=(n//4)%4,**dict(counts)),selected

def process_job(job):
    hands=[];states=[];waves=0
    for wi,w in enumerate(rows(BASE/'training'/job/'waves.jsonl.gz')):
        check(w['wave']==wi+1 and len(w['hands'])==4,'wave sequence')
        for j,h in enumerate(w['hands']):
            row,chosen=rebuild_hand(job,4*wi+j,h);hands.append(row);states.extend(chosen)
        check(sum(x['samples'] for x in hands[-4:])==w['samples'],'wave samples');waves+=1
    check(waves==400 and len(hands)==1600,'fixed corpus size')
    return hands,states

def aggregate(hands,scores):
    training=defaultdict(Counter);scoring=defaultdict(Counter)
    for h in hands:
        for key in (h['job'],f"{h['job']}/quarter-{h['quarter']}",f"{h['job']}/block-{h['block']}"):
            training[key].update({k:v for k,v in h.items() if k not in ('job','n','quarter','block')})
    for row in scores:
        m=row['metrics'];key=row['source']+'|'+row['model'];c=scoring[key];c['states']+=1
        for name in ('candidates','optional','optional_pass','finish_opportunity','finish_miss'):c[name]+=m[name]
        for name in ('pass_margin','finish_margin'):
            if m[name] is not None:c[name+'_sum']+=m[name];c[name+'_count']+=1
    return dict(training={k:dict(v) for k,v in sorted(training.items())},scoring={k:dict(v) for k,v in sorted(scoring.items())})

def protected():
    # Hash every historical source/input/immutable artifact required by prior acceptance.
    from experiments.controller_p5b_audit import freeze,historical
    freeze(BASE);history=historical()
    receipt=read(BASE/'delivery-receipt.json');count=0
    for key in ('artifact_sha256',):
        for path,h in receipt[key].items():
            if path=='docs/STATUS.md':continue
            check(digest(ROOT/path)==h,'P5b delivery '+path);count+=1
    return dict(history=history,p5b_bindings=count)

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();out=args.output.resolve()
    check(not out.exists(),'output must be fresh');history=protected()
    inputs={}
    for job in JOBS:
        for name in ('waves.jsonl.gz','final/checkpoint.pt','final/manifest.json'):
            path=BASE/'training'/job/name;inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    sources={name:digest(ROOT/name) for name in SOURCES};out.mkdir(parents=True)
    write(out/'preregistration.json',dict(version='gd-p5c-pool-diagnostic-v1',jobs=JOBS,inputs=inputs,sources=sources,historical=history))
    with zipfile.ZipFile(out/'source-snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES:z.write(ROOT/name,name)
    hands=[];states=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for job,(h,s) in zip(JOBS,pool.map(process_job,JOBS),strict=True):
            hands.extend(h);states.extend(s);print(f'replayed {job}: {len(h)} hands, {len(s)} states',flush=True)
    write(out/'hands.json',hands);write(out/'states.json',[s[0] for s in states])
    import torch
    from guandan.learning.model import DMCNetwork
    from guandan_gpu.training import score_many
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    allrows=[]
    with gzip.open(out/'scores.jsonl.gz','wt',encoding='utf-8') as f:
        for job in JOBS:
            model=DMCNetwork().cuda();payload=torch.load(BASE/'training'/job/'final/checkpoint.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(payload['model']);model.eval()
            for meta,obs,legal in states:
                values=score_many(model,[(obs,legal)],1024)[0].tolist()
                m=metrics(values,meta['passes'],meta['plays'],meta['finish'])
                row=dict(id=meta['id'],source=meta['source'],model=job,metrics=m,scores=values)
                f.write(canonical(row)+'\n');allrows.append({k:v for k,v in row.items() if k!='scores'})
            print(f'scored {job}: {len(states)} complete states',flush=True)
    summary=aggregate(hands,allrows)
    write(out/'summary.json',dict(status='COMPUTED_PENDING_INDEPENDENT_AUDIT',hands=len(hands),states=len(states),
        source_hands=384,unique_deal_seeds=64,score_rows=len(allrows),new_training_hands=0,new_evaluation_games=0,
        model_promoted=False,**summary))
    for name,h in {**inputs,**sources}.items():check(digest(ROOT/name)==h,'immutable input/source')
    protected()
    write(out/'receipt.json',dict(status='COMPUTED_PENDING_INDEPENDENT_AUDIT',artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':main()
