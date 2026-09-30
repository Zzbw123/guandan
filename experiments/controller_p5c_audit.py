"""Controller recomputation independent of the diagnostic's selection/metric helpers."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gzip,json,math,zipfile
from collections import Counter,defaultdict
from dataclasses import asdict
from hashlib import sha256
from experiments.p3e_common import check,read,write,digest
from guandan.env import HandEnv
from guandan.types import Action

BASE=ROOT/'artifacts/evaluations/p5b-pool-v1'
JOBS=[f'{a}-{s}' for s in (314510,314511,314512) for a in ('selfplay','mixed')]

def lines(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def verify_scores(row,values,legal,hand_size):
    import torch
    actual=torch.tensor(row['scores'],dtype=torch.float32);expected=torch.as_tensor(values,dtype=torch.float32).cpu()
    check(actual.shape==expected.shape and actual.numel()==len(legal),'score length')
    check(bool(torch.isfinite(actual).all()),'finite score')
    torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-6)
    check(actual.argmax().item()==expected.argmax().item(),'direct argmax')
    # Recompute saved metrics from the saved scores, after independently verifying their values.
    v=actual.tolist();index=max(range(len(v)),key=lambda i:v[i]);pa=[];pl=[];fi=[]
    for i,a in enumerate(legal):
        (pa if a.kind=='pass' else pl).append(i)
        if a.kind!='pass' and len(a.cards)==hand_size:fi.append(i)
    other=[i for i in range(len(v)) if i not in fi]
    m=dict(candidates=len(v),argmax=index,optional=bool(pa and pl),optional_pass=bool(pa and pl and index in pa),
        finish_opportunity=bool(fi),finish_miss=bool(fi and index not in fi),
        pass_margin=max(v[i] for i in pa)-max(v[i] for i in pl) if pa and pl else None,
        finish_margin=max(v[i] for i in fi)-max(v[i] for i in other) if fi and other else None)
    check(row['metrics']==m,'metric recomputation')
    return m,float((actual-expected).abs().max())

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();out=args.output.resolve()
    check(not (out/'controller-audit.json').exists(),'fresh audit output')
    pre=read(out/'preregistration.json');receipt=read(out/'receipt.json')
    check(pre['jobs']==JOBS,'job set')
    for group in ('inputs','sources'):
        for name,h in pre[group].items():check(digest(ROOT/name)==h,'frozen '+name)
    for name,h in receipt['artifacts'].items():check(digest(out/name)==h,'output binding '+name)
    with zipfile.ZipFile(out/'source-snapshot.zip') as z:
        check(set(z.namelist())==set(pre['sources']),'source archive coverage')
        for name,h in pre['sources'].items():check(sha256(z.read(name)).hexdigest()==h,'archive bytes')
    saved_hands=read(out/'hands.json');saved_states=read(out/'states.json');summary=read(out/'summary.json')
    check(len(saved_hands)==9600 and len({(x['job'],x['n']) for x in saved_hands})==9600,'hand coverage')
    hand_index={(x['job'],x['n']):x for x in saved_hands};reconstructed={};metadata=[];replay_count=0
    for job in JOBS:
        n=0
        for wave in lines(BASE/'training'/job/'waves.jsonl.gz'):
            for h in wave['hands']:
                c=Counter(hands=1,current_seat_hands=h['roles'].count('current'))
                c['current_seat_wins']=sum(h['team_rewards'][s%2]==1 for s in range(4) if h['roles'][s]=='current')
                for d,s in zip(h['decisions'],h['replay']['steps'],strict=True):
                    c['decisions']+=1
                    if h['roles'][s['player']]=='current':
                        check(d['sample'] is True,'independent sample ownership')
                        c['samples']+=1;label='positive' if h['team_rewards'][s['player']%2]==1 else 'negative'
                        kind='pass' if s['action']['kind']=='pass' else 'play'
                        c[label]+=1;c[kind]+=1;c[kind+'_'+label]+=1
                for key,v in c.items():check(hand_index[job,n].get(key,0)==v,'full-corpus label count '+key)
                if n%400<16:
                    env=HandEnv();env.reset(100000+n,initial_level=2+n%13,starting_player=n%4)
                    seen=set();behavior=Counter()
                    for step,s in enumerate(h['replay']['steps']):
                        obs=env.observe(s['player']);legal=env.legal_actions(s['player']);a=Action.from_dict(s['action'])
                        if h['roles'][s['player']]=='current':
                            pa=[i for i,x in enumerate(legal) if x.kind=='pass'];pl=[i for i,x in enumerate(legal) if x.kind!='pass']
                            fi=[i for i in pl if len(legal[i].cards)==len(obs.hand)]
                            behavior['optional']+=bool(pa and pl);behavior['optional_pass']+=bool(pa and pl and a.kind=='pass')
                            behavior['forced_pass']+=bool(pa and not pl)
                            behavior['finish_opportunity']+=bool(fi);behavior['finish_miss']+=bool(fi and a not in [legal[i] for i in fi])
                            flags=[]
                            for label,cond in [('first_current',True),('first_optional',bool(pa and pl)),('first_finish',bool(fi))]:
                                if cond and label not in seen:seen.add(label);flags.append(label)
                            if flags:
                                identifier=f'{job}/{n}/{step}'
                                value=[asdict(obs),[x.to_dict() for x in legal]]
                                identity=sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
                                m=dict(id=identifier,source=job,n=n,step=step,flags=flags,identity=identity,passes=pa,plays=pl,finish=fi)
                                metadata.append(m);reconstructed[identifier]=(obs,legal)
                        env.step(s['player'],a,state_version=s['state_version']);check(env.state_digest()==s['digest'],'selected-hand replay digest')
                    check(env.state.terminal and list(env.state.settlement.team_rewards)==h['team_rewards'],'selected-hand rewards')
                    for key,v in behavior.items():check(hand_index[job,n].get(key,0)==v,'independent behavior '+key)
                    replay_count+=1
                n+=1
        check(n==1600,'job hand count');print(f'independent labels and selected replays: {job}',flush=True)
    check(metadata==saved_states and replay_count==384,'exact state selection/order/identity')
    # Independently sum every published training cell; detailed behavior was independently replayed on 384 hands.
    train={}
    for key in summary['training']:
        job=key.split('/')[0];part=key.split('/')[1] if '/' in key else None
        subset=[h for h in saved_hands if h['job']==job and (part is None or h[part.split('-')[0]]==int(part.split('-')[1]))]
        fields=set().union(*(set(h)-{'job','n','quarter','block'} for h in subset))
        train[key]={field:sum(h.get(field,0) for h in subset) for field in fields}
    check(train==summary['training'],'independent training aggregation')
    import torch
    from guandan.learning.model import DMCNetwork
    from experiments.controller_p5a_audit import full_scores
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=None;current=None;seen=set();groups=defaultdict(list);max_error=0.;total_candidates=0;score_count=0
    for row in lines(out/'scores.jsonl.gz'):
        check(row['model'] in JOBS and row['id'] in reconstructed,'score identity')
        check(row['source']==row['id'].split('/')[0],'score source identity')
        pair=(row['model'],row['id']);check(pair not in seen,'duplicate score');seen.add(pair)
        if current!=row['model']:
            current=row['model'];model=DMCNetwork().cuda()
            model.load_state_dict(torch.load(BASE/'training'/current/'final/checkpoint.pt',map_location='cpu',weights_only=True)['model']);model.eval()
            print('independent direct forward: '+current,flush=True)
        obs,legal=reconstructed[row['id']];expected=full_scores(model,obs,legal)
        m,error=verify_scores(row,expected,legal,len(obs.hand));max_error=max(max_error,error)
        groups[row['source']+'|'+current].append(m);total_candidates+=len(legal);score_count+=1
    check(len(seen)==6*len(metadata),'six-model Cartesian coverage')
    aggregated={}
    for key,ms in groups.items():
        c=dict(states=len(ms))
        for k in ('candidates','optional','optional_pass','finish_opportunity','finish_miss'):c[k]=sum(m[k] for m in ms)
        for k in ('pass_margin','finish_margin'):
            values=[m[k] for m in ms if m[k] is not None]
            if values:c[k+'_sum']=sum(values);c[k+'_count']=len(values)
        aggregated[key]=c
    check(aggregated==summary['scoring'],'independent score aggregation')
    check(summary['hands']==9600 and summary['states']==len(metadata) and summary['score_rows']==score_count,'summary coverage')
    check(summary['new_training_hands']==summary['new_evaluation_games']==0 and summary['model_promoted'] is False,'scope')
    from experiments.p5c_diagnostic import protected
    protected()
    for group in ('inputs','sources'):
        for name,h in pre[group].items():check(digest(ROOT/name)==h,'post-audit immutable '+name)
    for name,h in receipt['artifacts'].items():check(digest(out/name)==h,'post-audit output '+name)
    write(out/'controller-audit.json',dict(status='PASS',scope='ACCEPTED_POOL_COLLAPSE_DIAGNOSTIC_V1',
        full_corpus_label_hands=9600,independently_replayed_hands=replay_count,states=len(metadata),
        independently_scored_rows=score_count,independently_scored_candidates=total_candidates,max_abs_error=max_error,
        new_training_hands=0,new_evaluation_games=0,model_promoted=False,preregistration_sha256=digest(out/'preregistration.json')))
    print('PASS independent diagnostic audit',flush=True)

if __name__=='__main__':main()
