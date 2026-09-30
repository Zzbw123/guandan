"""Independently reconstruct real exploration/shuffle order and batch denominators."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from random import Random
import gzip
import json
from hashlib import sha256
from p3e_common import check,read,write,digest
from p3j_protocol import specification
from guandan.env import HandEnv
from guandan.types import Action

def canonical_hash(v):return sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def review(root,job):
    import torch
    torch.set_num_threads(1)
    folder=root/'training'/job;out=root/'batch-order';out.mkdir(exist_ok=True)
    pins={name:digest(folder/name) for name in ('replays.jsonl.gz','waves.jsonl','final/checkpoint.pt')}
    provenance=dict(input_sha256=pins,script_sha256=digest(Path(__file__)))
    target=out/f'{job}.json'
    if target.exists():
        value=read(target);check(value['provenance']==provenance,'batch audit provenance drift');return value
    with gzip.open(folder/'replays.jsonl.gz','rt',encoding='utf-8') as f:replays=[json.loads(s) for s in f]
    waves=[json.loads(s) for s in (folder/'waves.jsonl').read_text('utf-8').splitlines() if s]
    arm,seedtext=job.split('-');rng=Random(int(seedtext));epsilon=.1
    orders=[];exploration_steps=observations=candidates=0
    check(len(replays)==600 and len(waves)==150,'fixed complete training set')
    for wave_index,wave in enumerate(waves):
        trajectories=[]
        for r in replays[4*wave_index:4*wave_index+4]:
            env=HandEnv();env.reset(r['seed'],initial_level=r['level'],starting_player=r['starting_player'])
            trajectory=[]
            for stepno,s in enumerate(r['replay']['steps']):
                legal=env.legal_actions(s['player']);chosen=Action.from_dict(s['action'])
                check(chosen in legal,'replayed action legality')
                trajectory.append(dict(identity=[r['seed'],stepno,s['player']],count=len(legal),
                    legal=legal,chosen=chosen))
                env.step(s['player'],chosen,state_version=s['state_version'])
            check(env.state.terminal and env.state_digest()==r['terminal_digest'],'complete replay')
            trajectories.append(trajectory)
        # Original sampler advances each of the four environments once per round.
        for step in range(max(map(len,trajectories))):
            for trajectory in trajectories:
                if step>=len(trajectory):continue
                row=trajectory[step]
                if rng.random()<epsilon:
                    pick=rng.randrange(row['count'])
                    check(row['legal'][pick]==row['chosen'],'independent actual exploration choice')
                    exploration_steps+=1
        samples=[dict(identity=row['identity'],count=row['count']) for trajectory in trajectories for row in trajectory]
        rng.shuffle(samples)
        check(len(samples)==wave['samples'],'shuffled observation coverage')
        records=[]
        for i,b in enumerate(wave['batches']):
            part=samples[256*i:256*(i+1)];size=sum(v['count'] for v in part)
            check(len(part)==b['samples'],'actual tail batch denominator')
            check(b['auxiliary_requests']==(len(part) if arm in ('absolute','centered') else 0),'actual auxiliary batch observations')
            check(b['auxiliary_candidates']==(size if arm in ('absolute','centered') else 0),'actual auxiliary batch complete candidate count')
            records.append(dict(observations=len(part),complete_candidates=size,expected_order_sha256=canonical_hash(part)))
        observations+=len(samples);candidates+=sum(v['count'] for v in samples)
        orders.append(dict(wave=wave_index+1,batches=records))
    payload=torch.load(folder/'final/checkpoint.pt',map_location='cpu',weights_only=True)
    check(rng.getstate()==payload['policy_rng'],'independent final policy RNG after every draw and shuffle')
    value=dict(status='PASS',job=job,observations=observations,candidates=candidates,exploration_steps=exploration_steps,
        batches=sum(len(w['batches']) for w in orders),waves=orders,provenance=provenance,
        scope='reconstructed expected batch identities; actual exploration choices, logged per-batch counts and final checkpoint RNG all match; not a full gradient recomputation')
    write(target,value);return value

if __name__=='__main__':
    root=Path(sys.argv[1]).resolve();completed='--completed' in sys.argv[2:];results={}
    for job in specification()['training_order']:
        if completed and not (root/'training'/job/'report.json').exists():continue
        value=review(root,job);results[job]={k:v for k,v in value.items() if k not in ('waves','provenance','scope')}
        print('batch order independently verified',job,flush=True)
    if len(results)==6:
        path=root/'batch-order-audit.json'
        if not path.exists():write(path,dict(status='PASS',jobs=results,script_sha256=digest(Path(__file__))))
    print(json.dumps(dict(status='PASS' if len(results)==6 else 'PARTIAL_REVIEW',jobs=results)))
