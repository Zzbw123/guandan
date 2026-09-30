"""Freeze, run, or verify the bounded P3k read-only CUDA diagnostic."""
from pathlib import Path
import sys, os, json, gzip, hashlib, zipfile, math, platform
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from dataclasses import asdict
from experiments.p3e_common import read,write,digest,check
from guandan.env import HandEnv
from guandan.types import Action
from experiments.p3j_protocol import schedule

INITS=(314380,314381,314382)
OUT=ROOT/'artifacts/evaluations/p3k-gradients-v1'

def paths():
    models={};sources={}
    for seed in INITS:
        base=ROOT/f'artifacts/evaluations/p3f-teacher-v1/training/teacher-{seed}/phase1'
        models[f'teacher-{seed}']=base/'raw.pt'
        sources[f'teacher-{seed}']=base/'replays.jsonl.gz'
        for arm in ('absolute','centered'):
            job=f'{arm}-{seed}';base=ROOT/'artifacts/evaluations/p3j-centered-v1'
            models[job]=base/'training'/job/'final/checkpoint.pt'
            sources[job]=base/'evaluations'/job/'replays.jsonl.gz'
    return models,sources

def jsonnorm(x):return json.loads(json.dumps(x))
def hashvalue(x):return hashlib.sha256(json.dumps(jsonnorm(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def selected_games():
    trials={t.trial_id:t for t in schedule() if t.deal_seed in range(108100,108113) and t.rotation==0 and t.swap==0}
    for source,path in paths()[1].items():
        count=0
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if source.startswith('teacher'):
                    if row['seed'] not in range(100000,100013):continue
                    identity=str(row['seed']);team=None;seed=row['seed']
                else:
                    if row['trial_id'] not in trials:continue
                    t=trials[row['trial_id']];identity=t.trial_id;team=t.focal_team;seed=t.deal_seed
                count+=1
                yield source,identity,seed,team,row['replay']
        check(count==13,'exactly 13 selected source games')

def samples():
    for source,game,seed,team,replay in selected_games():
        final=HandEnv.replay(replay)
        check(final.state.terminal and final.state_digest()==replay['final_digest'],'full terminal replay')
        eligible=[i for i,s in enumerate(replay['steps']) if team is None or s['player']%2==team]
        check(len(eligible)>=4,'enough eligible observations')
        wanted={eligible[(len(eligible)-1)//3],eligible[2*(len(eligible)-1)//3]}
        check(len(wanted)==2,'unique prespecified positions')
        env=HandEnv.from_hands(replay['initial_hands'],replay['initial_level'],replay['starting_player'])
        for step,s in enumerate(replay['steps']):
            action=Action.from_dict(s['action']);obs=env.observe(s['player'])
            if step in wanted:
                legal=env.legal_actions(obs.player_id)
                check(action in legal,'executed action in complete candidates')
                identity=dict(source=source,game=game,seed=seed,step=step,player=obs.player_id,level=obs.level,
                    reward=final.state.settlement.team_rewards[obs.player_id%2],eligible_count=len(eligible),
                    candidate_count=len(legal),candidate_sha256=hashvalue([a.to_dict() for a in legal]),
                    observation_sha256=hashvalue(asdict(obs)),action=s['action'],final_digest=replay['final_digest'])
                yield identity,obs,legal,action
            env.step(obs.player_id,action,state_version=s['state_version'])
        check(env.state_digest()==replay['final_digest'],'selection replay digest')

def setup():
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    check(torch.cuda.is_available(),'CUDA required')
    return dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),deterministic=torch.are_deterministic_algorithms_enabled(),threads=torch.get_num_threads(),
        tf32=False,cublas=os.environ['CUBLAS_WORKSPACE_CONFIG'])

def load_models():
    import torch
    from guandan.learning.model import DMCNetwork
    nets={}
    for name,path in paths()[0].items():
        manifest=read(path.parent/'manifest.json')
        check(digest(path)==manifest['sha256'],'model manifest SHA')
        data=torch.load(path,map_location='cpu',weights_only=True)
        model=DMCNetwork().cuda();model.load_state_dict(data['model']);model.eval();nets[name]=model
    return nets

def bindings(out):
    pre=read(out/'preregistration.json')
    for section in ('source_sha256','input_sha256'):
        for p,h in pre[section].items():check(digest(ROOT/p)==h,'frozen binding '+p)
    return pre

def freeze(out):
    check(not out.exists(),'fresh output directory required')
    models,sources=paths();inputs=list(models.values())+list(sources.values())+[p.parent/'manifest.json' for p in models.values()]
    inputs += [ROOT/'artifacts/evaluations/p3j-centered-v1/delivery-receipt.json']
    code=list((ROOT/'src').rglob('*.py'))+list((ROOT/'experiments').glob('p3k*.py'))+list((ROOT/'tests').glob('test_p3k*.py'))
    code += [ROOT/'experiments'/n for n in ('p3e_common.py','p3e_metrics.py','p3f_teacher.py','p3f_training.py','p3h_objective.py','p3j_objective.py','p3j_protocol.py')]
    code += [ROOT/'docs/P3K_PROTOCOL.md']
    out.mkdir(parents=True);(out/'previous-status.md').write_bytes((ROOT/'docs/STATUS.md').read_bytes())
    write(out/'preregistration.json',dict(version='gd-p3k-gradient-diagnostic-v1',expected_sources=9,expected_games=117,
        expected_states=234,expected_models=9,expected_probes=2106,weight=.1,zero_norm=1e-12,
        selection='13 fixed games/source; floor((n-1)/3) and floor(2*(n-1)/3); includes singletons',
        gradient_rtol=3e-4,gradient_atol=3e-6,loss_rtol=1e-5,loss_atol=2e-6,score_atol=2e-6,
        source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(code))},
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(inputs))},
        previous_status_sha256=digest(out/'previous-status.md'),model_promoted=False,reserved_test_executed=False,
        skill='reproducibility-skill; preserve source/input hashes and rerunnable entrypoint'))
    with zipfile.ZipFile(out/'source-snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(code)):z.write(p,p.relative_to(ROOT).as_posix())
    print(json.dumps(dict(status='FROZEN',sources=len(set(code)),inputs=len(set(inputs)))))

def run(out):
    import torch
    from experiments.p3k_core import production,oracle,summarize_probe
    from experiments.p3f_training import model_digest
    pre=bindings(out);runtime=setup();nets=load_models();before={k:model_digest(v) for k,v in nets.items()}
    selected=list(samples());check(len(selected)==234,'complete prespecified sample')
    write(out/'selection.json',[r[0] for r in selected])
    maxgrad=maxscore=maxloss=0.;count=0
    with (out/'probes.jsonl').open('x',encoding='utf-8') as f:
        for ix,(identity,obs,legal,action) in enumerate(selected):
            for name,model in nets.items():
                grads,losses,q,target=production(model,obs,legal,action,identity['reward'])
                og,ol,oq,ot=oracle(model,obs,legal,action,identity['reward'])
                for a,b in zip(grads,og):
                    torch.testing.assert_close(a,b,rtol=pre['gradient_rtol'],atol=pre['gradient_atol'])
                    maxgrad=max(maxgrad,(a-b).abs().max().item())
                for a,b in zip(losses,ol):check(math.isclose(a,b,rel_tol=pre['loss_rtol'],abs_tol=pre['loss_atol']),'dense independent loss')
                err=(q-oq).abs().max().item();maxscore=max(maxscore,err);check(err<=pre['score_atol'],'complete direct q')
                maxloss=max(maxloss,max(abs(a-b) for a,b in zip(losses,ol)))
                row=dict(identity=identity,model=name,scores=q.cpu().tolist(),targets=target.cpu().tolist(),metrics=summarize_probe(model,grads,losses,q,target,legal.index(action)),
                    oracle_gradient_max_abs=max((a-b).abs().max().item() for a,b in zip(grads,og)))
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();count+=1
            if (ix+1)%26==0:print(json.dumps(dict(source=identity['source'],states=ix+1,probes=count)),flush=True)
    after={k:model_digest(v) for k,v in nets.items()};check(before==after,'read-only model state');bindings(out)
    check(count==2106,'complete probe count')
    write(out/'run-receipt.json',dict(status='RUN_COMPLETE_PENDING_AUDIT',runtime=runtime,states=len(selected),probes=count,
        dense_gradient_comparisons=count*3,model_before=before,model_after=after,max_gradient_abs_error=maxgrad,
        max_score_abs_error=maxscore,max_loss_abs_error=maxloss,model_promoted=False,reserved_test_executed=False,
        new_training_hands=0,new_evaluation_games=0,preregistration_sha256=digest(out/'preregistration.json'),
        selection_sha256=digest(out/'selection.json'),probes_sha256=digest(out/'probes.jsonl')))
    print(json.dumps(read(out/'run-receipt.json')),flush=True)

if __name__=='__main__':
    out=OUT
    if sys.argv[1:] == ['--freeze']:freeze(out)
    elif sys.argv[1:] == ['--run']:run(out)
    else:raise SystemExit('use --freeze or --run')
