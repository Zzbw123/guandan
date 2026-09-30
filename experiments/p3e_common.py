"""P3e artifact IO, summaries and offline observation-only behavioral counts."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from guandan.env import HandEnv
from guandan.types import Action
from guandan.evaluation.schedule import deal_hands
from guandan.evaluation.statistics import summarize_matchup
from p3e_metrics import decision_metrics, sum_metrics

ROOT=Path(__file__).resolve().parents[1]
def digest(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def normalized(value): return json.loads(json.dumps(value,allow_nan=False))
def check(condition,message):
    if not condition: raise ValueError(message)

def summarize(rows,trials):
    return {key:summarize_matchup([r for r in rows if r['matchup_id']==key],
               [t for t in trials if t.matchup_id==key],5000,314375)
            for key in sorted({t.matchup_id for t in trials})}

def behavior(replay,trial):
    env=HandEnv.from_hands(deal_hands(trial),trial.level,trial.starting_player)
    check([list(h) for h in env.state.initial_hands]==replay['initial_hands'],'scheduled initial hands')
    counts=Counter(); examples=[]
    for i,step in enumerate(replay['steps']):
        seat=step['player']; obs=env.observe(seat); action=Action.from_dict(step['action'])
        if seat%2==trial.focal_team:
            counts.update(decision_metrics(obs,env.legal_actions(seat),action))
            if len(examples)<3 and counts and len(action.cards)!=len(obs.hand):
                legal=env.legal_actions(seat)
                if any(len(a.cards)==len(obs.hand) for a in legal):
                    examples.append(dict(step=i,player=seat,hand=list(obs.hand),chosen=action.to_dict(),
                        finishing_action=next(a.to_dict() for a in legal if len(a.cards)==len(obs.hand))))
        env.step(seat,action,state_version=step['state_version'])
        check(env.state_digest()==step['digest'],'step digest')
    check(env.state.terminal and env.state_digest()==replay['final_digest'],'terminal replay')
    return dict(trial_id=trial.trial_id,counts=dict(counts),missed_finish_examples=examples)

def ratios(counts):
    pairs={'optional_pass_rate':('optional_passes','optional_pass_opportunities'),
        'teammate_overtake_rate':('teammate_overtakes','teammate_response_opportunities'),
        'missed_finish_rate':('missed_finishes','finish_opportunities'),
        'endgame_optional_pass_rate':('endgame_optional_passes','endgame_optional_pass_opportunities')}
    return {name:dict(numerator=counts.get(n,0),denominator=counts.get(d,0),
             rate=counts.get(n,0)/counts[d] if counts.get(d,0) else None) for name,(n,d) in pairs.items()}
