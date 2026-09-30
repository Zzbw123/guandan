"""Evaluate one pinned GPU candidate on the predeclared independent deal groups."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from dataclasses import asdict

from _bootstrap import ROOT
sys.path.insert(0,str(ROOT/"experiments/p3d"))
from guard import GPUInferenceGuard
from protocol import schedule, specification, summarize, POLICY_SEED
from guandan.agents.baselines import make_agent
from guandan.env import HandEnv
from guandan.evaluation.schedule import deal_hands, policy_rng_seed
from guandan.types import Action


def run_trial(guard, trial):
    row=dict(trial_id=trial.trial_id,**asdict(trial),focal_team=trial.focal_team,
             status="error",win=None,team_reward=None,focal_level_gain=None,
             opponent_level_gain=None,finish_order=None,terminal_digest=None,
             steps=0,illegal_actions=0,timeouts=0,error=None)
    row["policies"]=list(row["policies"])
    measures=dict(trial_id=trial.trial_id,enumeration_ms=[],decision_ms=[],candidate_counts=[],
                  neural=[],replay_verified=False)
    replay=None
    try:
        env=HandEnv.from_hands(deal_hands(trial),trial.level,trial.starting_player)
        agents={seat:make_agent(name,policy_rng_seed(POLICY_SEED,trial,seat))
                for seat,name in enumerate(trial.policies) if name!="dmc"}
        while not env.state.terminal:
            if row["steps"]>=1000: raise RuntimeError("step budget exceeded")
            seat=env.state.current_player; obs=env.observe(seat)
            start=perf_counter(); legal=env.legal_actions(seat)
            measures["enumeration_ms"].append((perf_counter()-start)*1000)
            measures["candidate_counts"].append(len(legal))
            start=perf_counter()
            try:
                if trial.policies[seat]=="dmc":
                    action,meta=guard.act(obs,legal)
                    if meta["scored_candidates"]!=len(legal) or meta["device"]["type"]!="cuda":
                        raise RuntimeError("GPU scoring coverage/device mismatch")
                    measures["neural"].append(dict(step=row["steps"],**meta))
                else: action=agents[seat].act(obs,legal)
            except TimeoutError:
                row["timeouts"]+=1; raise
            finally:
                ms=(perf_counter()-start)*1000; measures["decision_ms"].append(ms)
            if type(action) is not Action or action not in legal:
                row["illegal_actions"]+=1; raise RuntimeError("illegal policy action")
            if ms>2000:
                row["timeouts"]+=1; raise TimeoutError("returned call exceeded 2 seconds")
            env.step(seat,action,state_version=obs.state_version); row["steps"]+=1
        replay=env.serialize_replay()
        repeated=HandEnv.replay(replay)
        if repeated.state_digest()!=env.state_digest(): raise RuntimeError("replay mismatch")
        settlement=env.state.settlement; win=settlement.winner_team==trial.focal_team
        row.update(status="ok",win=int(win),team_reward=settlement.team_rewards[trial.focal_team],
                   focal_level_gain=settlement.level_gain if win else 0,
                   opponent_level_gain=0 if win else settlement.level_gain,
                   finish_order=list(settlement.finish_order),terminal_digest=env.state_digest())
        measures["replay_verified"]=True
    except Exception as e:
        row["error"]=dict(type=type(e).__name__,message=str(e))
    return row,measures,replay


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("directory",type=Path)
    args=p.parse_args(); directory=args.directory
    pre=json.loads((directory/"preregistration.json").read_text())
    binding=json.loads((directory/"evaluation-preregistration.json").read_text())
    if pre["specification"]!=specification(): raise ValueError("preregistration mismatch")
    checkpoint=directory/"training/wave-100"
    if sha256((checkpoint/"checkpoint.pt").read_bytes()).hexdigest()!=binding["candidate_sha256"]:
        raise ValueError("candidate changed")
    rows=[]; start=perf_counter()
    with GPUInferenceGuard(checkpoint,binding["candidate_sha256"],startup_timeout=60.,decision_timeout=2.) as guard:
        worker_pid=guard.worker_pid
        with (directory/"results.jsonl").open("x",encoding="utf-8") as result_file, \
             gzip.open(directory/"measurements.jsonl.gz","xt",encoding="utf-8") as measurement_file, \
             gzip.open(directory/"replays.jsonl.gz","xt",encoding="utf-8") as replay_file:
            for trial in schedule():
                row,measure,replay=run_trial(guard,trial)
                result_file.write(json.dumps(row,allow_nan=False)+"\n"); result_file.flush()
                measurement_file.write(json.dumps(measure,allow_nan=False)+"\n"); measurement_file.flush()
                replay_file.write(json.dumps(dict(trial_id=trial.trial_id,replay=replay),allow_nan=False)+"\n"); replay_file.flush()
                rows.append(row)
                if row["status"]!="ok": raise RuntimeError(f"failed trial {trial.trial_id}: {row['error']}")
                if len(rows)%40==0:
                    print(json.dumps(dict(stage="evaluation",completed=len(rows),total=1560,
                                          elapsed_s=perf_counter()-start)),flush=True)
        summary=summarize(rows)
    if guard.is_alive(): raise RuntimeError("inference worker still alive after close")
    report=dict(status="PASS",**summary,candidate_sha256=binding["candidate_sha256"],
                evaluation_games=len(rows),independent_deals=65,elapsed_s=perf_counter()-start,
                inference_worker_pid=worker_pid,inference_worker_closed=guard.closed,
                inference_worker_exitcode=guard.worker_exitcode)
    with (directory/"evaluation-report.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(stage="evaluation",status="PASS",validation_gate=summary["validation_gate"])),flush=True)


if __name__=="__main__": main()
