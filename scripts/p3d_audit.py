"""Independent disk replay, checkpoint, schedule, and statistical audit."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
import sys
import zipfile
from _bootstrap import ROOT
sys.path.insert(0,str(ROOT/"experiments/p3d"))
from protocol import schedule,specification,summarize
from guandan.env import HandEnv
from guandan.evaluation.schedule import deal_hands


def check(value,label):
    if not value: raise ValueError(label)


def independent_interval(values,levels):
    groups=[[v for v,l in zip(values,levels) if l==level] for level in range(2,15)]
    check(all(len(g)==5 for g in groups),"five independent clusters per level")
    rng=Random(314361); draws=[]
    for _ in range(5000):
        draws.append(sum(g[rng.randrange(5)] for g in groups for _ in range(5))/65)
    draws.sort()
    def percentile(p):
        x=(len(draws)-1)*p; lo=int(x); f=x-lo
        return draws[lo]*(1-f)+draws[min(lo+1,len(draws)-1)]*f
    return [percentile(.025),percentile(.975)]


def audit(directory,verify_replays=True):
    directory=Path(directory)
    read=lambda p:json.loads((directory/p).read_text(encoding="utf-8"))
    pre=read("preregistration.json"); final=read("report.json"); binding=read("evaluation-preregistration.json")
    check(pre["specification"]==specification(),"frozen specification")
    for name,digest in final["artifact_sha256"].items():
        check(sha256((directory/name).read_bytes()).hexdigest()==digest,f"artifact hash {name}")
    check(binding["preregistration_sha256"]==sha256((directory/"preregistration.json").read_bytes()).hexdigest(),"prereg binding")
    registered=pre["source_sha256"]
    with zipfile.ZipFile(directory/"source-snapshot.zip") as z:
        check(set(z.namelist())==set(registered),"source snapshot members")
        for name,digest in registered.items():
            check(sha256((ROOT/name).read_bytes()).hexdigest()==digest==sha256(z.read(name)).hexdigest(),f"source {name}")
    protected={}
    for name,key in (("p2-validation-v1","source_sha256"),("p3a-cpu-v1","source_sha256"),
                     ("p3b-gpu-v1","source_sha256_before"),("p3c-gpu-wave-v1","source_sha256")):
        hashes=json.loads((ROOT/f"artifacts/evaluations/{name}/preregistration.json").read_text())[key]
        for path,digest in hashes.items():check(sha256((ROOT/path).read_bytes()).hexdigest()==digest,f"protected {name}/{path}")
        protected[name]=len(hashes)
    waves=[json.loads(line) for line in (directory/"training/waves.jsonl").read_text().splitlines()]
    check(len(waves)==100,"complete 100 wave budget")
    updates=samples=0
    for i,w in enumerate(waves):
        check(w["wave"]==w["learning_version"]==i+1 and w["behavior_version"]==i and w["episodes"]==4*(i+1),"training synchronization")
        wanted=[(104000+j,2+j%13,j%4) for j in range(4*i,4*i+4)]
        check([tuple(h[k] for k in ("seed","level","starting_player")) for h in w["hands"]]==wanted,"training seeds")
        check(all(h["replay_verified"] is True and 0<h["steps"]==h["samples"]<=1000 for h in w["hands"]),"training hand integrity")
        n=sum(h["samples"] for h in w["hands"]); local=math.ceil(n/256);updates+=local;samples+=n
        check(w["samples"]==n and w["updates"]==updates and math.isfinite(w["mean_loss"]),"training count/loss")
        proof=w["device_proof"]
        check(proof["device"]=="cuda:0" and proof["dtype"]=="torch.float32" and proof["forward_checks"]==local
              and proof["gradient_checks"]==7*local and proof["adam_tensor_checks"]==21*local,"CUDA evidence")
    train=read("training/report.json");manifest=read("training/wave-100/manifest.json")
    check(train["status"]=="PASS" and train["waves"]==100 and train["hands"]==400 and train["samples"]==samples and train["updates"]==updates,"training summary")
    digest=sha256((directory/"training/wave-100/checkpoint.pt").read_bytes()).hexdigest()
    check(digest==binding["candidate_sha256"]==manifest["sha256"]==final["candidate_sha256"]==train["candidate_sha256"],"candidate binding")
    import torch
    payload=torch.load(directory/"training/wave-100/checkpoint.pt",map_location="cpu",weights_only=True)
    check(payload["config"]==specification()["config"] and payload["episodes"]==400 and payload["waves"]==100
          and payload["updates"]==updates and payload["used_deal_seeds"]==list(range(104000,104400)),"checkpoint training state")
    check(all(torch.isfinite(v).all() for v in payload["model"].values()),"finite checkpoint model")
    check(set(payload["optimizer"]["state"])==set(range(7)) and all(s["step"].item()==updates for s in payload["optimizer"]["state"].values()),"Adam update counters")
    rows=[json.loads(line) for line in (directory/"results.jsonl").read_text().splitlines()]
    summary=summarize(rows); evaluation=read("evaluation-report.json")
    for key,value in summary.items():check(evaluation[key]==value,f"recomputed statistics {key}")
    check(evaluation["status"]==final["status"]=="PASS" and final["validation_gate"]==summary["validation_gate"],"final conclusion")
    check(final["model_promoted"] is final["reserved_test_executed"] is False,"scope")
    trials=schedule(); byid={r["trial_id"]:r for r in rows}
    expected={t.trial_id:t for t in trials}
    independent={}
    for opponent in ("greedy","random","team"):
        selected=[r for r in rows if r["matchup_id"].endswith(opponent)]
        values=[sum(r["win"] for r in selected if r["deal_seed"]==seed)/8 for seed in range(201000,201065)]
        interval=independent_interval(values,[2+i%13 for i in range(65)])
        reported=summary["results"][opponent]["metrics"]["win_rate"]
        check(math.isclose(sum(values)/65,reported["estimate"],abs_tol=1e-15) and
              all(math.isclose(a,b,abs_tol=1e-15) for a,b in zip(interval,reported["ci95"])),"independent grouped arithmetic/bootstrap")
        independent[opponent]=dict(wins=sum(r["win"] for r in selected),games=520,groups=65,
                                    estimate=sum(values)/65,ci95=interval,distribution=dict(Counter(values)))
    with gzip.open(directory/"measurements.jsonl.gz","rt",encoding="utf-8") as f:
        measures=[json.loads(line) for line in f]
    check(len(measures)==1560 and {m["trial_id"] for m in measures}==set(expected),"measurement coverage")
    bymeasure={m["trial_id"]:m for m in measures};times=[];inference=[];scored=0
    for m in measures:
        row=byid[m["trial_id"]];n=row["steps"]
        check(len(m["enumeration_ms"])==len(m["decision_ms"])==len(m["candidate_counts"])==n
              and m["replay_verified"] is True,"complete per-turn measurements")
        check(all(type(c) is int and c>0 for c in m["candidate_counts"]) and all(math.isfinite(t) and t>=0 for t in m["enumeration_ms"]+m["decision_ms"]),"finite measurements")
        check(all(t<=2000 for t in m["decision_ms"]),"returned call deadlines")
        steps=[v["step"] for v in m["neural"]]
        check(len(set(steps))==len(steps)>0 and all(type(s) is int and 0<=s<n for s in steps),"neural step indexes")
        for v in m["neural"]:
            check(v["scored_candidates"]==m["candidate_counts"][v["step"]] and v["device"]["type"]=="cuda"
                  and 0<=v["inference_ms"]<=v["roundtrip_ms"]<=2000,"neural coverage/device/deadline")
            times.append(v["roundtrip_ms"]);inference.append(v["inference_ms"]);scored+=v["scored_candidates"]
    replay_count=0
    if verify_replays:
        with gzip.open(directory/"replays.jsonl.gz","rt",encoding="utf-8") as f:
            seen=set()
            for line in f:
                item=json.loads(line);key=item["trial_id"]
                check(key in expected and key not in seen,"replay coverage");seen.add(key)
                t=expected[key];r=byid[key];replay=item["replay"]
                check(replay["initial_hands"]==[list(h) for h in deal_hands(t)] and replay["initial_level"]==t.level
                      and replay["starting_player"]==t.starting_player,"replay schedule origin")
                env=HandEnv.replay(replay);settlement=env.state.settlement
                check(env.state.terminal and env.state_digest()==r["terminal_digest"] and len(replay["steps"])==r["steps"],"disk replay")
                check(int(settlement.winner_team==t.focal_team)==r["win"] and list(settlement.finish_order)==r["finish_order"],"replay settlement")
                model_steps=[i for i,s in enumerate(replay["steps"]) if t.policies[s["player"]]=="dmc"]
                check(model_steps==[v["step"] for v in bymeasure[key]["neural"]],"all model decisions measured")
                replay_count+=1
        check(replay_count==1560,"complete disk replays")
    def stats(values):
        values=sorted(values)
        return dict(count=len(values),p50=values[math.ceil(.5*len(values))-1],p95=values[math.ceil(.95*len(values))-1],maximum=values[-1])
    return dict(status="PASS",training_hands=400,training_samples=samples,training_updates=updates,
                evaluation_games=1560,independent_deals=65,replayed_from_disk=replay_count,
                independent_results=independent,validation_gate=summary["validation_gate"],
                neural_roundtrip_ms=stats(times),neural_inference_ms=stats(inference),scored_candidates=scored,
                protected_sources=protected,registered_sources=len(registered),candidate_sha256=digest,
                auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),model_promoted=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);p.add_argument("--output",type=Path)
    args=p.parse_args();result=audit(args.directory);text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        with args.output.open("x",encoding="utf-8") as f:f.write(text+"\n")
    print(text)
