"""Trace actual CUDA DMC target assignment and backward gradients on four dev hands."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from p3e_common import check,digest,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'docs/P3F_DIAGNOSTIC_PROTOCOL.md']
    hashes={x.relative_to(ROOT).as_posix():digest(x) for x in paths}
    spec=dict(seed=314380,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4)
    write(out/'preregistration.json',dict(utc=datetime.now(timezone.utc).isoformat(),config=spec,
        deals=[[108500+i,2+i,i] for i in range(4)],source_sha256=hashes,gradient_tolerance=1e-6))
    import torch
    import guandan_gpu.training as training
    from guandan_gpu.checkpoint import runtime
    from guandan.env import HandEnv
    from guandan.learning.encoding import encode_observation,encode_action
    recorded=[];forward_keys=[];observed=Counter();expected=Counter();gradients=[];batch_rows=[]
    def encoded_key(obs,action):
        floats=encode_observation(obs)+encode_action(action)
        return sha256(struct.pack('<'+str(len(floats))+'f',*floats)).hexdigest()
    class RecordingEnv:
        def __init__(self):self.env=HandEnv();self.decisions=[];recorded.append(self)
        def __getattr__(self,key):return getattr(self.env,key)
        def step(self,player,action,state_version):
            obs=self.env.observe(player);self.decisions.append((encoded_key(obs,action),player%2))
            return self.env.step(player,action,state_version=state_version)
        @staticmethod
        def replay(value):return HandEnv.replay(value)
    trainer=training.GPUTrainer(spec)
    original=torch.nn.functional.mse_loss
    def before(module,inputs):
        states,actions=inputs
        check(states.device.type==actions.device.type=='cuda','actual training forward CUDA')
        for state,action in zip(states.detach().cpu(),actions.detach().cpu()):
            data=bytes(torch.cat((state,action)).contiguous().view(torch.uint8).tolist())
            forward_keys.append(sha256(data).hexdigest())
    hook=trainer.model.register_forward_pre_hook(before)
    def traced(predictions,targets,*args,**kwargs):
        if not expected:
            for record in recorded:
                check(record.env.state.terminal,'all waves terminate before learning')
                for key,team in record.decisions:expected[(key,record.env.state.settlement.team_rewards[team])]+=1
        n=len(targets);check(len(forward_keys)==n,'aligned actual model inputs')
        labels=targets.detach().cpu().tolist()
        for key,label in zip(forward_keys,labels):
            check(label in (-1.,1.),'terminal reward values');observed[(key,int(label))]+=1
        forward_keys.clear()
        check(targets.device.type==predictions.device.type=='cuda','MSE CUDA')
        wanted=2*(predictions.detach()-targets.detach())/n
        def gradient(grad):
            error=float((grad-wanted).abs().max().item());check(error<=1e-6,'actual MSE gradient')
            gradients.append(dict(samples=n,max_abs_error=error,device=str(grad.device)))
        predictions.register_hook(gradient)
        batch_rows.append(dict(samples=n,positive=sum(x==1 for x in labels),negative=sum(x==-1 for x in labels)))
        return original(predictions,targets,*args,**kwargs)
    try:
        with patch.object(training,'HandEnv',RecordingEnv),patch.object(torch.nn.functional,'mse_loss',traced):
            row=trainer.train_wave([(108500+i,2+i,i) for i in range(4)])
        check(observed==expected,'actual state/action/team-return multiset')
        check(sum(observed.values())==row['samples'] and len(gradients)==row['updates'],'sample/update/gradient coverage')
        for name,value in hashes.items():check(digest(ROOT/name)==value,'source unchanged')
        report=dict(status='PASS',wave=row,actual_samples=sum(observed.values()),
            actual_input_target_multiset_matched=True,gradient_batches=gradients,batches=batch_rows,
            runtime=runtime(),model_promoted=False,reserved_test_executed=False)
        write(out/'report.json',report)
        write(out/'target-multiset.json',[dict(input_sha256=k[0],reward=k[1],count=v) for k,v in sorted(expected.items())])
        print(json.dumps(dict(status='PASS',samples=row['samples'],updates=row['updates'])))
    finally:hook.remove()
if __name__=='__main__':main()
