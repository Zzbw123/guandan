"""Execute exactly the predeclared 400-hand GPU budget; no validation feedback."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from _bootstrap import ROOT
sys.path.insert(0, str(ROOT/"experiments/p3d"))
import torch
from protocol import CONFIG, TRAIN_SEEDS, specification
from guandan_gpu.training import GPUTrainer
from guandan_gpu.checkpoint import save_checkpoint, runtime


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("directory",type=Path)
    args=p.parse_args(); directory=args.directory
    pre=json.loads((directory/"preregistration.json").read_text(encoding="utf-8"))
    if pre["specification"] != specification(): raise ValueError("preregistration differs")
    output=directory/"training"; output.mkdir()
    trainer=GPUTrainer(CONFIG); start=time.perf_counter()
    initial={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
    samples=0
    with (output/"waves.jsonl").open("x",encoding="utf-8") as log:
        for w in range(100):
            deals=[(TRAIN_SEEDS[i],2+i%13,i%4) for i in range(4*w,4*w+4)]
            row=trainer.train_wave(deals); samples+=row["samples"]
            row["elapsed_s"]=time.perf_counter()-start
            log.write(json.dumps(row,allow_nan=False)+"\n"); log.flush(); os.fsync(log.fileno())
            if (w+1)%25==0: save_checkpoint(trainer,output/f"wave-{w+1}")
            if (w+1)%10==0:
                print(json.dumps(dict(stage="training",waves=w+1,hands=trainer.episodes,
                                      samples=samples,elapsed_s=time.perf_counter()-start)),flush=True)
    changed=any(not torch.equal(initial[k],v.cpu()) for k,v in trainer.model.state_dict().items())
    if not changed: raise ValueError("no model change")
    manifest=json.loads((output/"wave-100/manifest.json").read_text())
    write(output/"report.json",dict(status="PASS",waves=100,hands=400,samples=samples,
          updates=trainer.updates,elapsed_s=time.perf_counter()-start,runtime=runtime(),
          parameters_changed=changed,candidate_sha256=manifest["sha256"]))


if __name__=="__main__": main()
