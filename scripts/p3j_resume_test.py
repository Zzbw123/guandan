"""Separate-process deterministic continuation used only by numerical tests."""
from _bootstrap import ROOT
import sys,json
from pathlib import Path
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.p3j_training import GPUTrainer
from experiments.p3j_checkpoint import load_checkpoint,save_checkpoint
if __name__=='__main__':
    p=Path(sys.argv[1]);GPUTrainer(json.loads((p/'manifest.json').read_text('utf-8'))['config'])
    t=load_checkpoint(p,sys.argv[2]);row=t.train_wave([(109120+i,6+i,i) for i in range(4)])
    out=Path(sys.argv[3]);save_checkpoint(t,out)
    (out.parent/'wave.json').write_text(json.dumps(row),encoding='utf-8')
