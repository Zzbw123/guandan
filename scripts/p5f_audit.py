from _bootstrap import ROOT
import sys
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'scripts')]
from pathlib import Path
from experiments.controller_p5f_audit import audit
if __name__=='__main__':
    result=audit(Path(sys.argv[1]).resolve())
    print(result['scope'],result['development_gate'],flush=True)
