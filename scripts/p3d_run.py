"""Preregister, train once, pin final candidate, then independently evaluate."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
from _bootstrap import ROOT
sys.path.insert(0,str(ROOT/"experiments/p3d"))
from protocol import specification


def write(path,value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def sources():
    return (sorted((ROOT/"src").rglob("*.py")) + sorted((ROOT/"experiments/p3d").glob("*.py"))
            + sorted((ROOT/"scripts").glob("p3d_*.py")) + sorted((ROOT/"tests").glob("test_p3d_*.py"))
            + [ROOT/"scripts/_bootstrap.py",ROOT/"docs/P3D_PROTOCOL.md",ROOT/"requirements-learning-gpu.lock"])


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(); output=args.output.resolve()
    if output.exists(): p.error("output already exists")
    output.mkdir(parents=True)
    paths=sources(); hashes={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    pre=dict(kind="p3d-preregistration",utc=datetime.now(timezone.utc).isoformat(),
             specification=specification(),source_sha256=hashes,
             candidate_selection="final wave 100 only; no validation feedback or optional stopping")
    write(output/"preregistration.json",pre)
    with zipfile.ZipFile(output/"source-snapshot.zip","x",zipfile.ZIP_DEFLATED) as z:
        for path in paths: z.write(path,path.relative_to(ROOT).as_posix())
    started=time.perf_counter()
    try:
        for stage,script,timeout in (("training","p3d_train.py",1800),("evaluation","p3d_evaluate.py",1800)):
            if stage=="evaluation":
                training=json.loads((output/"training/report.json").read_text())
                digest=sha256((output/"training/wave-100/checkpoint.pt").read_bytes()).hexdigest()
                if training["candidate_sha256"]!=digest: raise ValueError("candidate binding")
                write(output/"evaluation-preregistration.json",dict(utc=datetime.now(timezone.utc).isoformat(),
                      candidate_sha256=digest,preregistration_sha256=sha256((output/"preregistration.json").read_bytes()).hexdigest(),
                      games=1560,independent_deals=65,reserved_test_executed=False))
            # Direct child process has a fixed wall deadline, with partial logs retained.
            with (output/f"{stage}-process.log").open("x",encoding="utf-8") as log:
                child=subprocess.Popen([sys.executable,str(ROOT/"scripts"/script),str(output)],
                                       stdout=log,stderr=subprocess.STDOUT)
                try: code=child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Windows taskkill closes the spawned inference descendant as well.
                    if sys.platform=="win32":
                        subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],capture_output=True,timeout=10)
                    else: child.kill()
                    child.wait(timeout=10)
                    raise TimeoutError(f"{stage} exceeded {timeout}s; process tree terminated")
            if code!=0: raise RuntimeError(f"{stage} process exit {code}; see retained log")
            print(json.dumps(dict(stage=stage,status="PASS",elapsed_s=time.perf_counter()-started)),flush=True)
        if {p:sha256((ROOT/p).read_bytes()).hexdigest() for p in hashes}!=hashes:
            raise ValueError("source changed during experiment")
        report=json.loads((output/"evaluation-report.json").read_text())
        write(output/"report.json",dict(status="PASS",candidate_sha256=report["candidate_sha256"],
              validation_gate=report["validation_gate"],model_promoted=False,reserved_test_executed=False,
              source_unchanged=True,elapsed_s=time.perf_counter()-started,
              artifact_sha256={name:sha256((output/name).read_bytes()).hexdigest() for name in (
              "preregistration.json","evaluation-preregistration.json","training/report.json",
              "training/waves.jsonl","results.jsonl","measurements.jsonl.gz","replays.jsonl.gz",
              "evaluation-report.json","source-snapshot.zip")}))
    except BaseException:
        write(output/"failure.json",dict(status="FAIL",traceback=traceback.format_exc(),elapsed_s=time.perf_counter()-started))
        raise


if __name__=="__main__": main()
