"""Controller audit tamper probes that leave original disk evidence unchanged."""
from hashlib import sha256
import io
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from p3d_audit import audit


def main():
    directory=Path(sys.argv[1]).resolve()
    read_text=Path.read_text;read_bytes=Path.read_bytes;gzip_open=gzip.open
    base_final=json.loads(read_text(directory/"report.json",encoding="utf-8"))
    original_rows=[json.loads(line) for line in read_text(directory/"results.jsonl").splitlines()]
    original_eval=json.loads(read_text(directory/"evaluation-report.json"))
    result={}
    for name in ("missing_trial","duplicate_trial","false_win","wrong_interval","false_gate","incomplete_neural_coverage"):
        replacements={};final=json.loads(json.dumps(base_final));fake_measure=None
        if name in ("missing_trial","duplicate_trial","false_win"):
            rows=json.loads(json.dumps(original_rows))
            if name=="missing_trial":rows.pop()
            if name=="duplicate_trial":rows[-1]=rows[0]
            if name=="false_win":rows[0]["win"]=1-rows[0]["win"]
            replacements["results.jsonl"]=("\n".join(json.dumps(r) for r in rows)+"\n").encode()
        elif name in ("wrong_interval","false_gate"):
            report=json.loads(json.dumps(original_eval))
            if name=="wrong_interval":report["results"]["greedy"]["metrics"]["win_rate"]["ci95"]=[.99,1.]
            else:report["validation_gate"]="FORCED_PROMOTION"
            replacements["evaluation-report.json"]=json.dumps(report).encode()
        else:
            with gzip_open(directory/"measurements.jsonl.gz","rt",encoding="utf-8") as f:
                rows=[json.loads(line) for line in f]
            rows[0]["neural"][0]["scored_candidates"]-=1
            fake_measure="\n".join(json.dumps(r) for r in rows)+"\n"
        for path,data in replacements.items():final["artifact_sha256"][path]=sha256(data).hexdigest()
        replacements["report.json"]=json.dumps(final).encode()
        def rb(path):
            if path.parent==directory and path.name in replacements:return replacements[path.name]
            return read_bytes(path)
        def rt(path,*args,**kwargs):
            if path.parent==directory and path.name in replacements:return replacements[path.name].decode()
            return read_text(path,*args,**kwargs)
        def gz(path,*args,**kwargs):
            if fake_measure is not None and Path(path)==directory/"measurements.jsonl.gz":return io.StringIO(fake_measure)
            return gzip_open(path,*args,**kwargs)
        with patch.object(Path,"read_bytes",rb),patch.object(Path,"read_text",rt),patch("gzip.open",gz):
            try:audit(directory,verify_replays=False)
            except ValueError as e:result[name]=dict(rejected=True,reason=str(e))
            else:raise AssertionError(f"audit accepted {name}")
    output=dict(status="PASS",cases=result,original_report_sha256=sha256(read_bytes(directory/"report.json")).hexdigest(),
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    with (directory/"controller-negative.json").open("x",encoding="utf-8") as f:json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2))


if __name__=="__main__":main()
