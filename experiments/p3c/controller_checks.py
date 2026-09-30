"""Controller negative audit and CLI boundaries; original artifacts never modified."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_gpu_training import audit


def main():
    directory = ROOT / "artifacts/evaluations/p3c-gpu-wave-v1"
    original_read = Path.read_text
    mutations = {
        "wrong_total": ("report.json", lambda v: v.update(samples=v["samples"]+1)),
        "truncated_candidates": ("high-branch.json", lambda v: v["scores"][0].pop()),
        "wrong_quantile": ("high-branch.json", lambda v: v["new_latency"].update(p50_ms=-1)),
        "missing_negative": ("negative-audit.json", lambda v: v["cases"].pop("adam")),
        "wrong_resume": ("resume-audit.json", lambda v: v["replayed_rows"][0].update(samples=1)),
    }
    results = {}
    for name, (filename, modify) in mutations.items():
        target = directory / filename
        value = json.loads(original_read(target, encoding="utf-8"))
        modify(value)
        def read(path, *args, **kwargs):
            if path.resolve() == target.resolve(): return json.dumps(value)
            return original_read(path, *args, **kwargs)
        with patch.object(Path, "read_text", read):
            try: audit(directory)
            except ValueError as e: results[name] = dict(rejected=True, reason=str(e))
            else: raise AssertionError(name)
    report_hash = sha256((directory/"report.json").read_bytes()).hexdigest()
    cases = {
        "existing_output": ["--output", str(directory)],
        "reserved_seed": ["--output", str(directory/"should-not-exist"), "--seed-start", "9000000"],
        "excessive_budget": ["--output", str(directory/"should-not-exist"), "--waves", "26"],
        "reused_seed": ["--output", str(directory/"should-not-exist"), "--resume", str(directory/"wave-4"), "--seed-start", "103000"],
    }
    cli = {}
    for name, flags in cases.items():
        child = subprocess.run([sys.executable, str(ROOT/"scripts/train_gpu.py"), *flags],
                               text=True, capture_output=True, timeout=60)
        if child.returncode != 2: raise AssertionError((name, child.returncode, child.stderr))
        cli[name] = dict(exit_code=child.returncode, error=child.stderr.strip().splitlines()[-1])
    assert not (directory/"should-not-exist").exists()
    assert sha256((directory/"report.json").read_bytes()).hexdigest() == report_hash
    result = dict(status="PASS", audit_negative_cases=results, cli_boundaries=cli,
                  original_report_unchanged=True,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    with (directory/"controller-negative.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
