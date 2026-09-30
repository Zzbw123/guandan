"""Exercise the independent P3b1 disk auditor against tampered copies."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_runtime_probe import audit


def run(source):
    original = json.loads((source / "report.json").read_text(encoding="utf-8"))
    cases = {
        "score_value": lambda r: r["results"]["score"]["fixtures"][0]["scores_by_device"]["cuda"].__setitem__(0, 99),
        "missing_candidates": lambda r: r["results"]["score"]["fixtures"][1]["coverage"]["cuda"]["chunk_lengths"].pop(),
        "percentile": lambda r: r["results"]["minibatch"]["minibatches"]["64"]["cpu"]["latency"].__setitem__("p95_ms", 999),
        "duplicate_hand": lambda r: r["results"]["workers"]["workers"]["4"]["hands"].__setitem__(1, copy.deepcopy(r["results"]["workers"]["workers"]["4"]["hands"][0])),
        "model_identity": lambda r: r["results"]["workers"]["workers"]["2"]["hands"][0].__setitem__("model_state_sha256", "0"*64),
        "wall_time": lambda r: r["results"]["episodes"]["devices"]["cpu"]["hands"][0].__setitem__("total_ms", 0),
    }
    rejected = {}
    with tempfile.TemporaryDirectory(prefix="gd-p3b-negative-") as temp:
        directory = Path(temp)
        for name in ("preregistration.json", "source-snapshot.zip"):
            shutil.copyfile(source / name, directory / name)
        for name, mutation in cases.items():
            changed = copy.deepcopy(original)
            mutation(changed)
            (directory / "report.json").write_text(json.dumps(changed), encoding="utf-8")
            try:
                audit(directory)
            except ValueError as error:
                rejected[name] = str(error)
            else:
                raise AssertionError(f"tampered receipt accepted: {name}")
    return dict(status="PASS", rejected=rejected,
                source_report_sha256=hashlib.sha256((source / "report.json").read_bytes()).hexdigest(),
                test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)
