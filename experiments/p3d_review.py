"""Post-run controller review, normalizing JSON object keys before comparison.

The frozen experimental auditor computes dicts with int/float keys; JSON stores
object keys as strings. Normalize that representation only; preserve the frozen
arithmetic, independent bootstrap, source checks and disk replay verification.
"""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import p3d_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--write",action="store_true",help="write new audit and negative receipts; refuses overwrite")
    args=parser.parse_args()
    directory=args.directory.resolve()
    original=p3d_audit.summarize
    def normalized_summary(rows):return json.loads(json.dumps(original(rows)))
    with patch.object(p3d_audit,"summarize",normalized_summary):
        result=p3d_audit.audit(directory)
        result["json_object_keys_normalized"]=True
        result["controller_review_sha256"]=sha256(Path(__file__).read_bytes()).hexdigest()
        if args.write:
            with (directory/"controller-audit.json").open("x",encoding="utf-8") as f:
                json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
            runpy.run_path(str(ROOT/"experiments/p3d/controller_negative.py"),run_name="__main__")
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":main()
