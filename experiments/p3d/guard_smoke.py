"""Controller production-CUDA smoke on development-only visible requests."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from guard import GPUInferenceGuard
from guandan.env import HandEnv


def main():
    import benchmark_learning_runtime as old
    import torch
    from guandan_gpu.training import GPUTrainer,score_many
    from guandan_gpu.checkpoint import load_checkpoint
    checkpoint=ROOT/"artifacts/evaluations/p3c-gpu-wave-v1/wave-4"
    pin=sha256((checkpoint/"checkpoint.pt").read_bytes()).hexdigest()
    config=json.loads((checkpoint/"manifest.json").read_text())["config"]
    GPUTrainer(config); trainer=load_checkpoint(checkpoint,pin)
    obs,legal=old.high_branch_fixture()
    expected=int(torch.argmax(score_many(trainer.model,[(obs,legal)],1024)[0]))
    del trainer; torch.cuda.empty_cache()
    start=time.perf_counter()
    with GPUInferenceGuard(checkpoint,pin) as guard:
        action,meta=guard.act(obs,legal)
        assert action==legal[expected] and meta["scored_candidates"]==8769 and meta["device"]["type"]=="cuda"
        env=HandEnv(); visible=env.reset(103090,initial_level=9)
        second,second_meta=guard.act(visible,env.legal_actions(0))
        assert second in env.legal_actions(0)
    assert guard.closed and not guard.is_alive()
    result=dict(status="PASS",checkpoint_sha256=pin,high_candidates=8769,argmax_matches_direct=True,
                high_metadata=meta,development_metadata=second_meta,worker_reaped=True,
                elapsed_s=time.perf_counter()-start,
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    target=ROOT/"artifacts/evaluations/p3d-guard-smoke.json"
    with target.open("x",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
